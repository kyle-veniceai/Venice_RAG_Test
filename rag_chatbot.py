import os
import requests
from pinecone import Pinecone
from typing import List, Dict, Any
import textwrap
import uuid
from pinecone import ServerlessSpec
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_env_variable(var_name):
    """
    Get an environment variable or raise an error if not found.
    """
    value = os.environ.get(var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable {var_name} not set")
    return value

# Required environment variables for API keys and URLs (no fallbacks for security)
EMBEDDING_API_KEY = get_env_variable("EMBEDDING_API_KEY")
EMBEDDING_API_URL = get_env_variable("EMBEDDING_API_URL")
PINECONE_API_KEY = get_env_variable("PINECONE_API_KEY")
VENICE_API_KEY = get_env_variable("VENICE_API_KEY")
VENICE_API_URL = get_env_variable("VENICE_API_URL")

# Configuration variables with safe defaults
VECTOR_DIMENSION = get_env_variable("VECTOR_DIMENSION")
CHUNK_SIZE = get_env_variable("CHUNK_SIZE")
CHUNK_OVERLAP = get_env_variable("CHUNK_OVERLAP")
TOP_K_RESULTS = get_env_variable("TOP_K_RESULTS")
INDEX_NAME = get_env_variable("INDEX_NAME")
metric = get_env_variable("METRIC")
dimensions = VECTOR_DIMENSION
index_name = INDEX_NAME

# Initialize Pinecone
def init_pinecone():
    """Initialize Pinecone and return the index object"""
    try:
        # Configure Client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check for Name
        existing_indexes = [
            index_info["name"] for index_info in pc.list_indexes()
        ]

        if index_name not in existing_indexes:
            print(f"Creating new index '{index_name}'")
            
            # Create with direct parameters and no spec (simplest approach)
            pc.create_index(
                name=index_name,
                dimension=dimensions,
                metric=metric,
                spec=spec
            )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            while not pc.describe_index(index_name).status['ready']:
                print(f"Index not ready")
                time.sleep(5)
                        
        # Connect to Index
        index = pc.Index(index_name)
        time.sleep(1)

        # View Index Stats
        index.describe_index_stats()
        
        # Return the index object
        return index

    except Exception as e:
        print(f"Error connecting to Pinecone: {str(e)}")
        return None  # Return None instead of a tuple
    
# Chunking Function
def chunk_text(text: str) -> List[str]:
    """Split text into chunks with specified size and overlap."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)
        
        # If we're not at the beginning, move start back by overlap
        if start > 0:
            start = start - CHUNK_OVERLAP
        
        # Create the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Set next start position
        start = end
    
    return chunks


# Get Embedding Function
def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text using the embedding API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EMBEDDING_API_KEY}"
    }
    
    data = {"input": text, "model": "text-embedding-bge-m3", "encoding_format": "float"}
    
    try:
        response = requests.post(EMBEDDING_API_URL, json=data, headers=headers)
        response.raise_for_status()
        
        # Parse the response to extract the embedding vector
        result = response.json()
        # Assuming the API returns embedding in a format like {"embedding": [0.1, 0.2, ...]}
        # Adjust this based on your actual API response structure
        embedding = result.get("data", [{}])[0].get("embedding", [])
        
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []


# Store Documents Function
def store_documents(documents, index):
    """
    Process and store documents in Pinecone using the updated API.
    
    Args:
        documents: List of document dictionaries with 'text' and 'metadata' keys
        index: Pinecone index object
    """
    if index is None:
        print("Cannot store documents: Pinecone index is not initialized")
        return
    
    vectors_to_upsert = []
    
    for doc in documents:
        text = doc['text']
        metadata = doc.get('metadata', {})
        
        # Chunk the document
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            # Get embedding for the chunk
            embedding = get_embedding(chunk)
            
            if not embedding:
                continue
            
            # Create a unique ID for this chunk
            chunk_id = str(uuid.uuid4())
            
            # Prepare metadata for this chunk
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'content': chunk  # Store the text in metadata for retrieval
            }
            
            # Add to vectors to upsert
            vectors_to_upsert.append({
                'id': chunk_id,
                'values': embedding,
                'metadata': chunk_metadata
            })
            
            # Upsert in batches of 100 to avoid request size limits
            if len(vectors_to_upsert) >= 100:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
    
    # Upsert any remaining vectors
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)


# Query Function
def query_knowledge_base(query, index):
    """
    Query the knowledge base with a user question using updated API.
    
    Args:
        query: User's question
        index: Pinecone index object
        
    Returns:
        List of relevant document chunks
    """
    if index is None:
        print("Cannot query knowledge base: Pinecone index is not initialized")
        return []
    
    print(f"Querying knowledge base for: '{query}'")
    
    # Check if index is empty
    try:
        stats = index.describe_index_stats()
        vector_count = stats.total_vector_count
        print(f"Index stats: {stats}")
        print(f"Total vectors in index: {vector_count}")
        
        if vector_count == 0:
            print("Knowledge base is empty. No documents have been indexed.")
            return []
    except Exception as e:
        print(f"Error checking index stats: {e}")
    
    # Get the embedding for the query
    query_embedding = get_embedding(query)
    
    if not query_embedding:
        print("Failed to generate embedding for query")
        return []
    
    print(f"Generated embedding of length: {len(query_embedding)}")
    
    # Query Pinecone
    try:
        query_results = index.query(
            vector=query_embedding,
            top_k=TOP_K_RESULTS,
            include_metadata=True
        )
        
        print(f"Query returned {len(query_results.matches)} matches")
        
        # Extract the text from the metadata of each result
        results = []
        for i, match in enumerate(query_results.matches):
            if match.metadata and 'content' in match.metadata:
                results.append({
                    'content': match.metadata['content'],
                    'score': match.score,
                    'metadata': {k:v for k,v in match.metadata.items() if k != 'content'}
                })
                print(f"Match {i+1}: Score {match.score:.4f} - Content snippet: {match.metadata['content'][:50]}...")
            else:
                print(f"Match {i+1}: Score {match.score:.4f} - No content in metadata")
        
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

# LLM Query Function
def query_llm(prompt: str, context: List[Dict]) -> str:
    """
    Query the LLM with the user prompt and retrieved context.
    """
    # Format the context
    formatted_context = "\n\n".join([
        f"Document (relevance: {item['score']:.2f}):\n{item['content']}"
        for item in context
    ])
    
    # Construct the system message with context
    system_message = f"""You are a helpful assistant that answers questions based on the provided context.

Context information:
{formatted_context}

Please answer the question based on the provided context. If the answer is not in the context, say so.
"""
    
    # Call the Venice.ai API with llama-3.3-70b model
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VENICE_API_KEY}"
    }
    
    data = {
        "model": "llama-3.3-70b",
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "venice_parameters": {
            "include_venice_system_prompt": True
        },
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(VENICE_API_URL, json=data, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the assistant's message from the response
        assistant_message = result.get("choices", [{}])[0].get("message", {}).get("content", 
                            "Sorry, I couldn't generate a response.")
        
        return assistant_message
    
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "Sorry, there was an error processing your request."


# Main RAG Chatbot Function
def rag_chatbot(query: str) -> str:
    """
    Main RAG chatbot function that processes a user query.
    """
    try:
        # Initialize Pinecone
        index = init_pinecone()
        
        # If Pinecone initialization failed, still proceed but without context
        if index is None:
            print("Warning: Could not connect to Pinecone. Proceeding without context.")
            context = []
        else:
            # Query the knowledge base
            context = query_knowledge_base(query, index)
        
        # Even if no relevant context found, still query the LLM
        if not context:
            print("No relevant context found. Querying LLM without context.")
            response = query_llm_without_context(query)
        else:
            # Query the LLM with the user prompt and retrieved context
            response = query_llm(query, context)
        
        return response
    except Exception as e:
        print(f"Error in rag_chatbot: {e}")
        return "I'm sorry, but I encountered an error while processing your request. Please try again later."

# New function for querying LLM without context
def query_llm_without_context(prompt: str) -> str:
    """
    Query the LLM with just the user prompt, without any context.
    """
    # Call the Venice.ai API with llama-3.3-70b model
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VENICE_API_KEY}"
    }
    
    data = {
        "model": "llama-3.3-70b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions to the best of your ability."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "venice_parameters": {
            "include_venice_system_prompt": True
        },
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(VENICE_API_URL, json=data, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the assistant's message from the response
        assistant_message = result.get("choices", [{}])[0].get("message", {}).get("content", 
                            "Sorry, I couldn't generate a response.")
        
        return assistant_message
    
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "Sorry, there was an error processing your request."

# Function to add documents to the knowledge base
def add_to_knowledge_base(documents: List[Dict[str, str]]):
    """
    Add documents to the knowledge base.
    
    Args:
        documents: List of document dictionaries with 'text' and 'metadata' keys
    """
    index = init_pinecone()
    store_documents(documents, index)
    return f"Added {len(documents)} documents to the knowledge base."


# Example usage for adding documents to the knowledge base
def example_add_documents():
    documents = [
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "example", "category": "test"}
        },
        {
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, " +
                   "as opposed to natural intelligence displayed by animals including humans.",
            "metadata": {"source": "wiki", "category": "AI"}
        }
    ]
    
    return add_to_knowledge_base(documents)


# Example usage for querying the chatbot
def example_query():
    query = "What is artificial intelligence?"
    return rag_chatbot(query)


# CLI interface for the chatbot
def chat_cli():
    print("RAG Chatbot Initialized. Type 'exit' to quit.")
    print("Type 'add' to add documents to the knowledge base.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        elif user_input.lower() == 'add':
            file_path = input("Enter file path to add: ")
            try:
                with open(file_path, 'r') as f:
                    text = f.read()
                
                source = input("Enter source name: ")
                category = input("Enter category: ")
                
                documents = [{
                    "text": text,
                    "metadata": {"source": source, "category": category}
                }]
                
                result = add_to_knowledge_base(documents)
                print(f"Result: {result}")
            
            except Exception as e:
                print(f"Error: {e}")
        
        else:
            response = rag_chatbot(user_input)
            print(f"\nChatbot: {response}")


if __name__ == "__main__":
    chat_cli()