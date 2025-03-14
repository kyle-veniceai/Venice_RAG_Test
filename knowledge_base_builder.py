import os
import glob
import argparse
from typing import List, Dict, Any
import json
from tqdm import tqdm
import pandas as pd
from rag_chatbot import chunk_text, get_embedding, init_pinecone, store_documents

def process_text_file(file_path: str) -> Dict[str, Any]:
    """Process a text file and return a document dictionary."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Extract filename for metadata
    filename = os.path.basename(file_path)
    
    return {
        "text": text,
        "metadata": {
            "source": filename,
            "file_type": "text",
            "path": file_path
        }
    }

def process_csv_file(file_path: str, text_column: str = None) -> List[Dict[str, Any]]:
    """Process a CSV file and return a list of document dictionaries."""
    df = pd.read_csv(file_path)
    documents = []
    
    # If no text column specified, try to find one
    if text_column is None:
        # Look for common text column names
        potential_columns = ['text', 'content', 'description', 'body']
        for col in potential_columns:
            if col in df.columns:
                text_column = col
                break
        
        # If still None, use the first column
        if text_column is None and len(df.columns) > 0:
            text_column = df.columns[0]
    
    # Extract filename for metadata
    filename = os.path.basename(file_path)
    
    # Convert each row to a document
    for idx, row in df.iterrows():
        # Skip if text column doesn't exist or value is empty
        if text_column not in df.columns or pd.isna(row[text_column]):
            continue
        
        text = str(row[text_column])
        
        # Create metadata from other columns
        metadata = {
            "source": filename,
            "file_type": "csv",
            "row_id": idx,
            "path": file_path
        }
        
        # Add other columns to metadata
        for col in df.columns:
            if col != text_column and not pd.isna(row[col]):
                metadata[col] = str(row[col])
        
        documents.append({
            "text": text,
            "metadata": metadata
        })
    
    return documents

def process_json_file(file_path: str, text_key: str = None) -> List[Dict[str, Any]]:
    """Process a JSON file and return a list of document dictionaries."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
    
    documents = []
    filename = os.path.basename(file_path)
    
    # Handle different JSON structures
    if isinstance(data, list):
        # List of objects
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                # If text_key is specified, use it
                if text_key and text_key in item:
                    text = str(item[text_key])
                # Otherwise try to find a suitable text field
                else:
                    potential_keys = ['text', 'content', 'description', 'body']
                    text = None
                    for key in potential_keys:
                        if key in item:
                            text = str(item[key])
                            break
                
                # If we found text, create a document
                if text:
                    # Create metadata from other fields
                    metadata = {
                        "source": filename,
                        "file_type": "json",
                        "item_id": idx,
                        "path": file_path
                    }
                    
                    # Add other fields to metadata
                    for key, value in item.items():
                        if (text_key and key != text_key) or (not text_key and key not in potential_keys):
                            if isinstance(value, (str, int, float, bool)):
                                metadata[key] = value
                    
                    documents.append({
                        "text": text,
                        "metadata": metadata
                    })
    
    elif isinstance(data, dict):
        # Single object or nested structure
        # Try to find text content
        if text_key and text_key in data:
            text = str(data[text_key])
            
            metadata = {
                "source": filename,
                "file_type": "json",
                "path": file_path
            }
            
            documents.append({
                "text": text,
                "metadata": metadata
            })
        else:
            # Look for text fields or process as a collection of documents
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 50:  # Arbitrary length to identify text fields
                    documents.append({
                        "text": value,
                        "metadata": {
                            "source": filename,
                            "file_type": "json",
                            "field": key,
                            "path": file_path
                        }
                    })
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    # List of strings
                    documents.append({
                        "text": " ".join(value),
                        "metadata": {
                            "source": filename,
                            "file_type": "json",
                            "field": key,
                            "path": file_path
                        }
                    })
    
    return documents

def process_directory(directory_path: str, 
                      file_types: List[str] = None, 
                      recursive: bool = True,
                      text_column: str = None,
                      text_key: str = None) -> List[Dict[str, Any]]:
    """
    Process all files in a directory.
    
    Args:
        directory_path: Path to the directory
        file_types: List of file extensions to process (e.g., ['.txt', '.csv'])
        recursive: Whether to process subdirectories
        text_column: Column name for CSV files
        text_key: Key name for JSON files
        
    Returns:
        List of document dictionaries
    """
    if file_types is None:
        file_types = ['.txt', '.csv', '.json']
    
    documents = []
    
    # Create the file pattern based on recursivity
    pattern = os.path.join(directory_path, '**' if recursive else '', '*')
    
    # Process all files
    for file_path in tqdm(glob.glob(pattern, recursive=recursive)):
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext not in file_types:
                continue
            
            try:
                if ext == '.txt':
                    documents.append(process_text_file(file_path))
                elif ext == '.csv':
                    documents.extend(process_csv_file(file_path, text_column))
                elif ext == '.json':
                    documents.extend(process_json_file(file_path, text_key))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return documents

def main():
    parser = argparse.ArgumentParser(description='Build knowledge base from files')
    parser.add_argument('--dir', type=str, help='Directory containing files to process')
    parser.add_argument('--file', type=str, help='Single file to process')
    parser.add_argument('--types', type=str, nargs='+', default=['.txt', '.csv', '.json'], 
                        help='File types to process (e.g., .txt .csv .json)')
    parser.add_argument('--recursive', action='store_true', help='Process subdirectories recursively')
    parser.add_argument('--text-column', type=str, help='Column name for text in CSV files')
    parser.add_argument('--text-key', type=str, help='Key name for text in JSON files')
    
    args = parser.parse_args()
    
    documents = []
    
    if args.dir:
        print(f"Processing directory: {args.dir}")
        documents.extend(process_directory(
            args.dir, 
            args.types, 
            args.recursive,
            args.text_column,
            args.text_key
        ))
    
    if args.file:
        print(f"Processing file: {args.file}")
        ext = os.path.splitext(args.file)[1].lower()
        
        if ext == '.txt':
            documents.append(process_text_file(args.file))
        elif ext == '.csv':
            documents.extend(process_csv_file(args.file, args.text_column))
        elif ext == '.json':
            documents.extend(process_json_file(args.file, args.text_key))
    
    if not documents:
        print("No documents found to process.")
        return
    
    print(f"Found {len(documents)} documents to process.")
    
    # Initialize Pinecone
    print("Initializing Pinecone...")
    index = init_pinecone()
    
    # Store documents
    print("Processing and storing documents...")
    store_documents(documents, index)
    
    print("Knowledge base building complete!")

if __name__ == "__main__":
    main()