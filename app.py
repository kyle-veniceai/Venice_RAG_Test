from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from rag_chatbot import rag_chatbot, add_to_knowledge_base, init_pinecone
from knowledge_base_builder import process_text_file, process_csv_file, process_json_file
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'csv', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Call the RAG chatbot function
        logger.info(f"Processing chat request: {user_message[:50]}...")
        response = rag_chatbot(user_message)
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if the user submitted an empty form
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        logger.info(f"File saved to {file_path}")
        
        # Process the file based on its extension
        try:
            ext = os.path.splitext(filename)[1].lower()
            documents = []
            
            if ext == '.txt':
                document = process_text_file(file_path)
                documents.append(document)
                logger.info(f"Processed text file: {len(document['text'])} characters")
            elif ext == '.csv':
                text_column = request.form.get('text_column')
                csv_documents = process_csv_file(file_path, text_column)
                documents.extend(csv_documents)
                logger.info(f"Processed CSV file: {len(csv_documents)} rows")
            elif ext == '.json':
                text_key = request.form.get('text_key')
                json_documents = process_json_file(file_path, text_key)
                documents.extend(json_documents)
                logger.info(f"Processed JSON file: {len(json_documents)} items")
            
            logger.info(f"Total documents to add: {len(documents)}")
            
            # Filter out empty documents
            valid_documents = [doc for doc in documents if doc.get('text')]
            logger.info(f"Valid documents after filtering: {len(valid_documents)}")
            
            if not valid_documents:
                return jsonify({'error': 'No valid text found in the uploaded file'}), 400
            
            # Add documents to knowledge base
            result = add_to_knowledge_base(valid_documents)
            
            # Verify documents were added
            index = init_pinecone()
            if index:
                try:
                    stats = index.describe_index_stats()
                    logger.info(f"Index stats after upload: {stats}")
                except Exception as e:
                    logger.error(f"Error checking index stats: {e}")
            
            return jsonify({
                'success': True, 
                'message': f'File uploaded and processed. {result}'
            })
            
        except Exception as e:
            logger.error("Error processing file", exc_info=True)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/health')
def health_check():
    """Health check endpoint for container orchestration systems"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    
    # In production, the app will be run with gunicorn
    # This is only used for local development
    app.run(host='0.0.0.0', port=port, debug=False)