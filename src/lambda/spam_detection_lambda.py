import json
import pickle
import re
import os
import boto3
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3 = boto3.client('s3')

# Config - Replace with your S3 bucket name from the previous upload
# This bucket should contain the model files
BUCKET_NAME = 'spam-detection-model-1742984560'  # Update this with the actual bucket name

def clean_text(text):
    """
    Clean and normalize text data
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def download_model_from_s3():
    """
    Download model files from S3 to Lambda's /tmp directory
    """
    model_files = [
        'spam_detection_model.pkl',
        'spam_detection_model_vectorizer.pkl',
        'spam_detection_model_label_encoder.pkl'
    ]
    
    for file_name in model_files:
        local_file_path = f'/tmp/{file_name}'
        if not os.path.exists(local_file_path):
            logger.info(f"Downloading {file_name} from S3")
            try:
                s3.download_file(BUCKET_NAME, file_name, local_file_path)
                logger.info(f"Successfully downloaded {file_name}")
            except Exception as e:
                logger.error(f"Error downloading {file_name}: {str(e)}")
                raise

def load_model():
    """
    Load model components from the /tmp directory
    """
    try:
        # Download model files if they don't exist
        download_model_from_s3()
        
        # Load model from pickle files
        with open('/tmp/spam_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        with open('/tmp/spam_detection_model_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        with open('/tmp/spam_detection_model_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        logger.info("Model loaded successfully")
        return model, vectorizer, label_encoder
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def lambda_handler(event, context):
    """
    Lambda function handler for spam detection
    """
    logger.info("Received event: " + json.dumps(event))
    
    try:
        # Parse input - handle both direct invocation and API Gateway
        if 'body' in event:
            # This is from API Gateway
            try:
                body = json.loads(event['body'])
            except:
                body = event['body']  # In case it's already parsed
        else:
            # Direct invocation
            body = event
            
        # Extract text from various possible formats
        text = None
        if isinstance(body, dict):
            text = body.get('text') or body.get('message')
        elif isinstance(body, str):
            text = body
            
        if not text:
            logger.error("No text provided in the request")
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'No text provided. Please include "text" or "message" in your request.'
                })
            }
            
        logger.info(f"Processing text: {text}")
        
        # Load model components
        model, vectorizer, label_encoder = load_model()
        
        # Clean text
        cleaned_text = clean_text(text)
        logger.info(f"Cleaned text: {cleaned_text}")
        
        # Transform text to feature vector
        X = vectorizer.transform([cleaned_text])
        logger.info(f"Feature vector created with shape: {X.shape}")
        
        # Make prediction
        prediction_id = model.predict(X)[0]
        prediction_label = label_encoder.inverse_transform([prediction_id])[0]
        logger.info(f"Prediction: {prediction_label} (ID: {prediction_id})")
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            probability = float(probabilities[0][prediction_id])
            logger.info(f"Prediction probability: {probability}")
        
        # Prepare response
        response_body = {
            'input': text,
            'prediction': prediction_label,
            'confidence': probability
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f"Internal server error: {str(e)}"
            })
        } 