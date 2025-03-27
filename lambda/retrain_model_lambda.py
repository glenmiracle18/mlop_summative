#!/usr/bin/env python3
"""
Lambda Function for Retraining Spam Detection Model

This Lambda function is triggered when a new CSV file is uploaded to S3.
It downloads the file, trains a new model, and deploys it through API Gateway.

Author: Glen
"""

import os
import json
import logging
import boto3
import urllib.parse
import pandas as pd
import joblib
import pickle
import numpy as np
import re
import shutil
import tarfile
import tempfile
import time
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
apigateway_client = boto3.client('apigateway')
iam_client = boto3.client('iam')

# Model constants
MODEL_NAME = 'spam-detection-model'
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'spam-detection-model-artifacts')  # Get bucket name from environment variable
API_NAME = 'spam-detection-api'
PREDICTION_LAMBDA_NAME = 'spam-detection-prediction'
REGION = os.environ.get('REGION_NAME', 'us-east-1')

def clean_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data(df):
    """Preprocess the dataset"""
    logger.info("Preprocessing data")
    
    # Clean the messages
    df['clean_message'] = df['message'].apply(clean_text)
    
    # Encode the labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    return df, label_encoder

def train_model(df):
    """Train a new model with the provided data"""
    logger.info("Starting model training")
    
    # Preprocess data
    processed_df, label_encoder = preprocess_data(df)
    
    # Extract features and labels
    X = processed_df['clean_message']
    y = processed_df['label_encoded']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and fit the vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # Train the model (using Naive Bayes as default)
    model = MultinomialNB()
    model.fit(X_train_features, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Generate and log a classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    logger.info(f"Classification report: {json.dumps(report, indent=2)}")
    
    return model, vectorizer, label_encoder, accuracy

def save_model_to_s3(model, vectorizer, label_encoder):
    """Save model components to S3"""
    logger.info("Saving model to S3")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
    # Save model components
        model_path = os.path.join(temp_dir, 'spam_detection_model.pkl')
        vectorizer_path = os.path.join(temp_dir, 'spam_detection_vectorizer.pkl')
        encoder_path = os.path.join(temp_dir, 'spam_detection_label_encoder.pkl')
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(label_encoder, encoder_path)
        
        # Generate timestamp for versioning
        timestamp = int(time.time())
        
        # Upload to S3
        model_s3_key = f"models/{timestamp}/spam_detection_model.pkl"
        vectorizer_s3_key = f"models/{timestamp}/spam_detection_vectorizer.pkl"
        encoder_s3_key = f"models/{timestamp}/spam_detection_label_encoder.pkl"
        
        s3_client.upload_file(model_path, MODEL_BUCKET, model_s3_key)
        s3_client.upload_file(vectorizer_path, MODEL_BUCKET, vectorizer_s3_key)
        s3_client.upload_file(encoder_path, MODEL_BUCKET, encoder_s3_key)
        
        logger.info(f"Model uploaded to s3://{MODEL_BUCKET}/{model_s3_key}")
        
        return {
            'timestamp': timestamp,
            'model_s3_key': model_s3_key,
            'vectorizer_s3_key': vectorizer_s3_key,
            'encoder_s3_key': encoder_s3_key,
            'model_s3_uri': f"s3://{MODEL_BUCKET}/{model_s3_key}"
        }
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

def create_prediction_lambda(model_info):
    """Create or update Lambda function for prediction"""
    logger.info("Creating prediction Lambda function")
    
    # Check if Lambda role exists, create if not
    role_name = f"{PREDICTION_LAMBDA_NAME}-role"
    role_arn = None
    
    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response['Role']['Arn']
        logger.info(f"Using existing role: {role_arn}")
    except iam_client.exceptions.NoSuchEntityException:
        # Create a new role
        logger.info(f"Creating new role: {role_name}")
        
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy)
        )
        role_arn = response['Role']['Arn']
        
        # Attach policies
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        )
        
        # Create S3 access policy
        s3_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{MODEL_BUCKET}",
                        f"arn:aws:s3:::{MODEL_BUCKET}/*"
                    ]
                }
            ]
        }
        
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName="S3ModelAccess",
            PolicyDocument=json.dumps(s3_policy)
        )
        
        # Wait for role propagation
        logger.info("Waiting for role propagation (15 seconds)...")
        time.sleep(15)
    
    # Create Lambda function code
    prediction_code = f"""
import json
import boto3
import joblib
import re
import os

# Initialize S3 client
s3_client = boto3.client('s3')

# S3 bucket and keys for model files
MODEL_BUCKET = '{MODEL_BUCKET}'
MODEL_KEY = '{model_info['model_s3_key']}'
VECTORIZER_KEY = '{model_info['vectorizer_s3_key']}'
ENCODER_KEY = '{model_info['encoder_s3_key']}'

# Temp file paths
MODEL_PATH = '/tmp/model.pkl'
VECTORIZER_PATH = '/tmp/vectorizer.pkl'
ENCODER_PATH = '/tmp/encoder.pkl'

# Variable to hold loaded model components
model = None
vectorizer = None
label_encoder = None

def clean_text(text):
    \"\"\"Clean and preprocess text data\"\"\"
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\\w\\s]', '', text)
    text = re.sub(r'\\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text

def load_model():
    \"\"\"Load model components from S3\"\"\"
    global model, vectorizer, label_encoder
    
    if model is not None and vectorizer is not None and label_encoder is not None:
        return
    
    # Download model files
    s3_client.download_file(MODEL_BUCKET, MODEL_KEY, MODEL_PATH)
    s3_client.download_file(MODEL_BUCKET, VECTORIZER_KEY, VECTORIZER_PATH)
    s3_client.download_file(MODEL_BUCKET, ENCODER_KEY, ENCODER_PATH)
    
    # Load model components
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

def lambda_handler(event, context):
    \"\"\"Handle prediction requests\"\"\"
    try:
        # Load model if not already loaded
        load_model()
        
        # Parse input
        if 'body' in event:
            # From API Gateway
            try:
                body = json.loads(event['body'])
                message = body.get('message')
                messages = body.get('messages', [message] if message else [])
            except:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'error': 'Invalid request body'
                    })
                }
        else:
            # Direct invocation
            message = event.get('message')
            messages = event.get('messages', [message] if message else [])
        
        if not messages:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'No message provided'
                })
            }
        
        # Preprocess messages
        cleaned_messages = [clean_text(msg) for msg in messages]
    
    # Vectorize
        X = vectorizer.transform(cleaned_messages)
    
    # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
    
        # Convert to labels
    predicted_labels = label_encoder.inverse_transform(predictions)
    
        # Format response
        if len(messages) == 1:
            class_idx = int(predictions[0])
            confidence = float(probabilities[0][class_idx])
            
            result = {
                'message': messages[0],
                'prediction': predicted_labels[0],
                'confidence': confidence
            }
        else:
            result = []
            for i, (msg, label) in enumerate(zip(messages, predicted_labels)):
        class_idx = int(predictions[i])
        confidence = float(probabilities[i][class_idx])
        
                result.append({
                    'message': msg,
            'prediction': label,
            'confidence': confidence
        })
    
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
"""
    
    # Create a zip file with the Lambda code
    temp_dir = tempfile.mkdtemp()
    try:
        lambda_file_path = os.path.join(temp_dir, 'lambda_function.py')
        zip_file_path = os.path.join(temp_dir, 'function.zip')
        
        with open(lambda_file_path, 'w') as f:
            f.write(prediction_code)
        
        # Create zip file
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(lambda_file_path, 'lambda_function.py')
        
        # Read the zip file
        with open(zip_file_path, 'rb') as f:
            zip_bytes = f.read()
        
        # Check if Lambda function exists
        try:
            lambda_client.get_function(FunctionName=PREDICTION_LAMBDA_NAME)
            # Update existing function
            logger.info(f"Updating existing Lambda function: {PREDICTION_LAMBDA_NAME}")
            response = lambda_client.update_function_code(
                FunctionName=PREDICTION_LAMBDA_NAME,
                ZipFile=zip_bytes,
                Publish=True
            )
        except lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            logger.info(f"Creating new Lambda function: {PREDICTION_LAMBDA_NAME}")
            response = lambda_client.create_function(
                FunctionName=PREDICTION_LAMBDA_NAME,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_bytes},
                Timeout=30,
                MemorySize=256,
                Publish=True
            )
        
        # Add permission for API Gateway if it doesn't exist
        try:
            lambda_client.add_permission(
                FunctionName=PREDICTION_LAMBDA_NAME,
                StatementId='apigateway-permission',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com'
            )
            logger.info("Added API Gateway permission to Lambda function")
        except lambda_client.exceptions.ResourceConflictException:
            logger.info("API Gateway permission already exists")
        
        return response['FunctionArn']
    
    finally:
        shutil.rmtree(temp_dir)

def deploy_api_gateway(lambda_arn):
    """Create or update API Gateway"""
    logger.info("Deploying API Gateway")
    
    # Check if API exists
    api_id = None
    try:
        apis = apigateway_client.get_rest_apis()
        for api in apis['items']:
            if api['name'] == API_NAME:
                api_id = api['id']
                logger.info(f"Found existing API: {api_id}")
                break
    except Exception as e:
        logger.error(f"Error checking existing APIs: {str(e)}")
    
    # Create new API if not exists
    if not api_id:
        logger.info(f"Creating new API: {API_NAME}")
        response = apigateway_client.create_rest_api(
            name=API_NAME,
            description='Spam Detection API',
            endpointConfiguration={'types': ['REGIONAL']}
        )
        api_id = response['id']
    
    # Get the root resource ID
    resources = apigateway_client.get_resources(restApiId=api_id)
    root_id = None
    predict_resource_id = None
    
    for resource in resources['items']:
        if resource['path'] == '/':
            root_id = resource['id']
        elif resource['path'] == '/predict':
            predict_resource_id = resource['id']
    
    # Create /predict resource if it doesn't exist
    if not predict_resource_id:
        logger.info("Creating /predict resource")
        resource = apigateway_client.create_resource(
            restApiId=api_id,
            parentId=root_id,
            pathPart='predict'
        )
        predict_resource_id = resource['id']
    
    # Create or update POST method
    try:
        logger.info("Creating POST method")
        apigateway_client.put_method(
            restApiId=api_id,
            resourceId=predict_resource_id,
            httpMethod='POST',
            authorizationType='NONE',
            apiKeyRequired=False
        )
    except apigateway_client.exceptions.ConflictException:
        logger.info("POST method already exists")
    
    # Create or update method integration
    try:
        logger.info("Setting up Lambda integration")
        uri = f"arn:aws:apigateway:{REGION}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"
        
        apigateway_client.put_integration(
            restApiId=api_id,
            resourceId=predict_resource_id,
            httpMethod='POST',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=uri
        )
    except apigateway_client.exceptions.ConflictException:
        logger.info("Integration already exists, updating...")
        apigateway_client.update_integration(
            restApiId=api_id,
            resourceId=predict_resource_id,
            httpMethod='POST',
            patchOperations=[
                {
                    'op': 'replace',
                    'path': '/uri',
                    'value': uri
                }
            ]
        )
    
    # Create or update CORS (OPTIONS method)
    try:
        logger.info("Setting up CORS")
        apigateway_client.put_method(
            restApiId=api_id,
            resourceId=predict_resource_id,
            httpMethod='OPTIONS',
            authorizationType='NONE'
        )
        
        apigateway_client.put_integration(
            restApiId=api_id,
            resourceId=predict_resource_id,
            httpMethod='OPTIONS',
            type='MOCK',
            integrationHttpMethod='OPTIONS',
            requestTemplates={
                'application/json': '{"statusCode": 200}'
            }
        )
        
        apigateway_client.put_method_response(
            restApiId=api_id,
            resourceId=predict_resource_id,
            httpMethod='OPTIONS',
            statusCode='200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Headers': True,
                'method.response.header.Access-Control-Allow-Methods': True,
                'method.response.header.Access-Control-Allow-Origin': True
            }
        )
        
        apigateway_client.put_integration_response(
            restApiId=api_id,
            resourceId=predict_resource_id,
            httpMethod='OPTIONS',
            statusCode='200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
                'method.response.header.Access-Control-Allow-Methods': "'GET,POST,OPTIONS'",
                'method.response.header.Access-Control-Allow-Origin': "'*'"
            }
        )
    except apigateway_client.exceptions.ConflictException:
        logger.info("CORS already set up")
    
    # Deploy the API
    logger.info("Deploying API")
    deployment = apigateway_client.create_deployment(
        restApiId=api_id,
        stageName='prod',
        description=f'Deployment {time.strftime("%Y-%m-%d %H:%M:%S")}'
    )
    
    # Get the API URL
    api_url = f"https://{api_id}.execute-api.{REGION}.amazonaws.com/prod/predict"
    logger.info(f"API deployed at: {api_url}")
    
    return {
        'api_id': api_id,
        'api_url': api_url
    }

def lambda_handler(event, context):
    """Lambda handler function"""
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Get the S3 bucket and key from the event
        if 'Records' in event:
            # Triggered by S3 upload
            s3_event = event['Records'][0]['s3']
            bucket = s3_event['bucket']['name']
            key = urllib.parse.unquote_plus(s3_event['object']['key'])
        elif 'bucket' in event and 'key' in event:
            # Direct invocation with bucket and key
            bucket = event['bucket']
            key = event['key']
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Invalid event format. Requires S3 upload or bucket/key parameters.'
                })
            }
        
        logger.info(f"Processing file s3://{bucket}/{key}")
        
        # Download the CSV file
        download_path = '/tmp/data.csv'
        s3_client.download_file(bucket, key, download_path)
        logger.info(f"Downloaded file to {download_path}")
        
        # Load the data
        try:
            df = pd.read_csv(download_path)
            logger.info(f"Loaded data with {len(df)} rows")
            
            # Ensure required columns exist
            required_columns = ['message', 'label']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns: {', '.join(missing)}")
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': f"Failed to load CSV file: {str(e)}"
                })
            }
        
        # Train a new model
        model, vectorizer, label_encoder, accuracy = train_model(df)
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Save model to S3
        model_info = save_model_to_s3(model, vectorizer, label_encoder)
        logger.info(f"Model saved to S3: {model_info['model_s3_uri']}")
        
        # Deploy model using API Gateway
        try:
            # Create prediction Lambda function
            lambda_arn = create_prediction_lambda(model_info)
            logger.info(f"Prediction Lambda deployed: {lambda_arn}")
            
            # Deploy to API Gateway
            api_info = deploy_api_gateway(lambda_arn)
            logger.info(f"API Gateway deployed: {api_info['api_url']}")
            
            # Update the frontend web app's API URL
            api_url = api_info['api_url']
            logger.info(f"API URL for frontend: {api_url}")
        
            # Return success response
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Model retrained and deployed successfully',
                    'model_accuracy': accuracy,
                    'api_url': api_url,
                    'model_s3_uri': model_info['model_s3_uri']
                })
            }
        except Exception as e:
            logger.error(f"Failed to deploy model via API Gateway: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': f"Failed to deploy model via API Gateway: {str(e)}"
                })
            }
    
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Failed to retrain model: {str(e)}"
            })
        } 