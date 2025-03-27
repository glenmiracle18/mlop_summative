#!/usr/bin/env python3
# spam_detection_pipeline.py
"""
Complete SMS Spam Detection Pipeline

This script runs the entire spam detection workflow from data loading to model saving
without requiring manual intervention. It can be executed from the command line with
customizable parameters.

Example usage:
    python spam_detection_pipeline.py --data_path spam.csv --output_dir models --model_type naive_bayes

Author: Your Name
Date: March 24, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
import joblib
import os
import time
import argparse
import json
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spam_detection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path, encoding='cp1252'):
    """
    Load data from CSV file
    
    Args:
        file_path (str): Path to the CSV data file
        encoding (str): File encoding
        
    Returns:
        DataFrame: The loaded dataframe
    """
    logger.info(f"Loading data from {file_path}")
    try:
        # Try with specified encoding
        df = pd.read_csv(file_path, encoding=encoding)
        logger.info(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data with {encoding} encoding: {str(e)}")
        try:
            # Try with alternative encoding
            alt_encoding = 'utf-8' if encoding != 'utf-8' else 'latin1'
            df = pd.read_csv(file_path, encoding=alt_encoding)
            logger.info(f"Data loaded with {alt_encoding} encoding. Shape: {df.shape}")
            return df
        except Exception as e2:
            logger.error(f"Failed to load data with alternative encoding: {str(e2)}")
            raise

def clean_text(text):
    """
    Clean and normalize text data
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Cleaned text
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

def preprocess_data(df, label_col='v1', text_col='v2'):
    """
    Clean and preprocess the data
    
    Args:
        df (DataFrame): The dataframe to preprocess
        label_col (str): Column name for the label
        text_col (str): Column name for the text messages
        
    Returns:
        tuple: Preprocessed dataframe and label encoder
    """
    logger.info("Starting data preprocessing")
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Rename columns if using default names
    if label_col != 'label' or text_col != 'message':
        column_mapping = {label_col: 'label', text_col: 'message'}
        processed_df = processed_df.rename(columns=column_mapping)
        logger.info(f"Renamed columns: {column_mapping}")
    
    # Drop unnecessary columns (if they exist)
    cols_to_drop = [col for col in processed_df.columns if col.startswith('Unnamed:')]
    if cols_to_drop:
        processed_df = processed_df.drop(columns=cols_to_drop)
        logger.info(f"Dropped unnecessary columns: {cols_to_drop}")
    
    # Check for and handle missing values
    missing_values = processed_df.isnull().sum()
    if missing_values.sum() > 0:
        logger.info(f"Found missing values: {missing_values}")
        processed_df = processed_df.dropna()
        logger.info(f"Dropped rows with missing values. New shape: {processed_df.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    processed_df['label_encoded'] = label_encoder.fit_transform(processed_df['label'])
    logger.info(f"Encoded labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Add text length as a feature
    processed_df['text_length'] = processed_df['message'].apply(len)
    
    # Clean messages
    logger.info("Cleaning text messages")
    processed_df['clean_message'] = processed_df['message'].apply(clean_text)
    
    # Log example of original vs. cleaned messages
    logger.info("Example of original vs. cleaned messages:")
    for i in range(min(3, len(processed_df))):
        logger.info(f"Original: {processed_df['message'].iloc[i]}")
        logger.info(f"Cleaned: {processed_df['clean_message'].iloc[i]}")
    
    logger.info("Data preprocessing completed")
    return processed_df, label_encoder

def create_features(df, vectorizer_type='tfidf', max_features=5000, test_size=0.2, random_state=42):
    """
    Create features and split into training/testing sets
    
    Args:
        df (DataFrame): Preprocessed dataframe
        vectorizer_type (str): Type of vectorization ('count' or 'tfidf')
        max_features (int): Maximum number of features to use
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train_features, X_test_features, y_train, y_test, vectorizer
    """
    logger.info(f"Starting feature engineering with {vectorizer_type} vectorizer")
    
    # Split the data
    X = df['clean_message']
    y = df['label_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples)")
    
    # Initialize vectorizer
    if vectorizer_type.lower() == 'count':
        vectorizer = CountVectorizer(max_features=max_features)
        logger.info(f"Using CountVectorizer with max_features={max_features}")
    else:
        vectorizer = TfidfVectorizer(max_features=max_features)
        logger.info(f"Using TfidfVectorizer with max_features={max_features}")
    
    # Fit and transform the training data
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    logger.info(f"Number of features: {len(feature_names)}")
    logger.info(f"Sample features (words): {', '.join(feature_names[:10])}")
    
    logger.info(f"Feature extraction completed. Train features shape: {X_train_features.shape}")
    
    return X_train_features, X_test_features, y_train, y_test, vectorizer

def train_model(X_train, y_train, model_type='naive_bayes'):
    """
    Train a specific machine learning model
    
    Args:
        X_train: Features for training
        y_train: Labels for training
        model_type (str): Type of model to train
        
    Returns:
        object: Trained model
    """
    models = {
        'naive_bayes': MultinomialNB(),
        'logistic': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'svm': SVC(kernel='linear', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if model_type not in models:
        error_msg = f"Unsupported model type: {model_type}. Available types: {list(models.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Training {model_type} model")
    start_time = time.time()
    
    model = models[model_type]
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    return model

def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all available models and select the best one
    
    Args:
        X_train: Features for training
        y_train: Labels for training
        X_test: Features for testing
        y_test: Labels for testing
        
    Returns:
        tuple: Results dictionary, best model name, best model
    """
    logger.info("Training all models for comparison")
    
    models = {
        'naive_bayes': MultinomialNB(),
        'logistic': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'svm': SVC(kernel='linear', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name} model")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': float(accuracy),
            'training_time': training_time
        }
        
        logger.info(f"{name} - Accuracy: {accuracy:.4f}, Training Time: {training_time:.2f} seconds")
    
    # Select the best model
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']
    
    logger.info(f"Best model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    return results, best_model_name, best_model

def evaluate_model(model, X_test, y_test, label_encoder, output_dir=None):
    """
    Evaluate the model performance
    
    Args:
        model: Trained model
        X_test: Features for testing
        y_test: Labels for testing
        label_encoder: LabelEncoder for class names
        output_dir (str, optional): Directory to save visualization files
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info(f"Confusion matrix:\n{conf_matrix}")
    logger.info(f"Classification report:\n{class_report}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the confusion matrix figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        logger.info(f"Confusion matrix saved to {confusion_matrix_path}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report_dict
    }

def save_model(model, vectorizer, label_encoder, output_dir, model_name='spam_detection_model'):
    """
    Save the model and related components
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        label_encoder: Fitted label encoder
        output_dir (str): Directory to save the model
        model_name (str): Base name for the model files
        
    Returns:
        dict: Paths where artifacts were saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    vectorizer_path = os.path.join(output_dir, f"{model_name}_vectorizer.pkl")
    label_encoder_path = os.path.join(output_dir, f"{model_name}_label_encoder.pkl")
    
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    
    logger.info(f"Saving vectorizer to {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)
    
    logger.info(f"Saving label encoder to {label_encoder_path}")
    joblib.dump(label_encoder, label_encoder_path)
    
    # Save metadata
    metadata = {
        'model_creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': type(model).__name__,
        'vectorizer_type': type(vectorizer).__name__,
        'files': {
            'model': model_path,
            'vectorizer': vectorizer_path,
            'label_encoder': label_encoder_path
        }
    }
    
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model metadata saved to {metadata_path}")
    
    saved_paths = {
        'model': model_path,
        'vectorizer': vectorizer_path,
        'label_encoder': label_encoder_path,
        'metadata': metadata_path
    }
    
    return saved_paths

def predict_spam(messages, model, vectorizer, label_encoder):
    """
    Predict whether messages are spam or ham
    
    Args:
        messages (str or list): Text message(s) to classify
        model: Trained model
        vectorizer: Fitted vectorizer
        label_encoder: Fitted label encoder
        
    Returns:
        list: Prediction results
    """
    # Handle single string input
    if isinstance(messages, str):
        messages = [messages]
    
    # Clean the text
    cleaned_texts = [clean_text(msg) for msg in messages]
    
    # Vectorize
    features = vectorizer.transform(cleaned_texts)
    
    # Predict
    predictions = model.predict(features)
    
    # Get probabilities if model supports it
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)
    
    # Convert numeric predictions back to labels
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    # Prepare results
    results = []
    for i, (msg, label) in enumerate(zip(messages, predicted_labels)):
        result = {
            'message': msg,
            'prediction': label
        }
        
        # Add probability if available
        if probabilities is not None:
            # Find the index of the predicted class
            class_idx = list(label_encoder.classes_).index(label)
            result['probability'] = float(probabilities[i][class_idx])
        
        results.append(result)
    
    return results

def run_pipeline(config):
    """
    Run the complete pipeline
    
    Args:
        config (dict): Configuration parameters for the pipeline
        
    Returns:
        dict: Results of the pipeline run
    """
    start_time = time.time()
    logger.info("Starting complete spam detection pipeline")
    
    # Extract configuration
    data_path = config['data_path']
    output_dir = config['output_dir']
    vectorizer_type = config['vectorizer_type']
    model_type = config['model_type']
    train_all = config['train_all']
    
    try:
        # Step 1: Load data
        df = load_data(data_path)
        
        # Step 2: Preprocess data
        processed_df, label_encoder = preprocess_data(df)
        
        # Step 3: Create features
        X_train_features, X_test_features, y_train, y_test, vectorizer = create_features(
            processed_df, vectorizer_type=vectorizer_type
        )
        
        # Step 4: Train model(s)
        if train_all:
            model_results, best_model_name, best_model = train_all_models(
                X_train_features, y_train, X_test_features, y_test
            )
            selected_model_name = best_model_name
            selected_model = best_model
        else:
            selected_model = train_model(X_train_features, y_train, model_type)
            selected_model_name = model_type
        
        # Step 5: Evaluate model
        evaluation_metrics = evaluate_model(
            selected_model, X_test_features, y_test, label_encoder, output_dir
        )
        
        # Step 6: Save model
        saved_paths = save_model(
            selected_model,
            vectorizer,
            label_encoder,
            output_dir
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
        
        # Test with some example messages
        test_messages = [
            "Congratulations! You've won a $1000 gift card. Call now to claim your prize!",
            "Hey, can we meet for coffee tomorrow at 3pm?",
            "URGENT: Your bank account has been suspended. Click here to verify your information.",
            "I'll be home late tonight. Don't wait up."
        ]
        
        logger.info("Testing the model with example messages")
        predictions = predict_spam(test_messages, selected_model, vectorizer, label_encoder)
        for pred in predictions:
            logger.info(f"Message: {pred['message']}")
            logger.info(f"Prediction: {pred['prediction']}")
            if 'probability' in pred:
                logger.info(f"Confidence: {pred['probability']:.2f}")
        
        # Prepare results
        pipeline_results = {
            'selected_model': selected_model_name,
            'accuracy': evaluation_metrics['accuracy'],
            'saved_paths': saved_paths,
            'execution_time': execution_time,
            'example_predictions': predictions
        }
        
        # Save pipeline results
        results_path = os.path.join(output_dir, 'pipeline_results.json')
        # Convert NumPy types to native Python types for JSON serialization
        with open(results_path, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        
        logger.info(f"Pipeline results saved to {results_path}")
        
        return pipeline_results
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise

def main():
    """Main entry point for the spam detection pipeline"""
    parser = argparse.ArgumentParser(description='SMS Spam Detection Pipeline')
    parser.add_argument('--data_path', required=True, help='Path to the CSV data file')
    parser.add_argument('--output_dir', default='model_output', help='Directory to save outputs')
    parser.add_argument('--vectorizer_type', choices=['count', 'tfidf'], default='tfidf', help='Type of vectorizer to use')
    parser.add_argument('--model_type', choices=['naive_bayes', 'logistic', 'svm', 'random_forest'], 
                        default='naive_bayes', help='Type of model to train')
    parser.add_argument('--train_all', action='store_true', help='Train all models and select the best one')
    
    args = parser.parse_args()
    
    # Prepare configuration
    config = {
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'vectorizer_type': args.vectorizer_type,
        'model_type': args.model_type,
        'train_all': args.train_all
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the pipeline
    try:
        results = run_pipeline(config)
        
        # Print summary
        print("\n===== Pipeline Execution Summary =====")
        print(f"Model: {results['selected_model']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Execution time: {results['execution_time']:.2f} seconds")
        print(f"Model saved to: {results['saved_paths']['model']}")
        print(f"Results saved to: {os.path.join(args.output_dir, 'pipeline_results.json')}")
        print("\nPipeline executed successfully!")
        
        return 0  # Success exit code
    
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        return 1  # Error exit code

if __name__ == "__main__":
    # Download NLTK resources if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    sys.exit(main())