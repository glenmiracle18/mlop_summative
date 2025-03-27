import pickle
import json
import numpy as np
import pandas as pd
from preprocessing import clean_text

def load_model(model_path='models/spam_detection_model.pkl',
              vectorizer_path='models/spam_detection_model_vectorizer.pkl',
              label_encoder_path='models/spam_detection_model_label_encoder.pkl',
              metadata_path='models/spam_detection_model_metadata.json'):
    """
    Load the saved model and necessary artifacts
    """
    print(f"Loading model from {model_path}")
    
    # Load the main model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load the vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load the label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("Model and artifacts loaded successfully")
    return model, vectorizer, label_encoder, metadata


def predict_message(message, model, vectorizer, label_encoder):
    """
    Make a prediction for a single message
    """
    # Clean the message
    cleaned_message = clean_text(message)
    
    # Transform the message using the vectorizer
    message_features = vectorizer.transform([cleaned_message])
    
    # Make prediction
    predicted_label_encoded = model.predict(message_features)[0]
    predicted_proba = model.predict_proba(message_features)[0]
    
    # Get the original label
    predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
    
    # Get the confidence (probability) of the prediction
    confidence = predicted_proba[predicted_label_encoded]
    
    result = {
        'message': message,
        'cleaned_message': cleaned_message,
        'prediction': predicted_label,
        'confidence': float(confidence),
        'is_spam': bool(predicted_label_encoded == 1) if 'spam' in label_encoder.classes_ else None
    }
    
    return result


def predict_batch(messages, model, vectorizer, label_encoder):
    """
    Make predictions for a batch of messages
    """
    cleaned_messages = [clean_text(msg) for msg in messages]
    
    # Transform all messages using the vectorizer
    message_features = vectorizer.transform(cleaned_messages)
    
    # Make predictions
    predicted_labels_encoded = model.predict(message_features)
    predicted_probas = model.predict_proba(message_features)
    
    # Get the original labels
    predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)
    
    # Prepare results
    results = []
    for i, (msg, clean_msg, label, label_encoded) in enumerate(zip(
            messages, cleaned_messages, predicted_labels, predicted_labels_encoded)):
        
        confidence = predicted_probas[i][label_encoded]
        
        result = {
            'message': msg,
            'cleaned_message': clean_msg,
            'prediction': label,
            'confidence': float(confidence),
            'is_spam': bool(label_encoded == 1) if 'spam' in label_encoder.classes_ else None
        }
        results.append(result)
    
    return results 