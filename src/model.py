import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score
)
import pickle
import json

def train_model(X_train, y_train, model_type='naive_bayes'):
    """
    Train a specific machine learning model
    naive bayes by default cuz it's the best for the job now
    """
    models = {
        'naive_bayes': MultinomialNB(),
        'logistic': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'svm': SVC(kernel='linear', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}. Available types: {list(models.keys())}")

    print(f"Training {model_type} model...")
    start_time = time.time()

    model = models[model_type]
    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")

    return model


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all available models and select the best one
    """
    print("Training all models for comparison...")

    models = {
        'naive_bayes': MultinomialNB(),
        'logistic': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'svm': SVC(kernel='linear', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} model...")
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

        print(f"{name} - Accuracy: {accuracy:.4f}, Training Time: {training_time:.2f} seconds")

    # Select the best model
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']

    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

    return results, best_model_name, best_model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model performance
    """
    print("\nEvaluating model performance...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    precision = float(precision_score(y_test, y_pred, average='weighted'))
    recall = float(recall_score(y_test, y_pred, average='weighted'))
    f1 = float(f1_score(y_test, y_pred, average='weighted'))

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        conf_matrix,
        index=[f'Actual {c}' for c in label_encoder.classes_],
        columns=[f'Predicted {c}' for c in label_encoder.classes_]
    ))

    print("\nClassification Report:")
    print(class_report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()

    # Return evaluation metrics
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report_dict
    }

    return results


def save_model(model, vectorizer, label_encoder, metadata, model_path='models/spam_detection_model.pkl'):
    """
    Save model and artifacts
    """
    print(f"\nSaving model to {model_path}")
    
    # Save the main model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the vectorizer
    vectorizer_path = model_path.replace('.pkl', '_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save the label encoder
    label_encoder_path = model_path.replace('.pkl', '_label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save metadata
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model and artifacts saved successfully")
    return {
        'model_path': model_path,
        'vectorizer_path': vectorizer_path,
        'label_encoder_path': label_encoder_path,
        'metadata_path': metadata_path
    } 