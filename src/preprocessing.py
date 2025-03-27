import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split


def clean_text(text):
    """
    Clean and normalize text data
    """
    # lowercasing
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
    """
    print("Starting data preprocessing...")

    processed_df = df.copy()

    # Rename columns if using default names
    if label_col != 'label' or text_col != 'message':
        column_mapping = {label_col: 'label', text_col: 'message'}
        processed_df = processed_df.rename(columns=column_mapping)
        print(f"Renamed columns: {column_mapping}")

    # Drop unnecessary columns (if they exist)
    cols_to_drop = [col for col in processed_df.columns if col.startswith('Unnamed:')]
    if cols_to_drop:
        processed_df = processed_df.drop(columns=cols_to_drop)
        print(f"Dropped unnecessary columns: {cols_to_drop}")

    # Check for and handle missing values
    missing_values = processed_df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found missing values: {missing_values}")
        processed_df = processed_df.dropna()
        print(f"Dropped rows with missing values. New shape: {processed_df.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    processed_df['label_encoded'] = label_encoder.fit_transform(processed_df['label'])
    print(f"Encoded labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # text length as a feature
    processed_df['text_length'] = processed_df['message'].apply(len)

    # Clean messages
    print("Cleaning text messages...")
    processed_df['clean_message'] = processed_df['message'].apply(clean_text)

    print("Data preprocessing completed")
    return processed_df, label_encoder


def create_features(df, vectorizer_type='tfidf', max_features=5000, test_size=0.2, random_state=42):
    """
    Create features and split into training/testing sets
    """
    print(f"\nStarting feature engineering with {vectorizer_type} vectorizer...")

    # Split the data
    X = df['clean_message']
    y = df['label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples)")

    # Initialize vectorizer
    if vectorizer_type.lower() == 'count':
        vectorizer = CountVectorizer(max_features=max_features)
        print(f"Using CountVectorizer with max_features={max_features}")
    else:
        vectorizer = TfidfVectorizer(max_features=max_features)
        print(f"Using TfidfVectorizer with max_features={max_features}")

    # Fit and transform the training data
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    print(f"Number of features: {len(feature_names)}")
    print("Sample features (words):", feature_names[:10])

    print(f"Feature extraction completed. Train features shape: {X_train_features.shape}")

    return X_train_features, X_test_features, y_train, y_test, vectorizer 