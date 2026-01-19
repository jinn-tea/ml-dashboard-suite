"""
Data processing and preprocessing functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_dataset(uploaded_file, max_size_mb=500, sample_size=None):
    """
    Load dataset from uploaded file with memory optimization
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_size_mb: Maximum file size in MB
        sample_size: Optional sample size for large datasets
        
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return None, f"File too large: {file_size_mb:.1f} MB. Maximum allowed: {max_size_mb} MB. Please use a smaller file."
        
        # Read CSV with optimized settings
        df = pd.read_csv(
            uploaded_file,
            low_memory=False,  # Avoid mixed type warnings
            nrows=sample_size if sample_size else None  # Limit rows if sampling
        )
        
        # Optimize dataframe
        if sample_size is None and len(df) > 100000:
            # Auto-sample very large datasets
            df = df.sample(n=50000, random_state=42).reset_index(drop=True)
            return df, f"⚠️ Dataset automatically sampled to 50,000 rows for performance. Original size was {len(df):,} rows."
        
        return df, None
    except Exception as e:
        return None, str(e)


def get_feature_info(df, features):
    """
    Get information about selected features
    
    Args:
        df: DataFrame
        features: List of feature names
        
    Returns:
        DataFrame with feature information
    """
    feature_info = []
    for feat in features:
        dtype = df[feat].dtype
        sample_val = str(df[feat].iloc[0]) if len(df) > 0 else "N/A"
        is_numeric = pd.api.types.is_numeric_dtype(df[feat])
        feature_info.append({
            'Feature': feat,
            'Type': 'Numeric' if is_numeric else 'Text/Categorical',
            'Data Type': str(dtype),
            'Sample Value': sample_val[:50] + '...' if len(sample_val) > 50 else sample_val
        })
    return pd.DataFrame(feature_info)


def preprocess_features(X, selected_features):
    """
    Preprocess features: encode categorical and handle missing values
    
    Args:
        X: DataFrame with features
        selected_features: List of feature names
        
    Returns:
        Tuple of (processed_X, categorical_features, feature_encoders)
    """
    categorical_features = []
    numerical_features = []
    feature_encoders = {}
    
    for feature in selected_features:
        # Try to convert to numeric first
        numeric_series = pd.to_numeric(X[feature], errors='coerce')
        
        # Check if conversion was successful (not all NaN)
        if numeric_series.isna().all() or X[feature].dtype == 'object' or X[feature].dtype.name == 'category':
            # This is a categorical/text feature - encode it
            categorical_features.append(feature)
            le = LabelEncoder()
            
            # Handle categorical dtype - convert to string first if needed
            if X[feature].dtype.name == 'category':
                # Convert category to string to allow adding 'missing_value'
                X[feature] = X[feature].astype(str)
                # Replace 'nan' strings with 'missing_value'
                X[feature] = X[feature].replace('nan', 'missing_value')
            else:
                # Handle any NaN values in categorical data
                X[feature] = X[feature].fillna('missing_value')
            
            # Convert to string and encode
            X[feature] = X[feature].astype(str)
            X[feature] = le.fit_transform(X[feature])
            feature_encoders[feature] = le
        else:
            # This is numeric, but might have some non-numeric values
            numerical_features.append(feature)
            X[feature] = numeric_series
    
    # Fill any NaN values in numerical features with mean
    for feature in numerical_features:
        if X[feature].isna().any():
            X[feature] = X[feature].fillna(X[feature].mean())
    
    # Ensure all columns are numeric (should be after encoding)
    X = X.astype(float)
    
    # Final check - ensure no NaN or infinite values
    if X.isnull().any().any() or np.isinf(X).any().any():
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
    
    return X, categorical_features, feature_encoders


def prepare_training_data(df, selected_features, target_variable, train_test_split_ratio):
    """
    Prepare data for training: preprocess features and split data
    
    Args:
        df: DataFrame
        selected_features: List of feature names
        target_variable: Name of target variable
        train_test_split_ratio: Train/test split ratio (percentage)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, label_encoder, feature_encoders, categorical_features)
    """
    # Prepare data
    X = df[selected_features].copy()
    y = df[target_variable].copy()
    
    # Preprocess features
    X, categorical_features, feature_encoders = preprocess_features(X, selected_features)
    
    # Encode target if needed
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    
    # Train-test split
    test_size = (100 - train_test_split_ratio) / 100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, le, feature_encoders, categorical_features
