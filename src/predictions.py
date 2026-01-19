"""
Prediction functions for manual and batch predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime


def prepare_prediction_input(input_values, selected_features, feature_encoders):
    """
    Prepare input array for prediction with proper encoding
    
    Args:
        input_values: Dictionary of feature values
        selected_features: List of feature names
        feature_encoders: Dictionary of feature encoders
        
    Returns:
        numpy array ready for model prediction
    """
    input_row = []
    for feature in selected_features:
        if feature in feature_encoders:
            # Encode categorical feature
            encoder = feature_encoders[feature]
            try:
                encoded_val = encoder.transform([str(input_values[feature])])[0]
                input_row.append(encoded_val)
            except ValueError:
                # Handle unseen categories
                input_row.append(0)  # Default to first category
        else:
            # Numerical feature
            input_row.append(float(input_values[feature]))
    
    return np.array([input_row])


def predict_manual(trained_models, input_array, label_encoder):
    """
    Make prediction from manual input
    
    Args:
        trained_models: Dictionary of trained models
        input_array: Input array for prediction
        label_encoder: Label encoder for target variable
        
    Returns:
        Dictionary with predictions from all models
    """
    predictions = {}
    
    for model_name, model_data in trained_models.items():
        model = model_data['model']
        pred = model.predict(input_array)[0]
        pred_proba = model.predict_proba(input_array)[0]
        confidence = max(pred_proba) * 100
        pred_class = label_encoder.inverse_transform([pred])[0]
        
        predictions[model_name] = {
            'class': pred_class,
            'confidence': confidence
        }
    
    return predictions


def predict_batch(trained_models, batch_df, selected_features, feature_encoders, label_encoder):
    """
    Make batch predictions from CSV file
    
    Args:
        trained_models: Dictionary of trained models
        batch_df: DataFrame with features
        selected_features: List of feature names
        feature_encoders: Dictionary of feature encoders
        label_encoder: Label encoder for target variable
        
    Returns:
        DataFrame with predictions added
    """
    # Check if required features are present
    missing_features = [f for f in selected_features if f not in batch_df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {', '.join(missing_features)}")
    
    batch_X = batch_df[selected_features].copy()
    
    # Encode categorical features if needed
    for feature in selected_features:
        if feature in feature_encoders:
            encoder = feature_encoders[feature]
            # Handle unseen categories
            batch_X[feature] = batch_X[feature].astype(str)
            unique_vals = batch_X[feature].unique()
            encoder_classes = set(encoder.classes_)
            for val in unique_vals:
                if val not in encoder_classes:
                    batch_X.loc[batch_X[feature] == val, feature] = encoder.classes_[0]
            batch_X[feature] = encoder.transform(batch_X[feature])
    
    # Ensure numeric
    batch_X = batch_X.astype(float)
    
    batch_predictions = {}
    for model_name, model_data in trained_models.items():
        model = model_data['model']
        preds = model.predict(batch_X)
        pred_probas = model.predict_proba(batch_X)
        confidences = np.max(pred_probas, axis=1) * 100
        pred_classes = label_encoder.inverse_transform(preds)
        
        batch_predictions[f"{model_name.upper()}_Pred"] = pred_classes
        batch_predictions[f"{model_name.upper()}_Confidence"] = confidences
    
    # Create results dataframe
    results_df = batch_df.copy()
    for key, values in batch_predictions.items():
        results_df[key] = values
    
    # Determine final prediction (majority vote)
    pred_cols = [col for col in results_df.columns if col.endswith('_Pred')]
    if pred_cols:
        results_df['Final_Pred'] = results_df[pred_cols].mode(axis=1)[0]
        conf_cols = [col for col in results_df.columns if col.endswith('_Confidence')]
        if conf_cols:
            results_df['Final_Confidence'] = results_df[conf_cols].mean(axis=1)
    
    return results_df


def export_predictions(results_df, filename_prefix="predictions"):
    """
    Export predictions to CSV string
    
    Args:
        results_df: DataFrame with predictions
        filename_prefix: Prefix for filename
        
    Returns:
        Tuple of (csv_string, filename)
    """
    csv = results_df.to_csv(index=False)
    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return csv, filename
