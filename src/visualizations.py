"""
Visualization functions for the ML Dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def plot_model_comparison(trained_models):
    """
    Create bar chart comparing model accuracies
    
    Args:
        trained_models: Dictionary of trained models
        
    Returns:
        Plotly figure
    """
    model_names = list(trained_models.keys())
    accuracies = [trained_models[m]['accuracy']*100 for m in model_names]
    
    fig = px.bar(
        x=model_names,
        y=accuracies,
        labels={'x': 'Model', 'y': 'Accuracy (%)'},
        color=accuracies,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


def plot_feature_importance(model, selected_features, top_n=10):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model with feature_importances_ attribute
        selected_features: List of feature names
        top_n: Number of top features to show
        
    Returns:
        Plotly figure or None
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    return fig


def plot_decision_boundary(model, X_test, y_test, selected_features, x_feature, y_feature):
    """
    Plot decision boundary visualization
    
    Args:
        model: Trained model
        X_test: Test features DataFrame
        y_test: Test labels
        selected_features: List of feature names
        x_feature: Feature name for x-axis
        y_feature: Feature name for y-axis
        
    Returns:
        Matplotlib figure
    """
    # Create meshgrid
    X_2d = X_test[[x_feature, y_feature]].values
    y_2d = y_test
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Get full feature set for prediction
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    # Create full feature array (use mean for other features)
    full_mesh = np.tile(X_test.mean().values, (len(mesh_points), 1))
    x_idx = selected_features.index(x_feature)
    y_idx = selected_features.index(y_feature)
    full_mesh[:, x_idx] = mesh_points[:, 0]
    full_mesh[:, y_idx] = mesh_points[:, 1]
    
    Z = model.predict(full_mesh)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    
    # Plot points
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdYlBu, edgecolors='black')
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"Decision Boundary - {type(model).__name__}")
    plt.colorbar(scatter, ax=ax)
    
    return fig


def plot_roc_curve(trained_models, X_test, y_test):
    """
    Plot ROC curve for binary classification
    
    Args:
        trained_models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Plotly figure or None
    """
    if len(np.unique(y_test)) != 2:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    
    for model_name, model_data in trained_models.items():
        try:
            model = model_data['model']
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{model_name.upper()} (AUC={auc:.3f})"
            ))
        except:
            pass
    
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title='ROC Curve',
        height=500
    )
    return fig


def plot_confusion_matrix(y_test, predictions, model_name):
    """
    Plot confusion matrix
    
    Args:
        y_test: True labels
        predictions: Predicted labels
        model_name: Name of the model
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{model_name.upper()} Confusion Matrix")
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    return fig


def plot_target_distribution(df, target_variable):
    """
    Plot target variable distribution
    
    Args:
        df: DataFrame
        target_variable: Name of target variable
        
    Returns:
        Plotly figure
    """
    target_counts = df[target_variable].value_counts()
    fig = px.bar(
        x=target_counts.index,
        y=target_counts.values,
        labels={'x': target_variable, 'y': 'Count'}
    )
    return fig
