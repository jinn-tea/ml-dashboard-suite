"""
Model training functions for all ML algorithms
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)


def train_knn(X_train, y_train, X_test, y_test, k, metric, weighted):
    """
    Train k-Nearest Neighbors model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        k: Number of neighbors
        metric: Distance metric ('Euclidean', 'Manhattan', 'Minkowski')
        weighted: Whether to use weighted voting
        
    Returns:
        Dictionary with model, accuracy, predictions, and config
    """
    metric_map = {
        "Euclidean": "euclidean",
        "Manhattan": "manhattan",
        "Minkowski": "minkowski"
    }
    
    model = KNeighborsClassifier(
        n_neighbors=k,
        metric=metric_map[metric],
        weights="distance" if weighted else "uniform"
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': predictions,
        'config': {'k': k, 'metric': metric, 'weighted': weighted}
    }


def train_random_forest(X_train, y_train, X_test, y_test, n_estimators, max_depth, criterion):
    """
    Train Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        criterion: Split criterion ('Gini' or 'Entropy')
        
    Returns:
        Dictionary with model, accuracy, predictions, and config
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion.lower(),
        random_state=42
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': predictions,
        'config': {'depth': max_depth, 'criterion': criterion, 'trees': n_estimators}
    }


def train_naive_bayes(X_train, y_train, X_test, y_test, nb_type, smoothing):
    """
    Train Naive Bayes model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        nb_type: Type of Naive Bayes ('Gaussian' or 'Multinomial')
        smoothing: Smoothing parameter
        
    Returns:
        Dictionary with model, accuracy, predictions, and config
    """
    if nb_type == "Gaussian":
        model = GaussianNB(var_smoothing=smoothing)
    else:
        model = MultinomialNB(alpha=smoothing)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': predictions,
        'config': {'type': nb_type, 'smoothing': smoothing}
    }


def calculate_metrics(y_test, predictions):
    """
    Calculate performance metrics
    
    Args:
        y_test: True labels
        predictions: Predicted labels
        
    Returns:
        Dictionary with precision, recall, and f1 score
    """
    return {
        'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
        'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
        'f1': f1_score(y_test, predictions, average='weighted', zero_division=0)
    }


def train_all_models(X_train, y_train, X_test, y_test, config):
    """
    Train all selected models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Dictionary with model configurations
        
    Returns:
        Dictionary with all trained models
    """
    trained_models = {}
    
    # Train k-NN
    if 'knn' in config:
        knn_config = config['knn']
        trained_models['knn'] = train_knn(
            X_train, y_train, X_test, y_test,
            knn_config['k'], knn_config['metric'], knn_config['weighted']
        )
        # Add metrics
        metrics = calculate_metrics(y_test, trained_models['knn']['predictions'])
        trained_models['knn'].update(metrics)
    
    # Train Random Forest
    if 'rf' in config:
        rf_config = config['rf']
        trained_models['rf'] = train_random_forest(
            X_train, y_train, X_test, y_test,
            rf_config['trees'], rf_config['depth'], rf_config['criterion']
        )
        # Add metrics
        metrics = calculate_metrics(y_test, trained_models['rf']['predictions'])
        trained_models['rf'].update(metrics)
    
    # Train Naive Bayes
    if 'nb' in config:
        nb_config = config['nb']
        trained_models['nb'] = train_naive_bayes(
            X_train, y_train, X_test, y_test,
            nb_config['type'], nb_config['smoothing']
        )
        # Add metrics
        metrics = calculate_metrics(y_test, trained_models['nb']['predictions'])
        trained_models['nb'].update(metrics)
    
    return trained_models
