"""
Configuration and constants for ML Dashboard Suite
"""

# Color Scheme
COLORS = {
    'primary': '#2563eb',
    'primary_hover': '#1d4ed8',
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'background': '#f3f4f6',
}

# Page Configuration
PAGE_CONFIG = {
    'page_title': "ML Dashboard Suite",
    'page_icon': "📊",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Navigation Pages
PAGES = {
    "Dashboard": "🏠",
    "Data Explorer": "🔍",
    "Model Training": "⚙️",
    "Visualizations": "📈",
    "Predictions": "🔮",
    "Model History": "📚",
    "Settings": "⚙️"
}

# Algorithm Defaults
ALGORITHM_DEFAULTS = {
    'knn': {
        'k': 5,
        'metric': 'Euclidean',
        'weighted': False
    },
    'dt_rf': {
        'depth': 10,
        'criterion': 'Gini',
        'n_trees': 100
    },
    'nb': {
        'type': 'Gaussian',
        'smoothing': 1e-9
    }
}

# Training Defaults
TRAINING_DEFAULTS = {
    'train_test_split': 70,
    'use_cv': True,
    'cv_folds': 5
}

# Custom CSS Styles
CUSTOM_CSS = """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2563eb;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .success-metric {
        color: #10b981;
        font-weight: bold;
    }
    .warning-metric {
        color: #f59e0b;
        font-weight: bold;
    }
    .error-metric {
        color: #ef4444;
        font-weight: bold;
    }
    </style>
"""
