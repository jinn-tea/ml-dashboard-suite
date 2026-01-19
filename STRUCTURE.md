# ML Dashboard Suite - Code Structure

This document explains the modular structure of the ML Dashboard Suite application.

## 📁 Folder Structure

```
ml-dashboard-suite/
├── app.py                 # Main Streamlit application (entry point)
├── requirements.txt       # Python dependencies
├── README.md             # User documentation
├── STRUCTURE.md          # This file - code structure documentation
├── run.sh                # Run script
└── src/                  # Source code modules
    ├── __init__.py       # Package initialization
    ├── config.py         # Configuration and constants
    ├── utils.py          # Utility functions and session state
    ├── data_processing.py # Data loading and preprocessing
    ├── model_training.py  # ML model training functions
    ├── visualizations.py  # Visualization functions
    ├── predictions.py    # Prediction functions
    └── pages/            # Page modules (for future expansion)
        └── __init__.py
```

## 📦 Module Descriptions

### `app.py` (Main Application)
- **Purpose**: Main entry point for the Streamlit application
- **Responsibilities**:
  - Page routing and navigation
  - UI layout and user interactions
  - Orchestrating calls to other modules
  - Displaying results to users

### `src/config.py`
- **Purpose**: Centralized configuration and constants
- **Contents**:
  - Color scheme definitions
  - Page configuration settings
  - Navigation menu structure
  - Algorithm default parameters
  - Custom CSS styles

### `src/utils.py`
- **Purpose**: Utility functions and session state management
- **Functions**:
  - `initialize_session_state()`: Initialize all session variables
  - `reset_session()`: Clear session state
  - `get_dataset_info()`: Get current dataset information

### `src/data_processing.py`
- **Purpose**: Data loading, preprocessing, and feature engineering
- **Functions**:
  - `load_dataset()`: Load CSV files
  - `get_feature_info()`: Get information about features
  - `preprocess_features()`: Encode categorical features and handle missing values
  - `prepare_training_data()`: Prepare data for model training

### `src/model_training.py`
- **Purpose**: Machine learning model training and evaluation
- **Functions**:
  - `train_knn()`: Train k-Nearest Neighbors model
  - `train_random_forest()`: Train Random Forest model
  - `train_naive_bayes()`: Train Naive Bayes model
  - `calculate_metrics()`: Calculate performance metrics
  - `train_all_models()`: Train all selected models

### `src/visualizations.py`
- **Purpose**: All visualization and plotting functions
- **Functions**:
  - `plot_model_comparison()`: Bar chart comparing model accuracies
  - `plot_feature_importance()`: Feature importance plot for tree models
  - `plot_decision_boundary()`: Decision boundary visualization
  - `plot_roc_curve()`: ROC curve for binary classification
  - `plot_confusion_matrix()`: Confusion matrix heatmap
  - `plot_target_distribution()`: Target variable distribution

### `src/predictions.py`
- **Purpose**: Prediction functionality for manual and batch predictions
- **Functions**:
  - `prepare_prediction_input()`: Prepare input array with encoding
  - `predict_manual()`: Make predictions from manual input
  - `predict_batch()`: Make batch predictions from CSV
  - `export_predictions()`: Export predictions to CSV

## 🔄 Data Flow

1. **Data Upload** → `data_processing.load_dataset()`
2. **Feature Selection** → User selects features in UI
3. **Data Preprocessing** → `data_processing.prepare_training_data()`
4. **Model Training** → `model_training.train_all_models()`
5. **Visualization** → `visualizations.plot_*()` functions
6. **Predictions** → `predictions.predict_manual()` or `predictions.predict_batch()`

## 🎯 Benefits of Modular Structure

1. **Maintainability**: Each module has a single, clear responsibility
2. **Readability**: Code is organized logically and easy to navigate
3. **Testability**: Individual functions can be tested in isolation
4. **Reusability**: Functions can be reused across different parts of the application
5. **Scalability**: Easy to add new features or algorithms
6. **Documentation**: Each module is self-contained and easier to document

## 🔧 Adding New Features

### To add a new algorithm:
1. Add training function to `src/model_training.py`
2. Add configuration UI in `app.py` (Dashboard page)
3. Update `train_all_models()` to include new algorithm

### To add a new visualization:
1. Add plotting function to `src/visualizations.py`
2. Add UI section in `app.py` (Visualizations page)
3. Call the new function from the UI

### To add a new page:
1. Create page module in `src/pages/`
2. Add page to `PAGES` in `src/config.py`
3. Add routing logic in `app.py`

## 📝 Code Style

- Functions are well-documented with docstrings
- Type hints can be added for better IDE support
- Error handling is implemented at appropriate levels
- Constants are defined in `config.py`
- Session state is managed through `utils.py`

## 🚀 Running the Application

The modular structure doesn't change how you run the application:

```bash
python3 -m streamlit run app.py
```

Or use the run script:
```bash
./run.sh
```
