# ML Dashboard Suite v2.0

Interactive Machine Learning Dashboard built with Python and Streamlit that allows users to upload datasets, train classification models, visualize performance metrics, and make predictions through a user-friendly GUI.

## Features

### 📊 Core Functionality
- **Dataset Management**: Upload CSV files, preview data, view statistics, and select features/target variables
- **Algorithm Selection**: Support for k-NN, Decision Tree, Random Forest, and Naive Bayes
- **Model Training**: Configure hyperparameters, train/test split, cross-validation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve
- **Visualizations**: Model comparison charts, feature importance plots, decision boundary visualizations
- **Predictions**: Manual input predictions and batch predictions via CSV upload
- **Model History**: Track all trained models with performance metrics

### 🎨 Design
- Modern, clean interface with blue color scheme (#2563eb)
- Responsive layout with sidebar navigation
- Interactive charts using Plotly
- Real-time progress indicators

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run directly with Streamlit (Recommended)

1. **Open Terminal** (Press `Cmd + Space`, type "Terminal", press Enter)

2. **Navigate to the project directory**:
```bash
cd "/Users/abdullah/ML Assignment 4/ml-dashboard-suite"
```

3. **Install dependencies** (if not already installed):
```bash
pip install -r requirements.txt
```

4. **Run the application**:
```bash
streamlit run app.py
```

5. **Access the dashboard**:
   - The app will automatically open in your default web browser at `http://localhost:8501`
   - If it doesn't open automatically, navigate to `http://localhost:8501` in your browser

### Option 2: Use the run script

1. **Open Terminal**

2. **Navigate to the project directory**:
```bash
cd "/Users/abdullah/ML Assignment 4/ml-dashboard-suite"
```

3. **Make the script executable** (first time only):
```bash
chmod +x run.sh
```

4. **Run the script**:
```bash
./run.sh
```

**Note**: On macOS, double-clicking `.sh` files opens them in a text editor. Always run shell scripts from the Terminal.

3. **Workflow**:
   - **Dashboard**: Upload your dataset (CSV format), select features and target variable, configure algorithms, and train models
   - **Data Explorer**: View detailed statistics and data preview
   - **Model Training**: See training metrics and confusion matrices
   - **Visualizations**: Explore model comparisons, feature importance, and decision boundaries
   - **Predictions**: Make predictions on new data (manual or batch)
   - **Model History**: View history of all trained models
   - **Settings**: Export/import models

## Supported Datasets

The dashboard works with any CSV dataset. Recommended datasets for testing:

1. **Iris Dataset** - 4 features, 3 classes (classification)
2. **Wine Quality Dataset** - Multiple features, multi-class classification
3. **Breast Cancer Wisconsin** - Binary classification
4. **Any custom CSV dataset** with features and a target column

## Algorithm Configuration

### k-Nearest Neighbors (k-NN)
- **k**: Number of neighbors (1-20)
- **Metric**: Euclidean, Manhattan, or Minkowski distance
- **Weighted Voting**: Enable/disable distance-weighted voting

### Decision Tree / Random Forest
- **Max Depth**: Maximum tree depth (1-20)
- **Criterion**: Gini or Entropy
- **n_estimators**: Number of trees for Random Forest (1-100)

### Naive Bayes
- **Type**: Gaussian or Multinomial
- **Smoothing**: Variance smoothing parameter

## Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **ML Libraries**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, Matplotlib, Seaborn

## Project Structure

```
ml-dashboard-suite/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Requirements

- Python 3.8 or higher
- All dependencies listed in `requirements.txt`

## Notes

- The application uses Streamlit's session state to maintain data and models across page navigation
- Models are stored in memory during the session
- Use "NEW SESSION" button to reset all data and models
- For production use, consider adding model persistence to disk/database

## License

This project is created for educational purposes as part of Assignment 4: Interactive Machine Learning Dashboard with GUI.
