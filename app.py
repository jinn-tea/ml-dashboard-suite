import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ML Dashboard Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
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
""", unsafe_allow_html=True)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_history' not in st.session_state:
    st.session_state.model_history = []
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'target_variable' not in st.session_state:
    st.session_state.target_variable = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"
if 'feature_encoders' not in st.session_state:
    st.session_state.feature_encoders = {}

# Sidebar Navigation
st.sidebar.markdown("## 📊 MACHINE LEARNING DASHBOARD v2.0")
st.sidebar.markdown("---")

pages = {
    "Dashboard": "🏠",
    "Data Explorer": "🔍",
    "Model Training": "⚙️",
    "Visualizations": "📈",
    "Predictions": "🔮",
    "Model History": "📚",
    "Settings": "⚙️"
}

selected_page = st.sidebar.radio(
    "Navigation",
    options=list(pages.keys()),
    format_func=lambda x: f"{pages[x]} {x}",
    key="page_selector"
)

st.session_state.current_page = selected_page

# Display dataset info in sidebar
if st.session_state.df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Dataset")
    st.sidebar.info(f"**{st.session_state.dataset}**\n\n"
                   f"Rows: {len(st.session_state.df)} | "
                   f"Features: {len(st.session_state.df.columns)} | "
                   f"Classes: {st.session_state.df[st.session_state.target_variable].nunique() if st.session_state.target_variable else 'N/A'}")

if st.sidebar.button("🔄 NEW SESSION", use_container_width=True):
    for key in list(st.session_state.keys()):
        if key != 'current_page':
            del st.session_state[key]
    st.session_state.current_page = "Dashboard"
    st.rerun()

# Dashboard Page
if selected_page == "Dashboard":
    st.markdown('<div class="main-header">MACHINE LEARNING DASHBOARD</div>', unsafe_allow_html=True)
    
    # Dataset Upload Section
    st.markdown("### 📁 DATASET UPLOAD")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="file_uploader")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.dataset = uploaded_file.name
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            st.markdown("#### Data Preview (First 5 rows)")
            st.dataframe(df.head(), use_container_width=True)
            
            # Feature and Target Selection
            st.markdown("#### Feature & Target Selection")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.selected_features = st.multiselect(
                    "Select Features",
                    options=df.columns.tolist(),
                    default=df.columns.tolist()[:-1] if len(df.columns) > 1 else df.columns.tolist()
                )
            with col2:
                st.session_state.target_variable = st.selectbox(
                    "Select Target Variable",
                    options=df.columns.tolist()
                )
            
            if st.session_state.selected_features and st.session_state.target_variable:
                if st.button("✅ Process Dataset", type="primary"):
                    st.success("Dataset processed successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Algorithm Selection
    if st.session_state.df is not None and st.session_state.selected_features:
        st.markdown("---")
        st.markdown("### 🤖 ALGORITHM SELECTION")
        
        col1, col2, col3, col4 = st.columns(4)
        algorithms = {}
        
        with col1:
            algorithms['knn'] = st.button("k-NN", use_container_width=True, type="primary" if 'knn' not in st.session_state.trained_models else "secondary")
        with col2:
            algorithms['dt'] = st.button("DECISION TREE", use_container_width=True)
        with col3:
            algorithms['rf'] = st.button("RANDOM FOREST", use_container_width=True, type="primary" if 'rf' not in st.session_state.trained_models else "secondary")
        with col4:
            algorithms['nb'] = st.button("NAIVE BAYES", use_container_width=True)
        
        # Algorithm Configuration
        st.markdown("---")
        st.markdown("### ⚙️ ALGORITHM CONFIGURATION")
        
        config_cols = st.columns(4)
        
        with config_cols[0]:
            st.markdown("#### k-NN")
            knn_k = st.slider("k", 1, 20, 5, key="knn_k")
            knn_metric = st.selectbox("Metric", ["Euclidean", "Manhattan", "Minkowski"], key="knn_metric")
            knn_weighted = st.checkbox("Weighted Voting", key="knn_weighted")
        
        with config_cols[1]:
            st.markdown("#### DT/RF")
            dt_depth = st.slider("Depth", 1, 20, 10, key="dt_depth")
            dt_criterion = st.radio("Criterion", ["Gini", "Entropy"], key="dt_criterion")
            rf_trees = st.slider("n_trees", 1, 100, 100, key="rf_trees")
        
        with config_cols[2]:
            st.markdown("#### NB")
            nb_type = st.radio("Type", ["Gaussian", "Multinomial"], key="nb_type")
            nb_smoothing = st.number_input("Var Smooth", value=1e-9, format="%e", key="nb_smoothing")
        
        # Training Settings
        st.markdown("---")
        st.markdown("### 🎯 TRAINING SETTINGS")
        
        train_test_split_ratio = st.slider("Train/Test Split", 60, 80, 70, format="%d%%")
        use_cv = st.checkbox("Cross-Validation", value=True)
        cv_folds = st.number_input("K-folds", min_value=2, max_value=10, value=5, disabled=not use_cv)
        
        if st.button("🚀 TRAIN MODEL", type="primary", use_container_width=True):
            if not st.session_state.selected_features or not st.session_state.target_variable:
                st.error("Please select features and target variable first!")
            else:
                with st.spinner("Training models..."):
                    progress_bar = st.progress(0)
                    
                    # Prepare data
                    X = st.session_state.df[st.session_state.selected_features].copy()
                    y = st.session_state.df[st.session_state.target_variable].copy()
                    
                    # Check and encode categorical features
                    categorical_features = []
                    numerical_features = []
                    feature_encoders = {}
                    
                    for feature in st.session_state.selected_features:
                        if X[feature].dtype == 'object' or X[feature].dtype.name == 'category':
                            categorical_features.append(feature)
                            # Use LabelEncoder for categorical features
                            le = LabelEncoder()
                            X[feature] = le.fit_transform(X[feature].astype(str))
                            feature_encoders[feature] = le
                        else:
                            numerical_features.append(feature)
                            # Convert to numeric, handling any remaining non-numeric values
                            X[feature] = pd.to_numeric(X[feature], errors='coerce')
                    
                    # Fill any NaN values created during conversion
                    X = X.fillna(X.mean(numeric_only=True))
                    
                    # Encode target if needed
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y.astype(str))
                    
                    # Train-test split
                    test_size = (100 - train_test_split_ratio) / 100
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.label_encoder = le
                    st.session_state.feature_encoders = feature_encoders
                    
                    # Show encoding info
                    if categorical_features:
                        st.info(f"📝 Encoded {len(categorical_features)} categorical feature(s): {', '.join(categorical_features)}")
                    
                    # Train k-NN
                    progress_bar.progress(25)
                    metric_map = {"Euclidean": "euclidean", "Manhattan": "manhattan", "Minkowski": "minkowski"}
                    knn_model = KNeighborsClassifier(
                        n_neighbors=knn_k,
                        metric=metric_map[knn_metric],
                        weights="distance" if knn_weighted else "uniform"
                    )
                    knn_model.fit(X_train, y_train)
                    knn_pred = knn_model.predict(X_test)
                    knn_acc = accuracy_score(y_test, knn_pred)
                    
                    st.session_state.trained_models['knn'] = {
                        'model': knn_model,
                        'accuracy': knn_acc,
                        'predictions': knn_pred,
                        'config': {'k': knn_k, 'metric': knn_metric, 'weighted': knn_weighted}
                    }
                    
                    # Train Random Forest
                    progress_bar.progress(50)
                    rf_model = RandomForestClassifier(
                        n_estimators=rf_trees,
                        max_depth=dt_depth,
                        criterion=dt_criterion.lower(),
                        random_state=42
                    )
                    rf_model.fit(X_train, y_train)
                    rf_pred = rf_model.predict(X_test)
                    rf_acc = accuracy_score(y_test, rf_pred)
                    
                    st.session_state.trained_models['rf'] = {
                        'model': rf_model,
                        'accuracy': rf_acc,
                        'predictions': rf_pred,
                        'config': {'depth': dt_depth, 'criterion': dt_criterion, 'trees': rf_trees}
                    }
                    
                    # Train Naive Bayes
                    progress_bar.progress(75)
                    if nb_type == "Gaussian":
                        nb_model = GaussianNB(var_smoothing=nb_smoothing)
                    else:
                        nb_model = MultinomialNB(alpha=nb_smoothing)
                    nb_model.fit(X_train, y_train)
                    nb_pred = nb_model.predict(X_test)
                    nb_acc = accuracy_score(y_test, nb_pred)
                    
                    st.session_state.trained_models['nb'] = {
                        'model': nb_model,
                        'accuracy': nb_acc,
                        'predictions': nb_pred,
                        'config': {'type': nb_type, 'smoothing': nb_smoothing}
                    }
                    
                    progress_bar.progress(100)
                    
                    # Calculate metrics
                    for model_name in ['knn', 'rf', 'nb']:
                        if model_name in st.session_state.trained_models:
                            pred = st.session_state.trained_models[model_name]['predictions']
                            st.session_state.trained_models[model_name]['precision'] = precision_score(y_test, pred, average='weighted', zero_division=0)
                            st.session_state.trained_models[model_name]['recall'] = recall_score(y_test, pred, average='weighted', zero_division=0)
                            st.session_state.trained_models[model_name]['f1'] = f1_score(y_test, pred, average='weighted', zero_division=0)
                    
                    # Add to history
                    for model_name, model_data in st.session_state.trained_models.items():
                        st.session_state.model_history.append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'model': model_name.upper(),
                            'accuracy': f"{model_data['accuracy']*100:.2f}%",
                            'dataset': st.session_state.dataset
                        })
                    
                    st.success("✅ Models trained successfully!")
                    st.rerun()

# Data Explorer Page
elif selected_page == "Data Explorer":
    st.markdown('<div class="main-header">DATA EXPLORER</div>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        st.markdown("### Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(st.session_state.df))
        with col2:
            st.metric("Total Columns", len(st.session_state.df.columns))
        with col3:
            st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", st.session_state.df.duplicated().sum())
        
        st.markdown("### Data Preview")
        num_rows = st.slider("Number of rows to display", 5, 100, 10)
        st.dataframe(st.session_state.df.head(num_rows), use_container_width=True)
        
        st.markdown("### Statistical Summary")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        st.markdown("### Data Types")
        dtype_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes.astype(str),
            'Non-Null Count': st.session_state.df.count().values
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        if st.session_state.target_variable:
            st.markdown("### Target Variable Distribution")
            target_counts = st.session_state.df[st.session_state.target_variable].value_counts()
            fig = px.bar(x=target_counts.index, y=target_counts.values, 
                        labels={'x': st.session_state.target_variable, 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload a dataset first from the Dashboard page.")

# Model Training Page
elif selected_page == "Model Training":
    st.markdown('<div class="main-header">MODEL TRAINING</div>', unsafe_allow_html=True)
    
    if st.session_state.df is not None and st.session_state.X_train is not None:
        if st.session_state.trained_models:
            st.markdown("### 📊 Training Metrics")
            
            metrics_cols = st.columns(len(st.session_state.trained_models))
            for idx, (model_name, model_data) in enumerate(st.session_state.trained_models.items()):
                with metrics_cols[idx]:
                    st.metric(
                        f"{model_name.upper()} Accuracy",
                        f"{model_data['accuracy']*100:.2f}%"
                    )
            
            st.markdown("### Detailed Performance Metrics")
            
            metrics_data = []
            for model_name, model_data in st.session_state.trained_models.items():
                metrics_data.append({
                    'Model': model_name.upper(),
                    'Training Accuracy': f"{model_data['accuracy']*100:.2f}%",
                    'Testing Accuracy': f"{model_data['accuracy']*100:.2f}%",
                    'Precision': f"{model_data.get('precision', 0)*100:.2f}%",
                    'Recall': f"{model_data.get('recall', 0)*100:.2f}%",
                    'F1 Score': f"{model_data.get('f1', 0)*100:.2f}%"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Confusion Matrices
            st.markdown("### Confusion Matrices")
            cm_cols = st.columns(len(st.session_state.trained_models))
            
            for idx, (model_name, model_data) in enumerate(st.session_state.trained_models.items()):
                with cm_cols[idx]:
                    cm = confusion_matrix(st.session_state.y_test, model_data['predictions'])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f"{model_name.upper()} Confusion Matrix")
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    st.pyplot(fig)
        else:
            st.info("No models trained yet. Please train models from the Dashboard page.")
    else:
        st.warning("Please upload a dataset and train models first from the Dashboard page.")

# Visualizations Page
elif selected_page == "Visualizations":
    st.markdown('<div class="main-header">VISUALIZATION DASHBOARD</div>', unsafe_allow_html=True)
    
    if st.session_state.trained_models:
        # Model Accuracy Comparison
        st.markdown("### Model Accuracy Comparison")
        model_names = list(st.session_state.trained_models.keys())
        accuracies = [st.session_state.trained_models[m]['accuracy']*100 for m in model_names]
        
        fig = px.bar(
            x=model_names,
            y=accuracies,
            labels={'x': 'Model', 'y': 'Accuracy (%)'},
            color=accuracies,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance (for tree-based models)
        if 'rf' in st.session_state.trained_models:
            st.markdown("### Feature Importance (Top 10)")
            rf_model = st.session_state.trained_models['rf']['model']
            if hasattr(rf_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.selected_features,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
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
                st.plotly_chart(fig, use_container_width=True)
        
        # Decision Boundary Visualization
        if len(st.session_state.selected_features) >= 2:
            st.markdown("### Decision Boundary Visualization (Select Features)")
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis Feature", st.session_state.selected_features, key="x_feat")
            with col2:
                y_feature = st.selectbox("Y-axis Feature", st.session_state.selected_features, key="y_feat")
            
            if x_feature and y_feature and x_feature != y_feature:
                # Create meshgrid
                X_2d = st.session_state.X_test[[x_feature, y_feature]].values
                y_2d = st.session_state.y_test
                
                # Plot decision boundary for first available model
                model_name = list(st.session_state.trained_models.keys())[0]
                model = st.session_state.trained_models[model_name]['model']
                
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
                full_mesh = np.tile(st.session_state.X_test.mean().values, (len(mesh_points), 1))
                x_idx = st.session_state.selected_features.index(x_feature)
                y_idx = st.session_state.selected_features.index(y_feature)
                full_mesh[:, x_idx] = mesh_points[:, 0]
                full_mesh[:, y_idx] = mesh_points[:, 1]
                
                Z = model.predict(full_mesh)
                Z = Z.reshape(xx.shape)
                
                ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
                
                # Plot points
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdYlBu, edgecolors='black')
                ax.set_xlabel(x_feature)
                ax.set_ylabel(y_feature)
                ax.set_title(f"Decision Boundary - {model_name.upper()}")
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig)
        
        # ROC Curve (for binary classification)
        if len(np.unique(st.session_state.y_test)) == 2:
            st.markdown("### ROC Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                    name='Random', line=dict(dash='dash', color='gray')))
            
            for model_name, model_data in st.session_state.trained_models.items():
                try:
                    model = model_data['model']
                    y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
                    auc = roc_auc_score(st.session_state.y_test, y_pred_proba)
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                            name=f"{model_name.upper()} (AUC={auc:.3f})"))
                except:
                    pass
            
            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                title='ROC Curve',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please train models first from the Dashboard page.")

# Predictions Page
elif selected_page == "Predictions":
    st.markdown('<div class="main-header">MAKE PREDICTIONS</div>', unsafe_allow_html=True)
    
    if st.session_state.trained_models and st.session_state.selected_features:
        # Manual Input
        st.markdown("### Manual Input")
        
        input_values = {}
        cols = st.columns(len(st.session_state.selected_features))
        for idx, feature in enumerate(st.session_state.selected_features):
            with cols[idx]:
                # Check if feature is categorical
                if feature in st.session_state.feature_encoders:
                    # Get unique values for categorical feature
                    unique_vals = st.session_state.df[feature].unique().tolist()
                    input_values[feature] = st.selectbox(
                        feature,
                        options=unique_vals,
                        key=f"pred_{feature}"
                    )
                else:
                    # Numerical feature
                    mean_val = float(st.session_state.df[feature].mean()) if feature in st.session_state.df.columns else 0.0
                    input_values[feature] = st.number_input(
                        feature,
                        value=mean_val,
                        key=f"pred_{feature}"
                    )
        
        if st.button("🔮 PREDICT", type="primary"):
            # Prepare input array with proper encoding
            input_row = []
            for feature in st.session_state.selected_features:
                if feature in st.session_state.feature_encoders:
                    # Encode categorical feature
                    encoder = st.session_state.feature_encoders[feature]
                    try:
                        encoded_val = encoder.transform([str(input_values[feature])])[0]
                        input_row.append(encoded_val)
                    except:
                        # If value not seen during training, use most common value
                        input_row.append(0)
                else:
                    input_row.append(float(input_values[feature]))
            
            input_array = np.array([input_row])
            
            predictions = {}
            for model_name, model_data in st.session_state.trained_models.items():
                model = model_data['model']
                pred = model.predict(input_array)[0]
                pred_proba = model.predict_proba(input_array)[0]
                confidence = max(pred_proba) * 100
                pred_class = st.session_state.label_encoder.inverse_transform([pred])[0]
                predictions[model_name] = {
                    'class': pred_class,
                    'confidence': confidence
                }
            
            # Display results
            st.success(f"**Result:** {predictions[list(predictions.keys())[0]]['class']} "
                      f"({predictions[list(predictions.keys())[0]]['confidence']:.1f}% confidence)")
            
            st.markdown("#### Model Predictions:")
            for model_name, pred_data in predictions.items():
                st.write(f"**{model_name.upper()}:** {pred_data['class']} ({pred_data['confidence']:.1f}%)")
        
        st.markdown("---")
        
        # Batch Prediction
        st.markdown("### Batch Prediction (Upload CSV)")
        batch_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'], key="batch_upload")
        
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                
                # Check if required features are present
                missing_features = [f for f in st.session_state.selected_features if f not in batch_df.columns]
                if missing_features:
                    st.error(f"Missing features: {', '.join(missing_features)}")
                else:
                    batch_X = batch_df[st.session_state.selected_features]
                    
                    batch_predictions = {}
                    for model_name, model_data in st.session_state.trained_models.items():
                        model = model_data['model']
                        preds = model.predict(batch_X)
                        pred_probas = model.predict_proba(batch_X)
                        confidences = np.max(pred_probas, axis=1) * 100
                        pred_classes = st.session_state.label_encoder.inverse_transform(preds)
                        batch_predictions[f"{model_name.upper()}_Pred"] = pred_classes
                        batch_predictions[f"{model_name.upper()}_Confidence"] = confidences
                    
                    # Create results dataframe
                    results_df = batch_df.copy()
                    for key, values in batch_predictions.items():
                        results_df[key] = values
                    
                    # Determine final prediction (majority vote)
                    pred_cols = [col for col in results_df.columns if col.endswith('_Pred')]
                    results_df['Final_Pred'] = results_df[pred_cols].mode(axis=1)[0]
                    results_df['Final_Confidence'] = results_df[[col for col in results_df.columns if col.endswith('_Confidence')]].mean(axis=1)
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 DOWNLOAD RESULTS",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing batch file: {str(e)}")
    else:
        st.warning("Please train models first from the Dashboard page.")

# Model History Page
elif selected_page == "Model History":
    st.markdown('<div class="main-header">MODEL HISTORY</div>', unsafe_allow_html=True)
    
    if st.session_state.model_history:
        history_df = pd.DataFrame(st.session_state.model_history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.model_history = []
            st.rerun()
    else:
        st.info("No model history available yet.")

# Settings Page
elif selected_page == "Settings":
    st.markdown('<div class="main-header">SETTINGS</div>', unsafe_allow_html=True)
    
    st.markdown("### Application Settings")
    st.info("""
    **ML Dashboard Suite v2.0**
    
    - Built with Streamlit
    - Supports k-NN, Decision Tree, Random Forest, and Naive Bayes algorithms
    - Interactive visualizations with Plotly
    - Real-time model training and evaluation
    """)
    
    st.markdown("### Export/Import Models")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.trained_models:
            model_to_export = st.selectbox("Select model to export", list(st.session_state.trained_models.keys()))
            if st.button("💾 Export Model"):
                model_data = st.session_state.trained_models[model_to_export]
                pickle_data = pickle.dumps(model_data['model'])
                st.download_button(
                    label="Download Model",
                    data=pickle_data,
                    file_name=f"{model_to_export}_model.pkl",
                    mime="application/octet-stream"
                )
    
    with col2:
        uploaded_model = st.file_uploader("Import Model", type=['pkl'])
        if uploaded_model:
            try:
                model = pickle.load(uploaded_model)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
