"""
ML Dashboard Suite - Main Application
Interactive Machine Learning Dashboard built with Streamlit
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import modules
from src.config import PAGE_CONFIG, PAGES, CUSTOM_CSS, COLORS
from src.utils import initialize_session_state, reset_session, get_dataset_info
from src.data_processing import (
    load_dataset, get_feature_info, prepare_training_data
)
from src.memory_utils import check_dataset_size, optimize_dataframe, clear_large_objects, cleanup_memory
from src.model_training import train_all_models
from src.visualizations import (
    plot_model_comparison, plot_feature_importance, plot_decision_boundary,
    plot_roc_curve, plot_confusion_matrix, plot_target_distribution
)
from src.predictions import (
    prepare_prediction_input, predict_manual, predict_batch, export_predictions
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Sidebar Navigation
st.sidebar.markdown("## 📊 MACHINE LEARNING DASHBOARD v2.0")
st.sidebar.markdown("---")

selected_page = st.sidebar.radio(
    "Navigation",
    options=list(PAGES.keys()),
    format_func=lambda x: f"{PAGES[x]} {x}",
    key="page_selector"
)

st.session_state.current_page = selected_page

# Display dataset info in sidebar
dataset_info = get_dataset_info()
if dataset_info:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Dataset")
    st.sidebar.info(
        f"**{dataset_info['name']}**\n\n"
        f"Rows: {dataset_info['rows']} | "
        f"Features: {dataset_info['features']} | "
        f"Classes: {dataset_info['classes']}"
    )

if st.sidebar.button("🔄 NEW SESSION", use_container_width=True):
    reset_session()
    cleanup_memory()
    st.rerun()

# Memory management section in sidebar
st.sidebar.markdown("---")
with st.sidebar.expander("🔧 Memory Management"):
    if st.button("🗑️ Clear Large Objects"):
        cleared = clear_large_objects()
        if cleared:
            st.success(f"Cleared: {', '.join(cleared)}")
        else:
            st.info("No large objects to clear")
    
    if st.button("🧹 Force Garbage Collection"):
        cleanup_memory()
        st.success("Memory cleaned!")
    
    # Show memory usage if psutil is available
    try:
        from src.memory_utils import get_memory_usage
        memory_mb = get_memory_usage()
        if memory_mb:
            st.info(f"Memory Usage: {memory_mb:.1f} MB")
    except:
        pass

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
if selected_page == "Dashboard":
    st.markdown('<div class="main-header">MACHINE LEARNING DASHBOARD</div>', unsafe_allow_html=True)
    
    # Dataset Upload Section
    st.markdown("### 📁 DATASET UPLOAD")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="file_uploader")
    
    if uploaded_file is not None:
        df, error = load_dataset(uploaded_file, max_size_mb=500)
        if error:
            st.error(f"Error loading file: {error}")
        else:
            # Check dataset size
            is_valid, size_error = check_dataset_size(df, max_rows=100000, max_cols=100)
            if not is_valid:
                st.error(size_error)
            else:
                # Optimize dataframe memory
                df = optimize_dataframe(df)
                st.session_state.df = df
                st.session_state.dataset = uploaded_file.name
                
                # Force garbage collection after loading
                cleanup_memory()
            
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
    
    # Algorithm Selection and Training
    if st.session_state.df is not None and st.session_state.selected_features:
        st.markdown("---")
        st.markdown("### 🤖 ALGORITHM SELECTION")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button("k-NN", use_container_width=True, type="primary" if 'knn' not in st.session_state.trained_models else "secondary")
        with col2:
            st.button("DECISION TREE", use_container_width=True)
        with col3:
            st.button("RANDOM FOREST", use_container_width=True, type="primary" if 'rf' not in st.session_state.trained_models else "secondary")
        with col4:
            st.button("NAIVE BAYES", use_container_width=True)
        
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
        
        # Data Type Info
        if st.session_state.selected_features:
            st.markdown("---")
            st.markdown("### 📋 Selected Features Info")
            info_df = get_feature_info(st.session_state.df, st.session_state.selected_features)
            st.dataframe(info_df, use_container_width=True, hide_index=True)
            if not all(pd.api.types.is_numeric_dtype(st.session_state.df[feat]) for feat in st.session_state.selected_features):
                st.info("ℹ️ Some features are text/categorical and will be automatically encoded during training.")
        
        # Training Settings
        st.markdown("---")
        st.markdown("### 🎯 TRAINING SETTINGS")
        
        train_test_split_ratio = st.slider("Train/Test Split", 60, 80, 70, format="%d%%")
        use_cv = st.checkbox("Cross-Validation", value=True)
        cv_folds = st.number_input("K-folds", min_value=2, max_value=10, value=5, disabled=not use_cv)
        
        if st.button("🚀 TRAIN MODEL", type="primary", use_container_width=True):
            if not st.session_state.selected_features or not st.session_state.target_variable:
                st.error("Please select features and target variable first!")
            elif st.session_state.target_variable in st.session_state.selected_features:
                st.error("Target variable cannot be in the feature list! Please remove it from selected features.")
            else:
                try:
                    with st.spinner("Training models..."):
                        progress_bar = st.progress(0)
                        
                        # Prepare training data
                        X_train, X_test, y_train, y_test, le, feature_encoders, categorical_features = prepare_training_data(
                            st.session_state.df,
                            st.session_state.selected_features,
                            st.session_state.target_variable,
                            train_test_split_ratio
                        )
                        
                        # Store in session state
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.label_encoder = le
                        st.session_state.feature_encoders = feature_encoders
                        
                        # Show encoding info
                        if categorical_features:
                            st.info(f"📝 Encoded {len(categorical_features)} categorical feature(s): {', '.join(categorical_features)}")
                        
                        # Prepare model configurations
                        model_config = {
                            'knn': {
                                'k': knn_k,
                                'metric': knn_metric,
                                'weighted': knn_weighted
                            },
                            'rf': {
                                'trees': rf_trees,
                                'depth': dt_depth,
                                'criterion': dt_criterion
                            },
                            'nb': {
                                'type': nb_type,
                                'smoothing': nb_smoothing
                            }
                        }
                        
                        # Train all models
                        progress_bar.progress(25)
                        trained_models = train_all_models(X_train, y_train, X_test, y_test, model_config)
                        progress_bar.progress(100)
                        
                        st.session_state.trained_models = trained_models
                        
                        # Add to history
                        for model_name, model_data in trained_models.items():
                            st.session_state.model_history.append({
                                'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'model': model_name.upper(),
                                'accuracy': f"{model_data['accuracy']*100:.2f}%",
                                'dataset': st.session_state.dataset
                            })
                        
                        st.success("✅ Models trained successfully!")
                        st.rerun()
                        
                except ValueError as e:
                    error_msg = str(e)
                    if "could not convert string to float" in error_msg:
                        st.error("❌ **Data Type Error**: Some selected features contain text/string values that cannot be converted to numbers. "
                                "Please either:\n"
                                "1. Remove text columns from your feature selection, OR\n"
                                "2. Ensure all selected features are numeric or can be encoded as categories.\n\n"
                                f"Error details: {error_msg}")
                    else:
                        st.error(f"❌ **Training Error**: {error_msg}")
                except Exception as e:
                    st.error(f"❌ **Unexpected Error**: {str(e)}\n\nPlease check your data and try again.")
                    st.exception(e)

# ============================================================================
# DATA EXPLORER PAGE
# ============================================================================
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
            fig = plot_target_distribution(st.session_state.df, st.session_state.target_variable)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload a dataset first from the Dashboard page.")

# ============================================================================
# MODEL TRAINING PAGE
# ============================================================================
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
                    # Recompute predictions if not stored (memory optimization)
                    if 'predictions' in model_data:
                        predictions = model_data['predictions']
                    else:
                        predictions = model_data['model'].predict(st.session_state.X_test)
                    
                    fig = plot_confusion_matrix(
                        st.session_state.y_test,
                        predictions,
                        model_name
                    )
                    st.pyplot(fig)
                    plt.close(fig)  # Close figure to free memory
        else:
            st.info("No models trained yet. Please train models from the Dashboard page.")
    else:
        st.warning("Please upload a dataset and train models first from the Dashboard page.")

# ============================================================================
# VISUALIZATIONS PAGE
# ============================================================================
elif selected_page == "Visualizations":
    st.markdown('<div class="main-header">VISUALIZATION DASHBOARD</div>', unsafe_allow_html=True)
    
    if st.session_state.trained_models:
        # Model Accuracy Comparison
        st.markdown("### Model Accuracy Comparison")
        fig = plot_model_comparison(st.session_state.trained_models)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance (for tree-based models)
        if 'rf' in st.session_state.trained_models:
            st.markdown("### Feature Importance (Top 10)")
            rf_model = st.session_state.trained_models['rf']['model']
            fig = plot_feature_importance(rf_model, st.session_state.selected_features)
            if fig:
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
                model_name = list(st.session_state.trained_models.keys())[0]
                model = st.session_state.trained_models[model_name]['model']
                fig = plot_decision_boundary(
                    model,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    st.session_state.selected_features,
                    x_feature,
                    y_feature,
                    max_points=5000  # Limit points for memory optimization
                )
                st.pyplot(fig)
                plt.close(fig)  # Close figure to free memory
        
        # ROC Curve (for binary classification)
        fig = plot_roc_curve(
            st.session_state.trained_models,
            st.session_state.X_test,
            st.session_state.y_test
        )
        if fig:
            st.markdown("### ROC Curve")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please train models first from the Dashboard page.")

# ============================================================================
# PREDICTIONS PAGE
# ============================================================================
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
                    unique_vals = st.session_state.df[feature].unique().tolist()
                    input_values[feature] = st.selectbox(
                        feature,
                        options=unique_vals,
                        key=f"pred_{feature}"
                    )
                else:
                    mean_val = float(st.session_state.df[feature].mean()) if feature in st.session_state.df.columns else 0.0
                    input_values[feature] = st.number_input(
                        feature,
                        value=mean_val,
                        key=f"pred_{feature}"
                    )
        
        if st.button("🔮 PREDICT", type="primary"):
            input_array = prepare_prediction_input(
                input_values,
                st.session_state.selected_features,
                st.session_state.feature_encoders
            )
            
            predictions = predict_manual(
                st.session_state.trained_models,
                input_array,
                st.session_state.label_encoder
            )
            
            # Display results
            first_model = list(predictions.keys())[0]
            st.success(f"**Result:** {predictions[first_model]['class']} "
                      f"({predictions[first_model]['confidence']:.1f}% confidence)")
            
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
                results_df = predict_batch(
                    st.session_state.trained_models,
                    batch_df,
                    st.session_state.selected_features,
                    st.session_state.feature_encoders,
                    st.session_state.label_encoder
                )
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download button
                csv, filename = export_predictions(results_df)
                st.download_button(
                    label="📥 DOWNLOAD RESULTS",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error processing batch file: {str(e)}")
    else:
        st.warning("Please train models first from the Dashboard page.")

# ============================================================================
# MODEL HISTORY PAGE
# ============================================================================
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

# ============================================================================
# SETTINGS PAGE
# ============================================================================
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
                import pickle
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
                import pickle
                model = pickle.load(uploaded_model)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
