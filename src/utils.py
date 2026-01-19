"""
Utility functions and session state management
"""

import streamlit as st


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'dataset': None,
        'df': None,
        'models': {},
        'model_history': [],
        'selected_features': [],
        'target_variable': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'trained_models': {},
        'current_page': "Dashboard",
        'feature_encoders': {},
        'label_encoder': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session():
    """Reset all session state except current_page"""
    current_page = st.session_state.get('current_page', "Dashboard")
    for key in list(st.session_state.keys()):
        if key != 'current_page':
            del st.session_state[key]
    st.session_state.current_page = current_page


def get_dataset_info():
    """Get current dataset information for sidebar"""
    if st.session_state.df is not None and st.session_state.target_variable:
        num_classes = st.session_state.df[st.session_state.target_variable].nunique()
        return {
            'name': st.session_state.dataset,
            'rows': len(st.session_state.df),
            'features': len(st.session_state.df.columns),
            'classes': num_classes
        }
    return None
