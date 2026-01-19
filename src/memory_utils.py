"""
Memory optimization utilities
"""

import streamlit as st
import gc
import sys
import os

# Try to import psutil, but make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_memory_usage():
    """Get current memory usage in MB"""
    if not PSUTIL_AVAILABLE:
        return None
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except:
        return None


def cleanup_memory():
    """Force garbage collection"""
    gc.collect()


def check_dataset_size(df, max_rows=100000, max_cols=100):
    """
    Check if dataset is within acceptable size limits
    
    Args:
        df: DataFrame to check
        max_rows: Maximum number of rows allowed
        max_cols: Maximum number of columns allowed
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "Dataset is None"
    
    num_rows = len(df)
    num_cols = len(df.columns)
    
    if num_rows > max_rows:
        return False, f"Dataset too large: {num_rows:,} rows. Maximum allowed: {max_rows:,} rows. Please use a smaller dataset or sample your data."
    
    if num_cols > max_cols:
        return False, f"Dataset too large: {num_cols} columns. Maximum allowed: {max_cols} columns."
    
    # Estimate memory size (rough estimate)
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if memory_mb > 500:  # 500 MB limit
        return False, f"Dataset too large: {memory_mb:.1f} MB. Maximum allowed: 500 MB. Please use a smaller dataset."
    
    return True, None


def optimize_dataframe(df, sample_size=None):
    """
    Optimize dataframe by sampling or reducing memory
    
    Args:
        df: DataFrame to optimize
        sample_size: If provided, sample this many rows
        
    Returns:
        Optimized DataFrame
    """
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        st.warning(f"⚠️ Dataset sampled to {sample_size:,} rows for performance.")
    
    # Convert object columns to category if beneficial
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        if num_unique < len(df) * 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df


def clear_large_objects():
    """Clear large objects from session state"""
    large_keys = ['df', 'X_train', 'X_test', 'trained_models']
    cleared = []
    
    for key in large_keys:
        if key in st.session_state and st.session_state[key] is not None:
            del st.session_state[key]
            st.session_state[key] = None
            cleared.append(key)
    
    cleanup_memory()
    return cleared
