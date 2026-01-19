# Memory Optimization Guide

This document explains the memory optimizations implemented to reduce RAM usage.

## 🔍 Issues Identified

The application was using excessive RAM due to:

1. **Large datasets stored in session state** - Entire DataFrames kept in memory
2. **Multiple trained models** - All models (k-NN, RF, NB) stored with full predictions
3. **Matplotlib figures not closed** - Figures accumulating in memory
4. **Large meshgrids for decision boundaries** - Creating huge arrays for visualization
5. **No dataset size limits** - Could load very large files
6. **Predictions stored separately** - Extra memory usage

## ✅ Optimizations Implemented

### 1. Dataset Size Limits (`src/data_processing.py`)
- **File size check**: Maximum 500 MB per file
- **Row limit**: Maximum 100,000 rows (auto-samples to 50,000 if larger)
- **Column limit**: Maximum 100 columns
- **Auto-sampling**: Very large datasets automatically sampled

```python
df, error = load_dataset(uploaded_file, max_size_mb=500)
is_valid, size_error = check_dataset_size(df, max_rows=100000, max_cols=100)
```

### 2. DataFrame Optimization (`src/memory_utils.py`)
- **Category conversion**: Object columns with low cardinality converted to categories
- **Memory-efficient types**: Automatically optimized data types
- **Garbage collection**: Force cleanup after loading

```python
df = optimize_dataframe(df)
cleanup_memory()
```

### 3. Model Predictions Storage (`src/model_training.py`)
- **Optional prediction storage**: Can train models without storing all predictions
- **On-demand prediction**: Recompute predictions only when needed for visualizations
- **Memory cleanup**: Remove large arrays after use

### 4. Visualization Optimizations (`src/visualizations.py`)

#### Decision Boundary:
- **Adaptive mesh resolution**: Adjusts based on data range
- **Limited mesh points**: Maximum 10,000 mesh points (was unlimited)
- **Sampled test data**: Limits test points to 5,000 for display
- **Adaptive step size**: Reduces unnecessary grid points

```python
fig = plot_decision_boundary(..., max_points=5000)
```

#### Matplotlib Figures:
- **Explicit cleanup**: All figures closed after display
- **Tight layout**: Prevents memory leaks from layout calculations

```python
st.pyplot(fig)
plt.close(fig)  # Free memory immediately
```

### 5. Memory Management Tools (`src/memory_utils.py`)
- **Clear large objects**: Button to remove datasets, models, etc.
- **Garbage collection**: Manual cleanup option
- **Memory monitoring**: Display current RAM usage

Available in sidebar: **Memory Management** section

### 6. Session State Cleanup (`app.py`)
- **Automatic cleanup**: Garbage collection after data loading
- **Explicit figure closing**: All matplotlib figures closed
- **Prediction recomputation**: Only store when needed

## 📊 Expected Memory Reduction

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Large Dataset (100K rows) | ~200 MB | ~100 MB | 50% |
| Decision Boundary | ~50 MB | ~5 MB | 90% |
| Model Predictions | ~20 MB | ~0-5 MB | 75-100% |
| Matplotlib Figures | Accumulates | Cleared | Variable |

**Total expected reduction: 40-60%**

## 🎯 Usage Recommendations

### For Large Datasets:
1. **Sample your data** before uploading (recommended: < 50,000 rows)
2. **Use Memory Management tools** in sidebar to clear unused objects
3. **Train models selectively** - Don't train all models if not needed

### Memory Management Sidebar:
- **🗑️ Clear Large Objects**: Removes datasets and models from memory
- **🧹 Force Garbage Collection**: Manually trigger cleanup
- **Memory Usage Display**: Monitor current RAM usage

### Best Practices:
1. **Use "NEW SESSION"** button to reset everything when done
2. **Clear large objects** after training if not needed
3. **Close browser tabs** when not using the app
4. **Limit concurrent operations** - Don't train multiple times without clearing

## 🔧 Configuration

You can adjust limits in `src/data_processing.py`:

```python
# Dataset size limits
df, error = load_dataset(uploaded_file, max_size_mb=500)  # Adjust MB limit
is_valid, _ = check_dataset_size(df, max_rows=100000, max_cols=100)  # Adjust limits

# Decision boundary points
fig = plot_decision_boundary(..., max_points=5000)  # Adjust point limit
```

## 🚨 Troubleshooting

### If still using too much RAM:

1. **Reduce dataset size** - Use smaller sample
2. **Clear session** - Click "NEW SESSION"
3. **Train fewer models** - Selectively train only needed algorithms
4. **Restart Streamlit** - If memory leaks persist

### Monitor Memory:
Check sidebar → Memory Management → Memory Usage Display

## 📝 Future Improvements

Potential further optimizations:
- **Model persistence to disk** instead of memory
- **Lazy loading** for large datasets
- **Streaming data processing** for very large files
- **Compression** for stored data structures
