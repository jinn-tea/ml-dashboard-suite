#!/bin/bash

# ML Dashboard Suite - Run Script
echo "Starting ML Dashboard Suite..."
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting Streamlit application..."
streamlit run app.py
