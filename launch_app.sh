#!/bin/bash

# 🌦️ RainFall AI - Streamlit App Launcher
# This script launches the modern Streamlit application

echo "🌦️ Starting RainFall AI - Rainfall Prediction Platform"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "app/streamlit_app.py" ]; then
    echo "❌ Error: Please run this script from the Rainfall project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if model exists
if [ ! -f "models/ann_rainfall_model.h5" ]; then
    echo "⚠️  Warning: Trained model not found at models/ann_rainfall_model.h5"
    echo "   Please run the training script first to generate the model"
    echo "   You can still use the app for demonstration purposes"
fi

# Check if data exists
if [ ! -f "data/rainfall in india 1901-2015.csv" ]; then
    echo "⚠️  Warning: Rainfall data not found"
    echo "   Some features may not work properly"
fi

echo "🚀 Launching Streamlit application..."
echo "   Local URL: http://localhost:8501"
echo "   Press Ctrl+C to stop the application"
echo ""

# Launch Streamlit app
streamlit run app/streamlit_app.py --server.port 8501 --server.headless false
