#!/usr/bin/env python3
"""
Debug script to test Streamlit app components
"""

print("🔍 Debugging Streamlit App Issues")
print("=" * 40)

# Test 1: Check imports
print("\n1️⃣ Testing imports...")
try:
    import streamlit as st
    print("✅ streamlit imported successfully")
except ImportError as e:
    print(f"❌ streamlit import failed: {e}")
    print("💡 Run: pip install streamlit")

try:
    import cv2
    print("✅ cv2 imported successfully")
except ImportError as e:
    print(f"❌ cv2 import failed: {e}")
    print("💡 Run: pip install opencv-python")

try:
    from PIL import Image
    print("✅ PIL imported successfully")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")
    print("💡 Run: pip install Pillow")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")
    print("💡 Run: pip install numpy")

try:
    from ultralytics import YOLO
    print("✅ ultralytics imported successfully")
except ImportError as e:
    print(f"❌ ultralytics import failed: {e}")
    print("💡 Run: pip install ultralytics")

# Test 2: Check model path
print("\n2️⃣ Testing model path...")
import os
model_path = "dataset/runs/detect/toy_model_v2_trained/weights/best.pt"
if os.path.exists(model_path):
    print(f"✅ Model found at: {model_path}")
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   File size: {file_size:.2f} MB")
else:
    print(f"❌ Model NOT found at: {model_path}")
    print("💡 Check if training completed successfully")

# Test 3: Try loading model
print("\n3️⃣ Testing model loading...")
if os.path.exists(model_path):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
else:
    print("⏭️ Skipping model test (file not found)")

# Test 4: Check current directory
print("\n4️⃣ Current directory check...")
print(f"Current directory: {os.getcwd()}")
print("Files in current directory:")
for file in sorted(os.listdir('.')):
    if os.path.isdir(file):
        print(f"  📁 {file}/")
    else:
        print(f"  📄 {file}")

print("\n5️⃣ Streamlit version check...")
try:
    import streamlit as st
    print(f"Streamlit version: {st.__version__}")
except:
    print("Cannot determine Streamlit version")

print("\n✅ Debug complete!")
print("\n💡 Next steps:")
print("1. Fix any import errors shown above")
print("2. Ensure model path is correct")
print("3. Try running: streamlit run app.py --server.headless true")