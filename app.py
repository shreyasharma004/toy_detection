import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

# Load your trained model with error handling
model_path = "dataset/runs/detect/toy_model_v2_trained/weights/best.pt"

if not os.path.exists(model_path):
    st.error(f"❌ Trained model not found at: {model_path}")
    st.error("Please run train_yolo.py first to train the model.")
    st.stop()

try:
    model = YOLO(model_path)
    st.success(f"✅ Model loaded successfully from: {model_path}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Set Streamlit page config
st.set_page_config(page_title = "Toy Detector 🎯", layout = "centered")

st.title("🔍 Toy Detector using YOLOv8")
st.markdown("Upload an image or use your webcam to detect **toys** using your custom-trained model.")

# Model info sidebar
with st.sidebar:
    st.header("📊 Model Info")
    st.write(f"**Model:** YOLOv8 Custom Trained")
    st.write(f"**Classes:** toy")
    st.write(f"**Confidence Threshold:** 25%")

    # Confidence threshold slider
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Main options
option = st.radio("Choose Input Type", ["Upload Image", "Use Webcam"])


# Function to perform detection
def detect_and_display(image, conf_thresh = 0.25):
    with st.spinner("🔍 Detecting toys..."):
        results = model.predict(image, conf = conf_thresh)
        result_image = results[0].plot()

        # Count detections
        detection_count = len(results[0].boxes) if results[0].boxes is not None else 0

        if detection_count == 0:
            st.warning("⚠️ No toys detected!")
            st.image(image, caption = "Original Image", use_container_width = True)
        else:
            st.success(f"🎯 Found {detection_count} toy(s)!")
            st.image(result_image, caption = f"Detected: {detection_count} toys", use_container_width = True)

            # Show confidence scores
            if results[0].boxes is not None:
                st.subheader("🔍 Detection Details:")
                for i, box in enumerate(results[0].boxes):
                    conf = box.conf[0].item()
                    st.write(f"Toy {i + 1}: **{conf:.2f}** confidence")


# ---- Upload Image ----
if option == "Upload Image":
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type = ["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption = "Uploaded Image", use_container_width = True)

        with col2:
            detect_and_display(np.array(image), conf_threshold)

# ---- Webcam ----
elif option == "Use Webcam":
    st.header("📷 Live Camera Detection")

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Webcam not found. Please check your camera connection.")
        else:
            st.info("🟢 Camera is running. Uncheck 'Start Webcam' to stop.")

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run detection
                results = model.predict(frame_rgb, conf = conf_threshold, verbose = False)
                result_frame = results[0].plot()

                detection_count = len(results[0].boxes) if results[0].boxes is not None else 0

                if detection_count == 0:
                    FRAME_WINDOW.image(frame_rgb, caption = "🔍 Scanning for toys...", use_container_width = True)
                else:
                    FRAME_WINDOW.image(result_frame, caption = f"🎯 Detected: {detection_count} toys", use_container_width = True)

                # Check if checkbox is still checked
                run = st.session_state.get('run', False)

            cap.release()

# Footer
st.markdown("---")
st.markdown("**🤖 Powered by YOLOv8 | Custom Toy Detection Model**")