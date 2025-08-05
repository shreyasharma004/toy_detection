# 🎯 Toy Detector using YOLOv8

A custom-trained object detection system that identifies toys in images and real-time video streams using YOLOv8. The project includes a user-friendly Streamlit web interface and real-time camera inference capabilities.

## 🚀 Features

- **Custom YOLOv8 Model**: Trained specifically for toy detection
- **Web Interface**: Interactive Streamlit app for image upload and webcam detection
- **Real-time Detection**: Live camera feed with toy detection overlay
- **Adjustable Confidence**: Configurable detection threshold
- **Multiple Input Methods**: Support for image upload and webcam input
- **Detection Analytics**: Confidence scores and detection counts

## 📁 Project Structure

```
toy-detector/
├── app.py                 # Streamlit web application
├── cam_infer.py           # Real-time camera inference
├── train_yolo.py          # Model training script
├── dataset/
│   ├── data.yaml          # Dataset configuration
│   ├── train/             # Training images and labels
│   ├── test/              # Test images and labels
│   └── runs/detect/       # Training outputs and model weights
└── README.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Webcam (for real-time detection)
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/shreyasharma004/toy_detection/tree/main
   cd toy-detector
   ```

2. **Install dependencies**
   ```bash
   pip install ultralytics streamlit opencv-python pillow numpy
   ```

3. **Prepare your dataset**
   - Organize your images and labels in the `dataset/` folder
   - Ensure `data.yaml` is properly configured with class definitions
   - Follow YOLO format for annotations

## 🏋️ Training the Model

Before running the detection applications, you need to train the custom model:

```bash
python train_yolo.py
```

**Training Features:**
- Uses YOLOv8n as base model
- 50 epochs with batch size 8
- Input image size: 640x640
- Automatic model validation and testing
- Results saved in `dataset/runs/detect/toy_model_v2_trained/`

**Training Output:**
- Best model weights: `dataset/runs/detect/toy_model_v2_trained/weights/best.pt`
- Training metrics and validation results
- Test predictions saved automatically

## 🖥️ Usage

### Web Application (Streamlit)

Launch the interactive web interface:

```bash
streamlit run app.py
```

**Features:**
- **Image Upload**: Upload JPG, JPEG, or PNG images
- **Live Webcam**: Real-time detection through your camera
- **Adjustable Confidence**: Use the sidebar slider (10%-100%)
- **Detection Details**: View confidence scores for each detection

### Real-time Camera Detection

Run standalone camera inference:

```bash
python cam_infer.py
```

**Controls:**
- Press `q` to quit
- Confidence threshold: 50% (hardcoded)
- Automatic camera selection (tries index 0, then 1)

## ⚙️ Configuration

### Model Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | YOLOv8n | Lightweight, fast inference |
| Input Size | 640x640 | Standard YOLO input resolution |
| Classes | toy | Single class detection |
| Default Confidence | 25% | Minimum detection threshold |

### Dataset Format

Ensure your `dataset/data.yaml` follows this structure:

```yaml
train: dataset/train/images
val: dataset/test/images
test: dataset/test/images

nc: 1  # number of classes
names: ['toy']  # class names
```

## 📊 Model Performance

The model is evaluated on:
- **Training Accuracy**: Monitored during 50 epochs
- **Validation mAP**: Mean Average Precision on test set
- **Real-time FPS**: Performance on live video streams

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Found Error**
   ```
   ❌ Trained model not found at: dataset/runs/detect/toy_model_v2_trained/weights/best.pt
   ```
   **Solution**: Run `python train_yolo.py` first to train the model.

2. **Webcam Not Working**
   ```
   ❌ Webcam not found. Please check your camera connection.
   ```
   **Solution**: Ensure your camera is connected and not used by other applications.

3. **Low Detection Accuracy**
   - Lower the confidence threshold in the sidebar
   - Ensure good lighting conditions
   - Check if objects are clearly visible and not too small

### Performance Tips

- **Better Accuracy**: Increase training epochs or add more training data
- **Faster Inference**: Use smaller input size (320x320) for real-time applications
- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base detection framework
- [Streamlit](https://streamlit.io/) for the web interface framework
- [OpenCV](https://opencv.org/) for computer vision utilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an issue on GitHub
3. Review the training logs in `dataset/runs/detect/`

---

**Made with ❤️ and YOLOv8** | *Detecting toys, one frame at a time* 🧸
