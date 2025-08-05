import os
from ultralytics import YOLO

print("=== YOLOv8 Toy Detection Training Script ===")
print(f"Current working directory: {os.getcwd()}")

# Step 1: Use consistent relative paths
data_yaml_path = "dataset/data.yaml"
pretrained_weights = "yolov8n.pt"
test_images_path = "dataset/test/images"

# Step 2: Verify all paths exist before training
print("\n=== Path Verification ===")
if not os.path.exists(data_yaml_path):
    print(f"❌ Data YAML not found at: {data_yaml_path}")
    exit(1)
else:
    print(f"✅ Data YAML found: {data_yaml_path}")

if not os.path.exists(pretrained_weights):
    print(f"❌ YOLOv8 weights not found at: {pretrained_weights}")
    exit(1)
else:
    print(f"✅ YOLOv8 weights found: {pretrained_weights}")

if not os.path.exists(test_images_path):
    print(f"❌ Test images not found at: {test_images_path}")
    exit(1)
else:
    test_count = len([f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"✅ Test images found: {test_count} images")

# Step 3: Load base YOLOv8 model
print(f"\n=== Loading Model ===")
model = YOLO(pretrained_weights)
print(f"✅ YOLOv8n model loaded successfully")

# Step 4: Train the model with CORRECTED paths
print(f"\n=== Starting Training ===")
print("Training parameters:")
print(f"  - Data: {data_yaml_path}")
print(f"  - Epochs: 50")
print(f"  - Image size: 640")
print(f"  - Batch size: 8")

try:
    results = model.train(
            data = data_yaml_path,
            epochs = 50,
            imgsz = 640,
            batch = 8,
            project = "dataset/runs/detect",  # This creates the folder structure
            name = "toy_model_v2_trained",  # Clear identifier for new training
            exist_ok = True,
            save = True,  # Ensure model saving is enabled
            verbose = True  # Show detailed output
    )
    print("✅ Training completed successfully!")

except Exception as e:
    print(f"❌ Training failed with error: {e}")
    exit(1)

# Step 5: Verify model was saved correctly
expected_model_path = "dataset/runs/detect/toy_model_v2_trained/weights/best.pt"
print(f"\n=== Verifying Model Save ===")

if os.path.exists(expected_model_path):
    model_size = os.path.getsize(expected_model_path) / (1024 * 1024)  # Size in MB
    print(f"✅ Model saved successfully!")
    print(f"   Location: {expected_model_path}")
    print(f"   Size: {model_size:.2f} MB")

    # Step 6: Load and test the trained model
    print(f"\n=== Testing Trained Model ===")
    try:
        trained_model = YOLO(expected_model_path)
        print("✅ Trained model loaded successfully!")

        # Test on a few images
        if os.path.exists(test_images_path):
            print(f"Running test predictions...")
            test_results = trained_model.predict(
                    source = test_images_path,
                    save = True,
                    conf = 0.25,
                    project = "dataset/runs/detect",
                    name = "toy_model_v2_predictions",
                    exist_ok = True
            )

            detection_count = 0
            for result in test_results:
                if result.boxes is not None:
                    detection_count += len(result.boxes)

            print(f"✅ Test completed!")
            print(f"   Total detections: {detection_count}")
            print(f"   Results saved in: dataset/runs/detect/toy_model_v2_predictions/")

    except Exception as e:
        print(f"❌ Error testing model: {e}")

else:
    print(f"❌ Model not found at expected location: {expected_model_path}")

    # Check what was actually created
    runs_path = "dataset/runs"
    if os.path.exists(runs_path):
        print(f"\nChecking what was created in {runs_path}:")
        for root, dirs, files in os.walk(runs_path):
            level = root.replace(runs_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

print(f"\n=== Training Summary ===")
print(f"✅ Script completed!")
print(f"Expected model location: {expected_model_path}")
print(f"Use this path in your app.py and cam_infer.py files.")