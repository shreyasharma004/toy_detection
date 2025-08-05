import os

print("=== Directory Structure Diagnostic ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Python script location: {__file__}")

print("\n=== Files in current directory ===")
try:
    files = os.listdir('.')
    for file in sorted(files):
        print(f"  {file}")
except Exception as e:
    print(f"Error listing current directory: {e}")

print("\n=== Checking for dataset folder ===")
if os.path.exists("dataset"):
    print("✅ dataset folder found")
    print("\n=== Files in dataset folder ===")
    try:
        dataset_files = os.listdir("dataset")
        for file in sorted(dataset_files):
            path = os.path.join("dataset", file)
            if os.path.isdir(path):
                print(f"  📁 {file}/")
            else:
                print(f"  📄 {file}")
    except Exception as e:
        print(f"Error listing dataset directory: {e}")

    print("\n=== Checking for data.yaml ===")
    if os.path.exists("dataset/data.yaml"):
        print("✅ data.yaml found at dataset/data.yaml")
    else:
        print("❌ data.yaml NOT found at dataset/data.yaml")

    print("\n=== Checking for required folders ===")
    required_folders = ["train", "valid", "test"]
    for folder in required_folders:
        folder_path = f"dataset/{folder}"
        if os.path.exists(folder_path):
            print(f"✅ {folder_path} exists")
            images_path = f"{folder_path}/images"
            if os.path.exists(images_path):
                image_count = len([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"   📸 {image_count} images found")
            else:
                print(f"   ⚠️ {images_path} not found")
        else:
            print(f"❌ {folder_path} does not exist")
else:
    print("❌ dataset folder not found in current directory")

print("\n=== Recommended Actions ===")
if not os.path.exists("dataset"):
    print("1. Make sure you're running this from the ToyDetector directory")
    print("2. Run dataset.py first to download the dataset")
elif not os.path.exists("dataset/data.yaml"):
    print("1. The data.yaml file is missing from the dataset folder")
    print("2. Try re-downloading the dataset using dataset.py")
else:
    print("Everything looks good! You can proceed with training.")