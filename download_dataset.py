import kagglehub
import shutil
import os

# Set Kaggle config directory to current folder so it finds kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(os.path.abspath(__file__))

# Download latest version
print("Downloading dataset...")
cache_path = kagglehub.dataset_download("kiranmahesh/nslkdd")

print(f"Downloaded to cache: {cache_path}")

# Destination directory
dest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# Create destination directory if it doesn't exist
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
    print(f"Created directory: {dest_path}")

# Copy files from cache to local dataset directory
print(f"Moving files to {dest_path}...")
for item in os.listdir(cache_path):
    s = os.path.join(cache_path, item)
    d = os.path.join(dest_path, item)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Dataset is now available at: {dest_path}")