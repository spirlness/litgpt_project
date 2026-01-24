import tarfile
import os
from pathlib import Path

tar_path = Path("data/tinystories/TinyStories_all_data.tar.gz")
extract_path = Path("data/tinystories/TinyStories_all_data")

extract_path.mkdir(exist_ok=True, parents=True)

print(f"Extracting {tar_path} to {extract_path}...")

if not tar_path.exists():
    print(f"Error: {tar_path} does not exist!")
    exit(1)

try:
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete.")
except Exception as e:
    print(f"Extraction failed: {e}")
