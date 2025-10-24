from PIL import Image
import os

root = "data/pokemon"
for subdir, _, files in os.walk(root):
    for file in files:
        path = os.path.join(subdir, file)
        try:
            with Image.open(path) as img:
                img.verify()  # Check integrity
        except Exception:
            print("Removing corrupted:", path)
            os.remove(path)