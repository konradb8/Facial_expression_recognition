import json
import os
import shutil
import kagglehub

dataset_path = kagglehub.dataset_download("astraszab/facial-expression-dataset-image-folders-fer2013")

dest_dir = "."
os.makedirs(dest_dir, exist_ok=True)
for file in os.listdir(dataset_path):
    shutil.move(os.path.join(dataset_path, file), os.path.join(dest_dir, file))
print("Pobrano dane do:", dataset_path)


