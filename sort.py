import json
import os
import shutil
import kagglehub

dataset_path = kagglehub.dataset_download("wjybuqi/traffic-light-detection-dataset")

dest_dir = "/"
os.makedirs(dest_dir, exist_ok=True)
for file in os.listdir(dataset_path):
    shutil.move(os.path.join(dataset_path, file), os.path.join(dest_dir, file))
print("Pobrano dane do:", dataset_path)


json_path = "train_dataset/train.json"
images_root = "train_dataset/train_images"
output_root = "train_dataset/sorted_images"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for annotation in data["annotations"]:
    filename = annotation["filename"]
    file_path = os.path.join(images_root, os.path.basename(filename))

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        continue

    colors = set(inbox["color"] for inbox in annotation.get("inbox", []))

    if not colors:
        continue

    for color in colors:
        color_dir = os.path.join(output_root, color)
        os.makedirs(color_dir, exist_ok=True)

        target_path = os.path.join(color_dir, os.path.basename(filename))
        shutil.copy2(file_path, target_path)
        print(f"Copied {file_path} -> {target_path}")

print("Images sorted!")