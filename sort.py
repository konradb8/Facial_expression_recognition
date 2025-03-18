import json
import os
import shutil

json_path = "data/train_dataset/train.json"
images_root = "data/train_dataset/train_images"
output_root = "data/train_dataset/sorted_images"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for annotation in data["annotations"]:
    filename = annotation["filename"]
    file_path = os.path.join(images_root, os.path.basename(filename))

    if not os.path.exists(file_path):
        print(f"Plik {file_path} nie istnieje, pomijam...")
        continue

    colors = set(inbox["color"] for inbox in annotation.get("inbox", []))

    if not colors:
        continue

    for color in colors:
        color_dir = os.path.join(output_root, color)
        os.makedirs(color_dir, exist_ok=True)

        target_path = os.path.join(color_dir, os.path.basename(filename))
        shutil.copy2(file_path, target_path)
        print(f"Skopiowano {file_path} -> {target_path}")

print("Sortowanie zakończone!")