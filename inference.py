import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

device = torch.device("cpu")

emotion_labels = {
    0: "angry",
    1: "fear",
    2: "happy",
    3: "sad",
    4: "surprise",
    5: "neutral"
}

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = EmotionCNN()
model.load_state_dict(torch.load("emotion_cnn.pth", map_location=device))
model.to(device)
model.eval()

def visualize_predictions(model, dataloader, label_map, num_images=25):
    model.eval()
    
    images_list = []
    labels_list = []

    class_images = {class_id: [] for class_id in range(len(label_map))}
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        for i in range(len(labels)):
            class_images[labels[i].item()].append(images[i].cpu())
        
        if all(len(class_images[class_id]) >= num_images // len(label_map) for class_id in range(len(label_map))):
            break

    selected_images = []
    selected_labels = []

    for class_id, images in class_images.items():
        if len(images) >= num_images // len(label_map):
            selected_images.extend(random.sample(images, num_images // len(label_map)))
            selected_labels.extend([class_id] * (num_images // len(label_map)))

    total_images = len(selected_images)
    if total_images < num_images:
        num_images = total_images 

    indices = torch.randperm(len(selected_images)).tolist()
    selected_images = [selected_images[i] for i in indices]
    selected_labels = [selected_labels[i] for i in indices]

    rows = (num_images + 4) // 5 
    cols = min(num_images, 5) 

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.ravel()

    with torch.no_grad():
        inputs = torch.stack(selected_images).to(device)
        labels_tensor = torch.tensor(selected_labels).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(selected_images[i].squeeze(), cmap='gray')
        ax.set_title(f"True: {label_map[labels_tensor[i].item()]}\nPred: {label_map[predicted[i].item()]}")
        ax.axis('off')

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_dataset = ImageFolder('data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    visualize_predictions(model, test_loader, emotion_labels)
