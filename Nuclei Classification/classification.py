import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

# Define the ResNet-based model for nuclei classification
class NucleiClassifier(nn.Module):
    def __init__(self, num_classes):
        super(NucleiClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Use pre-trained ResNet-50 model
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)  # Replace the fully connected layer

    def forward(self, x):
        x = self.resnet(x)
        return x

# Define the necessary transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Set the device for training (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and validation datasets
train_dataset = DatasetFolder('NucleiData_Training', loader=torchvision.datasets.folder.default_loader, extensions='.png', transform=transform)
val_dataset = DatasetFolder('NucleiData_Validation', loader=torchvision.datasets.folder.default_loader, extensions='.png', transform=transform)

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Create an instance of the NucleiClassifier
num_classes = 2  # Number of nuclei types to classify
model = NucleiClassifier(num_classes)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {accuracy:.2f}%")
