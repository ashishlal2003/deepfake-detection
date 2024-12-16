import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DeepfakeDetectorCNN
from dataset import DeepfakeDataset, transform

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Load datasets
train_dataset = DeepfakeDataset('data/split/train', transform=transform)
test_dataset = DeepfakeDataset('data/split/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = DeepfakeDetectorCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), 'models/deepfake_detector.pth')