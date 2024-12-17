import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DeepfakeDetectorCNN
from dataset import DeepfakeDataset, transform
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
model = DeepfakeDetectorCNN().to(device=device)
criterion = nn.CrossEntropyLoss().to(device=device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    
    # Wrap train_loader with tqdm
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Update progress bar with current loss
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save the model
torch.save(model.state_dict(), 'models/deepfake_detector.pth')