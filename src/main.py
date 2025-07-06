import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from src.optimizers.szopgdam import SZOPGDAM
from src.models.resnet18 import ResNet18
from src.utils.training import train_model, evaluate

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Fashion MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Initialize models
model_szopgdam = ResNet18(num_classes=10).to(device)
model_adam = ResNet18(num_classes=10).to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizers
optimizer_szopgdam = SZOPGDAM(model_szopgdam.parameters(), eta0=0.1, beta=0.9, mu=0.01, weight_decay=5e-4)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)

# Number of epochs
epochs = 20

# Train with SZOPGDAM
print("Training with SZOPGD-AM optimizer...")
model_szopgdam, history_szopgdam = train_model(
    model_szopgdam, train_loader, test_loader, optimizer_szopgdam, criterion, epochs=epochs, device=device
)

# Train with Adam
print("Training with Adam optimizer...")
model_adam, history_adam = train_model(
    model_adam, train_loader, test_loader, optimizer_adam, criterion, epochs=epochs, device=device
)

# Plot results
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 3, 1)
plt.plot(history_szopgdam['train_loss'], label='SZOPGD-AM')
plt.plot(history_adam['train_loss'], label='Adam')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy
plt.subplot(1, 3, 2)
plt.plot(history_szopgdam['train_acc'], label='SZOPGD-AM')
plt.plot(history_adam['train_acc'], label='Adam')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Plot test accuracy
plt.subplot(1, 3, 3)
plt.plot(history_szopgdam['test_acc'], label='SZOPGD-AM')
plt.plot(history_adam['test_acc'], label='Adam')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('optimizer_comparison.png')
plt.show()

# Print final test accuracies
final_acc_szopgdam = evaluate(model_szopgdam, test_loader, device)
final_acc_adam = evaluate(model_adam, test_loader, device)

print(f"Final Test Accuracy - SZOPGD-AM: {final_acc_szopgdam:.2f}%")
print(f"Final Test Accuracy - Adam: {final_acc_adam:.2f}%")

# Save models
torch.save(model_szopgdam.state_dict(), 'saved_models/resnet_fashion_mnist_szopgdam.pth')
torch.save(model_adam.state_dict(), 'saved_models/resnet_fashion_mnist_adam.pth')
print("Models saved.")