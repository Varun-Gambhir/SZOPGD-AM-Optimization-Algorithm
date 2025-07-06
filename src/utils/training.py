import torch
import time
import copy
from torch.utils.data import DataLoader

def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=5, device='cpu'):
    """
    Train the model and track performance metrics.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: Optimizer for training
        criterion: Loss function
        epochs: Number of training epochs
        device: Device to run training on ('cuda' or 'cpu')
    
    Returns:
        model: Trained model with best weights
        model_history: Dictionary containing training loss, training accuracy, and test accuracy
    """
    model_history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {i+1}/{len(train_loader)}, '
                      f'Loss: {running_loss/(i+1):.4f}, Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate test accuracy
        test_acc = evaluate(model, test_loader, device)
        
        model_history['train_loss'].append(epoch_loss)
        model_history['train_acc'].append(epoch_acc)
        model_history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds')
        print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%')
        print(f'Test Accuracy: {test_acc:.2f}%')
        print('-' * 50)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, model_history

def evaluate(model, test_loader, device='cpu'):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        float: Test accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total