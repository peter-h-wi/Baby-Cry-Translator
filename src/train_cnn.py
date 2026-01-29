import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .dataset import BabyCryDataset
from .model import BabyCryCNN

def get_file_paths(data_path=config.DATA_RAW_DIR):
    """
    Scans directories and returns a list of (path, label).
    """
    paths = []
    labels = []
    
    print(f"Scanning {data_path}...")
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                
                label = None
                for label_name, label_idx in config.LABELS.items():
                    if label_name in root.lower() or label_name in file.lower():
                        label = label_idx
                        break
                
                if label is not None:
                    paths.append(file_path)
                    labels.append(label)
                    
    return paths, labels

def train_cnn():
    # 1. Prepare Data
    paths, labels = get_file_paths()
    if not paths:
        print("No data found.")
        return

    # Split: Train/Test
    # We split paths first so we can create Datasets separately
    X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Create Datasets
    # Train set gets augmentation ON
    train_dataset = BabyCryDataset(X_train, y_train, augment=True) 
    # Test set gets augmentation OFF
    test_dataset = BabyCryDataset(X_test, y_test, augment=False)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 2. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BabyCryCNN(num_classes=5).to(device)
    
    # Loss & Optimizer
    # Calculate class weights
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    class_counts = torch.bincount(y_train_tensor)
    # Avoid div by zero if any class is missing in split (unlikely with stratify but safe)
    # Weight = Total / (NumClasses * Count) is a common formula, or just 1/Count
    # Let's use 1/Count normalized
    weights = 1.0 / (class_counts.float() + 1e-6)
    weights = weights / weights.sum()
    weights = weights.to(device)
    
    print(f"Class Weights: {weights}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 20
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for specs, targets in train_loader:
            specs, targets = specs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for specs, targets in test_loader:
                specs, targets = specs.to(device), targets.to(device)
                outputs = model(specs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(config.BASE_DIR, 'cnn_model.pth'))
            
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    print("\nFinal Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=list(config.LABELS.keys())))

if __name__ == "__main__":
    train_cnn()
