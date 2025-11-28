import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score

class ModelTrainer:
    """
    Trainer class for hyperspectral image classification models
    """
    def __init__(self, model, device=None):
        self.model = model
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Data loaders
        self.train_loader = None
        self.test_loader = None
        
    def prepare_data(self, X_train, y_train, X_test, y_test, batch_size=32):
        """
        Prepare data loaders
        
        Args:
            X_train: Training patches (N, H, W, bands)
            y_train: Training labels (N,)
            X_test: Test patches (N, H, W, bands)
            y_test: Test labels (N,)
            batch_size: Batch size for training
        """
        print(f"Preparing data...")
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
        
        # Convert to tensors - IMPORTANT: Keep as (N, H, W, bands)
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        print(f"Tensor shapes: X_train={X_train_tensor.shape}, X_test={X_test_tensor.shape}")
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"Train batches: {len(self.train_loader)}, Test batches: {len(self.test_loader)}")
        
    def train(self, epochs=50, verbose=True):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            verbose: Print training progress
            
        Returns:
            history: Dictionary with training history
            best_acc: Best test accuracy achieved
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # Calculate training metrics
            train_loss = train_loss / len(self.train_loader)
            train_acc = train_correct / train_total
            
            # Evaluation phase
            test_loss, test_acc = self.evaluate()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Update best accuracy
            if test_acc > best_acc:
                best_acc = test_acc
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        
        print(f"\nTraining completed! Best Test Accuracy: {best_acc*100:.2f}%")
        
        return history, best_acc
    
    def evaluate(self):
        """
        Evaluate the model on test set
        
        Returns:
            test_loss: Average test loss
            test_acc: Test accuracy
        """
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(self.test_loader)
        test_acc = test_correct / test_total
        
        return test_loss, test_acc
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test patches (N, H, W, bands)
            
        Returns:
            predictions: Predicted labels (N,)
        """
        self.model.eval()
        
        # Convert to tensor
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Create dataloader
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for (inputs,) in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")