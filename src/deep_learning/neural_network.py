#!/usr/bin/env python3
"""
Deep Learning Neural Network Module

This module provides basic neural network functionality using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import os


class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for demonstration.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layer
            num_classes (int): Number of output classes
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CNNModel(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    """
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize the CNN model.
        
        Args:
            num_classes (int): Number of output classes
        """
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass of the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ModelTrainer:
    """
    A class for training PyTorch models.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): PyTorch model to train
            device (str): Device to use for training ('cpu' or 'cuda')
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        
    def train_step(self, data_loader, optimizer, criterion):
        """
        Perform one training epoch.
        
        Args:
            data_loader: DataLoader with training data
            optimizer: Optimizer for training
            criterion: Loss function
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, data_loader, criterion):
        """
        Evaluate the model on validation/test data.
        
        Args:
            data_loader: DataLoader with evaluation data
            criterion: Loss function
            
        Returns:
            Tuple[float, float]: Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy


def create_sample_data(num_samples: int = 1000, input_size: int = 20, 
                      num_classes: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample data for demonstration.
    
    Args:
        num_samples (int): Number of samples to generate
        input_size (int): Size of input features
        num_classes (int): Number of classes
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Features and labels
    """
    # Generate random features
    X = torch.randn(num_samples, input_size)
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y


def demo_neural_network():
    """
    Demonstrate basic neural network functionality.
    """
    print("=== Deep Learning Neural Network Demo ===")
    
    # Check PyTorch installation and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    # Set parameters
    input_size = 20
    hidden_size = 50
    num_classes = 3
    num_samples = 1000
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 5
    
    print(f"\\nModel Parameters:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Number of samples: {num_samples}")
    
    # Create sample data
    print("\\nGenerating sample data...")
    X, y = create_sample_data(num_samples, input_size, num_classes)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    print("\\nCreating neural network model...")
    model = SimpleNN(input_size, hidden_size, num_classes)
    print(f"Model architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ModelTrainer(model, device)
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop demonstration
    print(f"\\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        avg_loss = trainer.train_step(data_loader, optimizer, criterion)
        val_loss, accuracy = trainer.evaluate(data_loader, criterion)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]: "
              f"Train Loss: {avg_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Accuracy: {accuracy:.2f}%")
    
    print("\\nTraining completed successfully!")
    
    # Test prediction
    print("\\nTesting prediction on sample data...")
    model.eval()
    with torch.no_grad():
        sample_input = X[:5].to(trainer.device)
        predictions = model(sample_input)
        predicted_classes = torch.argmax(predictions, dim=1)
        actual_classes = y[:5]
        
        print("Sample predictions:")
        for i in range(5):
            print(f"  Sample {i+1}: Predicted={predicted_classes[i].item()}, "
                  f"Actual={actual_classes[i].item()}")
    
    return True


def demo_cnn():
    """
    Demonstrate CNN functionality.
    """
    print("\\n=== CNN Model Demo ===")
    
    # Create a simple CNN
    num_classes = 10
    cnn_model = CNNModel(num_classes)
    
    print(f"CNN Model architecture:")
    print(cnn_model)
    
    # Count parameters
    total_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"\\nTotal CNN parameters: {total_params:,}")
    
    # Test with sample image data
    batch_size = 4
    sample_images = torch.randn(batch_size, 3, 32, 32)  # Random RGB images
    
    print(f"\\nTesting CNN with sample images of shape: {sample_images.shape}")
    
    cnn_model.eval()
    with torch.no_grad():
        output = cnn_model(sample_images)
        print(f"CNN output shape: {output.shape}")
        print(f"Sample output (first image): {output[0]}")
    
    print("CNN demo completed successfully!")
    return True


if __name__ == "__main__":
    # Run demonstrations
    demo_neural_network()
    demo_cnn()