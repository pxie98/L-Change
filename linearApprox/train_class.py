import torch
import torch.nn as nn
import torch.optim as optim



class train_model():
    """
    Neural network training handler with monitoring and checkpointing capabilities.
    
    This class encapsulates the complete training workflow including forward/backward
    propagation, loss tracking, and model checkpointing at specified intervals.
    
    Attributes:
        num_epochs (int): Total number of training iterations to execute
    """
    
    def __init__(self, num_epochs):
        """
        Initialize training configuration.
        
        Args:
            num_epochs (int): Maximum number of training epochs
        """
        self.num_epochs = num_epochs

    def train(self, model, optimizer, criterion, train_data, train_labels, 
              test_data, test_labels, layer_num, hidden_size, path):
        """
        Execute complete training cycle with monitoring and checkpointing.
        
        Training Process:
        - Performs standard forward/backward propagation with gradient optimization
        - Tracks test loss at regular intervals for model evaluation
        - Implements automatic checkpointing to preserve training progress
        - Maintains loss trajectory for convergence analysis
        
        Args:
            model (nn.Module): Neural network instance to be trained
            optimizer (torch.optim): Optimization algorithm (e.g., Adam, SGD)
            criterion (nn.Module): Loss function for optimization
            train_data (Tensor): Training input features
            train_labels (Tensor): Ground truth labels for training set
            test_data (Tensor): Validation input features for evaluation
            test_labels (Tensor): Ground truth labels for validation set
            layer_num (int): Number of hidden layers in current architecture
            hidden_size (int): Number of neurons per hidden layer
            path (str): Directory path for model checkpoint storage
            
        Returns:
            list: Test loss values recorded at evaluation intervals
        """
        # Initialize loss container for convergence tracking
        losses = []
        
        # Main training loop over specified number of epochs
        for epoch in range(self.num_epochs):
            model.train()
            # Reset gradient buffers to prevent accumulation
            optimizer.zero_grad()
            
            # Forward pass: compute model predictions
            outputs = model(train_data)
            
            # Calculate training loss between predictions and ground truth
            loss = criterion(outputs, train_labels)

            # Backward propagation: compute gradients via automatic differentiation
            loss.backward()
            
            # Parameter update: adjust weights using computed gradients
            optimizer.step()

            # Periodic evaluation and checkpointing (every 500 epochs)
            if epoch % 500 == 0:
                # Switch to evaluation mode for inference
                model.eval()
                with torch.no_grad():  # Disable gradient computation for efficiency
                    outputs = model(test_data)
                    test_loss = criterion(outputs, test_labels)
                
                # Record test loss for performance monitoring
                losses.append(test_loss.item())
                
                # Training progress logging
                print(f"Epoch: {epoch}, Test Loss: {test_loss.item():.6f}")
                
                # Generate model checkpoint filename with architecture details
                model_path = f'{path}/model_layer{layer_num}_hidden{hidden_size}_epoch{epoch}.pt'
                
                # Save model state dictionary for later resumption or deployment
                torch.save(model.state_dict(), model_path)

        return losses
 