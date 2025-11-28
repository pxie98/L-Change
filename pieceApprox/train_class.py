import torch
import torch.nn as nn
import torch.optim as optim

class train_model():
    def __init__(self, num_epochs):
        # Number of training epochs
        self.num_epochs = num_epochs


    def train(self, model, optimizer, criterion, train_data, train_labels, test_data, test_labels, layer_num, hidden_size, path, round):
        """
        Execute the training loop for neural network model
        
        Args:
            model: Neural network model instance
            optimizer: Optimization algorithm (e.g., Adam, SGD)
            criterion: Loss function criterion
            train_data: Training input features
            train_labels: Training target labels
            test_data: Test input features (currently unused in loop)
            test_labels: Test target labels (currently unused in loop)
            layer_num: Number of hidden layers in model
            hidden_size: Number of neurons per hidden layer
            path: Directory path for model checkpoint saving
            round: Training round identifier for file naming
            
        Returns:
            losses: List of loss values recorded during training
        """
        # Training model
        losses = []  # Track loss values over epochs
        
        for epoch in range(self.num_epochs):
            # Forward propagation
            optimizer.zero_grad()  # Clear gradients from previous iteration
            outputs = model(train_data)  # Generate model predictions
            loss = criterion(outputs, train_labels)  # Compute loss between predictions and ground truth

            # Backward propagation and optimization
            loss.backward()  # Compute gradients via backpropagation
            optimizer.step()  # Update model parameters

            # Checkpointing and monitoring at regular intervals
            if epoch % 5 == 0:
                losses.append(loss.item())  # Record current loss value
                #print("epoch:{}, loss is {}".format(epoch,loss.item()))
                
                # Save model checkpoint with architecture and training metadata
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}.pt'.format(path, layer_num, hidden_size, epoch, round)
                torch.save(model.state_dict(), model_path)  # Persist model weights to disk


        return losses 