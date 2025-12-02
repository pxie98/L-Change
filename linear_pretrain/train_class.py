import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    def __init__(self, num_epochs):
        # Total number of training epochs
        self.num_epochs = num_epochs


    def train(self, model, optimizer, criterion, train_data, train_labels, test_data, test_labels, layer_num, hidden_size, path, round):
        """
        Execute the training loop with periodic checkpointing and monitoring
        
        Args:
            model: Neural network model to train
            optimizer: Optimization algorithm instance
            criterion: Loss function criterion
            train_data: Training input features
            train_labels: Training target values
            test_data: Test input features for evaluation
            test_labels: Test target values for evaluation
            layer_num: Number of hidden layers in model architecture
            hidden_size: Number of neurons per hidden layer
            path: Directory path for model checkpoint storage
            round: Training round identifier for file naming
            
        Returns:
            losses: List of loss values recorded during training
            model: Trained neural network model
        """
        # Training model
        losses = []  # Track loss values over training epochs
        
        for epoch in range(self.num_epochs):
            # Forward propagation
            optimizer.zero_grad()  # Clear gradients from previous iteration
            outputs = model(train_data)  # Generate model predictions
            loss = criterion(outputs, train_labels)  # Compute loss between predictions and targets

            # Backward propagation and optimization
            loss.backward()  # Compute gradients via backpropagation
            optimizer.step()  # Update model parameters

            # Periodic checkpointing and monitoring
            if epoch % 100 == 0:
                losses.append(loss.item())  # Record current loss value

                # Save model checkpoint with architecture and training metadata
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}.pt'.format(path, layer_num, hidden_size, epoch, round)
                torch.save(model.state_dict(), model_path)
                
            # Progress reporting at larger intervals
            if epoch % 10000 == 0:
                print("epoch:{}, loss is {}".format(epoch, loss.item()))
                
        return losses, model  # Return loss trajectory and trained model