import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    def __init__(self, num_epochs):
        # Total number of training epochs
        self.num_epochs = num_epochs


    def train(self, model, optimizer, criterion, train_data, train_labels, test_data, test_labels, layer_num, hidden_size, path):
        """
        Execute the training loop with adaptive checkpointing strategy
        
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
        """
        # Training model
        losses = []  # Track test loss values over training
        
        for epoch in range(self.num_epochs):
            # Forward propagation
            optimizer.zero_grad()  # Clear gradients from previous iteration
            outputs = model(train_data)  # Generate model predictions
            loss = criterion(outputs, train_labels)  # Compute training loss

            # Backward propagation and optimization
            loss.backward()  # Compute gradients via backpropagation
            optimizer.step()  # Update model parameters

            # Adaptive checkpointing strategy based on training phase
            if epoch < 500:
                # Early training phase: frequent evaluation and checkpointing
                if epoch % 20 == 0:
                    outputs = model(test_data)  # Generate predictions on test set
                    test_loss = criterion(outputs, test_labels)  # Compute test loss
                    losses.append(test_loss.item())  # Record test loss
                    print("epoch:{}, loss is {}".format(epoch, test_loss.item()))
                    # Save model checkpoint with architecture and epoch metadata
                    model_path = '{}/model_layer{}_hidden{}_epoch{}.pt'.format(path, layer_num, hidden_size, epoch)
                    torch.save(model.state_dict(), model_path)
            else:
                # Later training phase: reduced evaluation frequency
                if epoch % 500 == 0:
                    outputs = model(test_data)  # Generate predictions on test set
                    test_loss = criterion(outputs, test_labels)  # Compute test loss
                    losses.append(test_loss.item())  # Record test loss
                    print("epoch:{}, loss is {}".format(epoch, test_loss.item()))
                    # Save model checkpoint with architecture and epoch metadata
                    model_path = '{}/model_layer{}_hidden{}_epoch{}.pt'.format(path, layer_num, hidden_size, epoch)
                    torch.save(model.state_dict(), model_path)

        return losses  # Return test loss trajectory for analysis



