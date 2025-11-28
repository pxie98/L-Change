import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    def __init__(self, num_epochs):
        # Total number of training epochs for each phase
        self.num_epochs = num_epochs

    def train(self, model, optimizer, criterion, x1, y1, x2, y2, layer_num, hidden_size, path, round):
        """
        Two-phase training procedure: pre-training followed by fine-tuning

        Args:
            model: Neural network model to train
            optimizer: Optimization algorithm instance
            criterion: Loss function criterion
            x1: Pre-training phase input features
            y1: Pre-training phase target values
            x2: Fine-tuning phase input features
            y2: Fine-tuning phase target values
            layer_num: Number of hidden layers in model architecture
            hidden_size: Number of neurons per hidden layer
            path: Directory path for model checkpoint storage
            round: Training round identifier for file naming

        Returns:
            losses: Loss values recorded during fine-tuning phase
        """
        # Training model - Phase 1: Pre-training on initial dataset
        losses = []
        for epoch in range(self.num_epochs):
            # Forward propagation
            optimizer.zero_grad()  # Clear gradients from previous iteration
            outputs = model(x1)  # Generate model predictions
            loss = criterion(outputs, y1)  # Compute loss on pre-training data

            # Backward propagation and optimization
            loss.backward()  # Compute gradients via backpropagation
            optimizer.step()  # Update model parameters

            # Loss monitoring at regular intervals
            if epoch % 10 == 0:
                losses.append(loss.item())
            # Progress reporting
            if epoch % 10000 == 0:
                print("epoch:{}, loss is {}".format(epoch, loss.item()))

        # Phase 2: Fine-tuning on secondary dataset
        losses = []  # Reset loss tracking for fine-tuning phase
        for epoch in range(self.num_epochs):
            # Forward propagation
            optimizer.zero_grad()  # Clear gradients from previous iteration
            outputs = model(x2)  # Generate model predictions
            loss = criterion(outputs, y2)  # Compute loss on fine-tuning data

            # Backward propagation and optimization
            loss.backward()  # Compute gradients via backpropagation
            optimizer.step()  # Update model parameters

            # Loss monitoring and checkpointing
            if epoch % 10 == 0:
                losses.append(loss.item())
            if epoch % 10000 == 0:
                print("epoch:{}, loss is {}".format(epoch, loss.item()))
                # Save model checkpoint at specified intervals
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}.pt'.format(path, layer_num, hidden_size, epoch,
                                                                                   round)
                torch.save(model.state_dict(), model_path)

        return losses