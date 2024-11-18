import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    # train the train data
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs

    def train(self, model, optimizer, criterion, train_data, train_labels, test_data, test_labels, layer_num, hidden_size, path, round):
        losses = []
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            if epoch%100==0:
                losses.append(loss.item())
            if epoch%10000==0:
                print("epoch:{}, loss is {}".format(epoch, loss.item()))
        return losses, model



