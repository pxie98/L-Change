import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs

    def train(self, model, optimizer, criterion, x, y, layer_num, hidden_size, path, round):
        # Train model
        losses = []
        for epoch in range(self.num_epochs):
            # 前向传播
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if epoch%10==0:
                losses.append(loss.item())
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}.pt'.format(path, layer_num, hidden_size, epoch, round)
                torch.save(model.state_dict(), model_path)

            if epoch%10000==0:
                print("epoch:{}, loss is {}".format(epoch, loss.item()))

        return losses



