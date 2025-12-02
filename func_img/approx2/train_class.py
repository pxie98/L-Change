import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs


    def train(self, model, optimizer, criterion, train_data, train_labels, test_data, test_labels, layer_num, hidden_size, path, round, jiange):
        # Pretrain the model
        losses = []
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

        
        losses = []
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = model(test_data)
            loss = criterion(outputs, test_labels)
            loss.backward()
            optimizer.step()

            if epoch%10000==0:
                outputs = model(test_data)
                test_loss = criterion(outputs, test_labels)
                losses.append(test_loss.item())
                #print("epoch:{}, loss is {}".format(epoch,test_loss.item()))
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}_chajiange{}.pt'.format(path, layer_num, hidden_size, epoch, round, jiange)
                torch.save(model.state_dict(), model_path)


        return losses



