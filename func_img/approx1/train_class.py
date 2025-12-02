import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs


    def train(self, model, optimizer, criterion, pretrain_data, pretrain_labels, traindata, trainlabels, layer_num, hidden_size, path, round):
        # Train the model
        losses = []
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = model(pretrain_data)
            loss = criterion(outputs, pretrain_labels/2)
            loss.backward()
            optimizer.step()

            if epoch%10000==0:
                outputs = model(traindata)
                test_loss = criterion(outputs, trainlabels/2)
                losses.append(test_loss.item())
                #print("epoch:{}, loss is {}".format(epoch,test_loss.item()))
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}_pretrain.pt'.format(path, layer_num, hidden_size, epoch, round)
                torch.save(model.state_dict(), model_path)
        
        losses = []
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = model(pretrain_data)
            loss = criterion(outputs, pretrain_labels)
            loss.backward()
            optimizer.step()

            if epoch%10000==0:
                outputs = model(traindata)
                test_loss = criterion(outputs, trainlabels)
                losses.append(test_loss.item())
                #print("epoch:{}, loss is {}".format(epoch,test_loss.item()))
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}_zheng.pt'.format(path, layer_num, hidden_size, epoch, round)
                torch.save(model.state_dict(), model_path)


        return losses



