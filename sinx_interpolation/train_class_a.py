import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs


    def train(self, model, optimizer, criterion, train_data, train_labels, test_data, test_labels, layer_num, hidden_size, path, round):
        # Train model  Pretrain
        losses = []
        y = train_labels / 2
        for epoch in range(self.num_epochs[0]):
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if epoch%10==0:
                losses.append(loss.item())
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}.pt'.format(path[0], layer_num, hidden_size, epoch, round)
                torch.save(model.state_dict(), model_path)
            if epoch%10000==0:
                print("epoch:{}, loss is {}".format(epoch, loss.item()))
        
        save_file_name = 'func3_poly/group1/results/test_loss_0.txt'
        with open(save_file_name, 'a+') as fl:
            data_name = 'loss_model_layer{}_hidden{} = '.format(layer_num, hidden_size)
            fl.write(data_name)
            fl.write('\n')
            fl.write(str(losses))
            fl.write('\n')

        # Train model
        losses = []
        for epoch in range(self.num_epochs[1]):
            # 前向传播
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            if epoch%10==0:
                losses.append(loss.item())
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}.pt'.format(path[1], layer_num, hidden_size, epoch, round)
                torch.save(model.state_dict(), model_path)
            if epoch%10000==0:
                print("epoch:{}, loss is {}".format(epoch, loss.item()))
        save_file_name = 'func3_poly/group1/results/test_loss_1.txt'
        with open(save_file_name, 'a+') as fl:
            data_name = 'loss_model_layer{}_hidden{} = '.format(layer_num, hidden_size)
            fl.write(data_name)
            fl.write('\n')
            fl.write(str(losses))
            fl.write('\n')

        return losses



