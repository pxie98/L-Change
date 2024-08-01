import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs


    def train(self, model, optimizer, criterion, train_data, train_labels, test_data, test_labels, layer_num, hidden_size, path):
        # 训练模型
        
        losses = []
        for epoch in range(self.num_epochs):
            # 前向传播
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)

            # 反向传播和优化

            loss.backward()
            optimizer.step()

            if epoch%500==0:
                outputs = model(test_data)
                test_loss = criterion(outputs, test_labels)
                losses.append(test_loss.item())
                

            if epoch%500==0:
                print("epoch:{}, loss is {}".format(epoch,test_loss.item()))
                model_path = '{}/model_layer{}_hidden{}_epoch{}.pt'.format(path, layer_num, hidden_size, epoch)
                torch.save(model.state_dict(), model_path)


        return losses



