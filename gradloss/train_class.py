import torch
import torch.nn as nn
import torch.optim as optim


class train_model():
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs


    def train(self, model, optimizer, criterion, train_data, train_labels, true_grad, test_data, test_labels, layer_num, hidden_size, path):
        # 训练模型
        true_grad = true_grad.reshape([-1, 1])
        
        losses = []
        losses1 = []
        losses2 = []
        
        
        for epoch in range(self.num_epochs):
            train_data = torch.tensor(train_data, requires_grad=True)
            # 前向传播
            optimizer.zero_grad()
            outputs = model(train_data)

            data_grad = torch.autograd.grad(outputs.sum(), train_data, create_graph=True)[0]

            loss2 =  (data_grad-true_grad) * (data_grad-true_grad) #/ ( (0.0001 + torch.sqrt((data_grad * data_grad))) * (0.0001 + torch.sqrt((true_grad * true_grad))) )
            loss =  - (data_grad*true_grad) / ( (0.0001 + torch.sqrt((data_grad * data_grad))) * (0.0001 + torch.sqrt((true_grad * true_grad))) ) + (data_grad - true_grad)* (data_grad - true_grad)
            loss1 = - (data_grad*true_grad) / ( (0.0001 + torch.sqrt((data_grad * data_grad))) * (0.0001 + torch.sqrt((true_grad * true_grad))) )
            #print("loss1 is {}".format(loss1.sum()))
            #print("loss2 is {}".format(loss2.sum()))
            #loss1 = - (1 + data_grad[0] * true_grad[0]) / torch.sqrt((1+data_grad[0]*data_grad[0]) * (1+true_grad[0]*true_grad[0]))
            #loss2 = 
            #for i in range(1,data_grad.size()[1]):
                #print(data_grad.size(), true_grad.size())


                #loss = loss - (1 + data_grad[i] * true_grad[i]) / torch.sqrt((1+data_grad[i]*data_grad[i]) * (1+true_grad[i]*true_grad[i]))

            # 反向传播和优化

            loss.sum().backward()
            optimizer.step()

            if epoch%500==0:
                outputs = model(test_data)
                test_loss = criterion(outputs, test_labels)
                losses.append(test_loss.item())
                losses1.append(loss1.sum().item())
                losses2.append(loss2.sum().item())
                

            if epoch%500==0:
                print("epoch:{}, loss is {}".format(epoch,test_loss.item()))
                model_path = '{}/model_layer{}_hidden{}_epoch{}2.pt'.format(path, layer_num, hidden_size, epoch)
                torch.save(model.state_dict(), model_path)


        return losses, losses1, losses2



