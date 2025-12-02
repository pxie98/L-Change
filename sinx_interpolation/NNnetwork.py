import torch
import torch.nn as nn
import torch.optim as optim

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.layer_inter = nn.Linear(hidden_size, hidden_size)
        self.layer_end = nn.Linear(hidden_size, output_size)
        self.layer_num = layer_num

        self.layer_onlyone = nn.Linear(input_size, output_size)

    def forward(self, x):
        if self.layer_num == 1:                  # 一层网络
            x = self.layer_onlyone(x)
        else:                                     # 多于一层网络
            x = self.layer1(x)
            x = self.activation(x)
            if self.layer_num > 2:                    # 多于二层网络
                for i in range(self.layer_num-2):
                    x = self.layer_inter(x)
                    x = self.activation(x)
            x = self.layer_end(x)
        return x