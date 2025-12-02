# python库
import torch
import torch.nn as nn
import torch.optim as optim

#自定义库
from data_generation import dataGenerator     # 生成数据样本的库
from NNnetwork import NeuralNetwork
from train_class import train_model
from save_data import save
import matplotlib.pyplot as plt
import numpy as np
import copy
torch.manual_seed(1)


# 定义超参数类
class Argument:
    def __init__(self):
        self.cuda_use = True
        # 生成数据的参数
        self.name = "newlinear"
        self.func = ["0*x"]
        # 需要拟合的函数
        self.start = [-1, -1]        # 生成拟合数据的区间
        self.end = [1, 1]          # 生成拟合数据的区间
        self.num_points = 201          # 生成拟合数据的步长
        self.coff = [[1,10,20],[1,10]]
 
        # 模型参数
        self.input_size = [2, 3]           # 输入数据维度
        self.output_size = 1          # 输出数据维度
        self.layer_num_list = [2,5]            # 模型的层数，最低是1，需要是整数
        self.hidden_size_list = [20,50,100,300]           #range(1, 10+1, 1)  # 每一层NN的参数维度

        # 训练阶段的参数
        self.learning_rate = 0.01     # 模型训练的学习率
        self.num_epochs = 20001         # 模型训练的迭代次数
        self.path = ['approx2/model_x1','approx2/model_x2','approx2/model_x3','approx2/model_x4','approx2/model_x5']



args = Argument()
device = torch.device("cpu")


################################ 生成数据 #########################


def all_pic(args, device, i, path_num):
    input_size = args.input_size[i]
    coffs = args.coff[i]
    for coff in coffs:
        coff = [coff for i in range(input_size)]
        print(input_size, coff)
        # 定义输入数据和目标数据
        generator = dataGenerator(args.func[0], args.start[0], args.end[0], args.num_points, coff)
        x, y = generator.generate_data()
        
        
        path_num = path_num + 1

        save_file_name = 'approx2/results/test_loss_{}.txt'.format(path_num)

        criterion = nn.MSELoss()

        path = args.path[path_num]
        #fig_path = args.fig_path[path_num]

        for hidden_size in args.hidden_size_list:
            print(i,hidden_size)
            for layer_num in args.layer_num_list:
                losses = []

                for epoch in range(0, 20001, 500):

                    for round in range(5):
                        
                        model = NeuralNetwork(input_size, hidden_size, args.output_size, layer_num).to(device)

                        # 需要读取的文件名字
                        file_name = "{}/model_layer{}_hidden{}_epoch{}_round{}.pt".format(path, layer_num, hidden_size, epoch, round)

                        # 读取文件，载入模型
                        model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
                        #########################################################################

                        ########################### 预测 #######################################
                        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
                        optimizer.zero_grad()
                        outputs = model(x)
                        loss = criterion(outputs, y)
                        loss = torch.log10(loss)
                        if round == 0:
                            loss_ave = loss 
                        else:
                            loss_ave = loss_ave + loss
                    loss_ave = loss_ave / round

                    losses.append(loss_ave.item())


                with open(save_file_name, 'a+') as fl:
                    data_name = 'loss_model_layer{}_hidden{} = '.format(layer_num, hidden_size)
                    fl.write(data_name)
                    fl.write('\n')
                    fl.write(str(losses))
                    fl.write('\n')
    
                

path_num = -1   
for i in range(2):
    if i==1:
        path_num = 2
    all_pic(args, device, i, path_num)



