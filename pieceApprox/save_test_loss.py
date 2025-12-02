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
# 定义超参数类
class Argument:
    def __init__(self):
        self.cuda_use = True
        # 生成数据的参数
        self.name = "newlinear"
        self.func = ["0*x", "2*x+2", "0*x", "-x+1", "0*x"]
        self.start = [-5.0, -2., 0., 2., 5]  # 生成拟合数据的区间
        self.end = [-2., 0., 2., 5.]  # 生成拟合数据的区间
        self.step = 0.01          # 生成拟合数据的步长
 
        # 模型参数
        self.input_size = 1           # 输入数据维度
        self.output_size = 1          # 输出数据维度
        self.layer_num_list = [2,5]            # 模型的层数，最低是1，需要是整数
        self.hidden_size_list = [20,50,100,300]           #range(1, 10+1, 1)  # 每一层NN的参数维度

        # 训练阶段的参数
        self.learning_rate = 0.01     # 模型训练的学习率
        self.num_epochs = 20001         # 模型训练的迭代次数
        self.path = ['pieceApprox/model_x1','pieceApprox/model_x2','pieceApprox/model_x3']
# test_loss_0_1第一个1代表-x+1函数，第一个0代表0*x，第二个1代表-1.5--0.5区间

args = Argument()
device = torch.device("cpu")


################################ 生成数据 #########################

def multi_interval(func1, func2, func3, func4, start, step, end):
    # 创建数据生成器实例
    generator = dataGenerator(func2, -1.5 + step / 2, -0.5, step)
    x1, y1 = generator.generate_data()

    generator = dataGenerator(func3, 0.5 + step / 2, 1.5, step)
    x2, y2 = generator.generate_data()
    
    return x1, y1, x2, y2


def all_pic(args, device, i):
    # 定义输入数据和目标数据
    x1, y1, x2, y2 = multi_interval(args.func[0], args.func[1], args.func[i+2], args.func[4], args.start, args.step, args.end)
    save_file_name = 'pieceApprox/results/test_loss_{}_1.txt'.format(i)
    criterion = nn.MSELoss()
    path = args.path[i]
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            losses = []

            for epoch in range(0, 1001, 5):

                round = 0
                    
                model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)

                # 需要读取的文件名字
                file_name = "{}/model_layer{}_hidden{}_epoch{}_round{}.pt".format(path, layer_num, hidden_size, epoch, round)

                # 读取文件，载入模型
                model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
                #########################################################################

                ########################### 预测 #######################################
                optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
                optimizer.zero_grad()
                outputs = model(x1)
                loss = criterion(outputs, y1)
                loss = torch.log10(loss)
                if round == 0:
                    loss_ave = loss
                else:
                    loss_ave = loss_ave + loss
                loss_ave = loss_ave / (round + 1)
                losses.append(loss_ave.item())

            # for epoch in range(1000, 20000, 2000):
            #plt.plot(range(0, 20000, 500), losses, label='hidden={}, layer={}'.format(hidden_size, layer_num))

            with open(save_file_name, 'a+') as fl:
                data_name = 'loss_model_layer{}_hidden{} = '.format(layer_num, hidden_size)
                fl.write(data_name)
                fl.write('\n')
                fl.write(str(losses))
                fl.write('\n')

    save_file_name = 'pieceApprox/results/test_loss_{}_2.txt'.format(i)
    criterion = nn.MSELoss()
    path = args.path[i]
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            losses = []

            for epoch in range(0, 1001, 5):

                round = 0
                    
                model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)

                # 需要读取的文件名字
                file_name = "{}/model_layer{}_hidden{}_epoch{}_round{}.pt".format(path, layer_num, hidden_size, epoch, round)

                # 读取文件，载入模型
                model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
                #########################################################################

                ########################### 预测 #######################################
                optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
                optimizer.zero_grad()
                outputs = model(x2)
                loss = criterion(outputs, y2)
                loss = torch.log10(loss)
                if round == 0:
                    loss_ave = loss
                else:
                    loss_ave = loss_ave + loss
                loss_ave = loss_ave / (round + 1)
                losses.append(loss_ave.item())

            # for epoch in range(1000, 20000, 2000):
            #plt.plot(range(0, 20000, 500), losses, label='hidden={}, layer={}'.format(hidden_size, layer_num))

            with open(save_file_name, 'a+') as fl:
                data_name = 'loss_model_layer{}_hidden{} = '.format(layer_num, hidden_size)
                fl.write(data_name)
                fl.write('\n')
                fl.write(str(losses))
                fl.write('\n')
        

    #fig_file_name = 'png/{}/loss_model_interval_1_0.png'.format(fig_path)
    #plt.legend()
    #plt.show()
    #plt.savefig(fig_file_name)
    #plt.close()

    # plt.close()
for i in range(1):
    all_pic(args, device, i)


#######################################################################
