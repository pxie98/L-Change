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
        self.name = "jieti2"
        # 生成数据的参数
        self.func = ["0.125*x", "x*0.25", "x*0.5", "1.*x", "2*x", "4*x", "8*x"]
        # 需要拟合的函数
        self.start = [-8, -4, -2, -1, -0.5, -0.25, -0.125]  # 生成拟合数据的区间
        self.end = [8, 4, 2, 1, 0.5, 0.25, 0.125]  # 生成拟合数据的区间
        self.step = 0.01  # 生成拟合数据的步长

        # 模型参数
        self.input_size = 1           # 输入数据维度
        self.output_size = 1          # 输出数据维度
        self.layer_num_list = [2, 5]            # 模型的层数，最低是1，需要是整数
        self.hidden_size_list = [20, 50, 100, 300]           #range(1, 10+1, 1)  # 每一层NN的参数维度

        # 训练阶段的参数
        self.learning_rate = 0.01     # 模型训练的学习率
        self.num_epochs = 20001         # 模型训练的迭代次数
        self.path = ['model_x0728125','model_x072825','model_x07285','model_x07281','model_x07282','model_x07284','model_x07288']


        self.fig_path = ['model_x1_png', 'model_x2_png', 'model_x3_png', 'model_x07132_png', 'model_x07134_png', 'model_x07138_png']



args = Argument()
device = torch.device("cpu")


################################ 生成数据 #########################


def all_pic(args, device, i):
    # 定义输入数据和目标数据
    generator = dataGenerator(args.func[i], args.start[i], args.end[i] + args.step, args.step)
    x, y = generator.generate_data()
    save_file_name = 'func_fit/results/test_loss_{}.txt'.format(i)

    criterion = nn.MSELoss()

    path = args.path[i]


    for hidden_size in args.hidden_size_list:
        for layer_num in args.layer_num_list:
            losses = []

            for epoch in range(0, 20001, 100):

                for round in range(5):
                    
                    model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)

                    # 需要读取的文件名字
                    file_name = "func_fit/{}/model_layer{}_hidden{}_epoch{}_round{}.pt".format(path, layer_num, hidden_size, epoch, round)

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
for i in range(7):
    all_pic(args, device, i)


#######################################################################
