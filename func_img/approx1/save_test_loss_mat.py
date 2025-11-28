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

from scipy.io import savemat
torch.manual_seed(1)

# 定义超参数类
class Argument:
    def __init__(self):
        self.cuda_use = True
        # 生成数据的参数
        self.name = "newlinear"
        self.step = 0.01          # 生成拟合数据的步长
 
        # 模型参数
        self.input_size = 2           # 输入数据维度
        self.output_size = 1          # 输出数据维度
        self.layer_num_list = [5]            # 模型的层数，最低是1，需要是整数
        self.hidden_size_list = [20,50,100,300]           #range(1, 10+1, 1)  # 每一层NN的参数维度

        # 训练阶段的参数
        self.learning_rate = 0.0005     # 模型训练的学习率
        self.num_epochs = 200001         # 模型训练的迭代次数
        self.path = ['func_img/approx1/model_x1','func_img/model_x2','func_img/model_x3','func_img/model_x07281','func_linear/model_x07282','func_linear/model_x07284','func_linear/model_x07288']

# 先训练y/2后训练y

args = Argument()
if args.cuda_use:
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def data_process(img_data):
    width, height = img_data.size()
    positions = []
    pixes = []
    for i in range(width):
        for j in range(height):
            positions.append([i, j])
            pixes.append(img_data[i, j])
    positions = torch.tensor(positions)
    pixels = torch.tensor(pixes).reshape([-1,1])
    return positions, pixels

def all_pic(args, device, i):
    # 定义输入数据和目标数据
    generator = dataGenerator()
    img_data = generator.generate_data()
    x, y = data_process(img_data)
    x = x*1.
    y = y*1.
    x = x.to(device)
    y = y.to(device)/255
    x_raw = x
    y_raw = y
    data_size = y_raw.size()[0]
    sample_index = torch.randperm(data_size)[:int(torch.floor(torch.tensor(data_size/20)))]
    x = x[sample_index,:]
    y = y[sample_index,:]
    print(y.size())
    path = args.path[i]

    criterion = nn.MSELoss()
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            outputs_all = []

            for epoch in range(0, 200001, 10000):

                round = 0
                    
                model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)

                # 需要读取的文件名字
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}_zheng.pt'.format(path, layer_num, hidden_size, epoch, round)

                # 读取文件，载入模型
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
                #########################################################################

                ########################### 预测 #######################################
                optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
                optimizer.zero_grad()
                outputs = model(x_raw)


                outputs_all.append(outputs.detach().clone().reshape([-1]).tolist())

            mat_file_name = 'func_img/approx1/matfile/layer{}hz{}f{}_alldata_randomx.mat'.format(layer_num, hidden_size,i)
            savemat(mat_file_name, {'x':x_raw.detach().clone().tolist() , 'y': outputs_all})
    
   

    #fig_file_name = 'png/{}/loss_model_interval_1_0.png'.format(fig_path)
    #plt.legend()
    #plt.show()
    #plt.savefig(fig_file_name)
    #plt.close()

    # plt.close()
for i in range(1):
    all_pic(args, device, i)


#######################################################################
