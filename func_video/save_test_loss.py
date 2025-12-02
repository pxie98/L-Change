# python库
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

#自定义库
from NNnetwork import NeuralNetwork
from train_class import train_model
from save_data import save
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(1)

# 定义超参数类
class Argument:
    def __init__(self):
        self.cuda_use = True

        # 模型参数
        self.input_size = 2           # 输入数据维度
        self.output_size = 1          # 输出数据维度
        self.layer_num_list = [4]            # 模型的层数，最低是1，需要是整数
        self.hidden_size_list = [100]           #range(1, 10+1, 1)  # 每一层NN的参数维度

        # 训练阶段的参数
        self.learning_rate = 1     # 模型训练的学习率
        self.num_epochs = 20001         # 模型训练的迭代次数
        self.path = ['model_x1','model_x2','model_x07285','model_x07281','model_x07282','model_x07284','model_x07288']


        self.fig_path = ['model_x1_png', 'model_x2_png', 'model_x3_png', 'model_x07132_png', 'model_x07134_png', 'model_x07138_png']



args = Argument()
device = torch.device("cpu")


################################ 生成数据 #########################


def loaddata():
    mat_file_path = 'func_video/approx2/data/vorticity_small.mat'
    mat_contents = sio.loadmat(mat_file_path)
    data=mat_contents["vorticity_small"]
    return data


def data_process(data):
    time, width, height = data.shape
    positions = []
    labels = []
    time = 1
    for t in range(time):
        for i in range(15, width-15):
            for j in range(15, height-15):
                positions.append([i-15, j-15])
                labels.append(data[t, i, j])
    positions = torch.tensor(positions)
    labels = torch.tensor(labels).reshape([-1,1])
    
    
    frame = torch.zeros((width-30, height-30))
    batch_size = width-30
    flag = 0
    for i in range(0, labels.size()[0], height-30):
        outputs = labels[i:i+height-30]
        frame[flag,:] = outputs.reshape([-1])
        flag += 1

    data = np.array(frame.data)
    # 绘制二维图
    plt.imshow(data, cmap='viridis', aspect='auto')  # cmap 是颜色映射，aspect='auto' 自动调整宽高比
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('Value')  # 设置颜色条的标签
    # 添加标题和坐标轴标签
    plt.title("2D Matrix Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # 显示图形
    plt.show()
    fig_name = 'fitpng_raw'
    plt.savefig(fig_name)
    return positions, labels, width, height

def positions_to_data(positions, labels, width, height):
    frame = torch.zeros((width, height))
    return 0
    


def all_pic(args, device, i):
    # 定义输入数据和目标数据
    data = loaddata()
    x, y, width, height = data_process(data)
    print(x.shape)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    root_file = 'func_video/approx2'
    save_file_name = '{}/results/test_loss_{}.txt'.format(root_file, i)

    criterion = nn.MSELoss()

    path = args.path[i]


    for hidden_size in args.hidden_size_list:
        for layer_num in args.layer_num_list:
            losses = []

            for epoch in range(0, 20001, 1000):

                for round in range(1):
                    
                    model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)

                    # 需要读取的文件名字
                    file_name = "{}/{}/model_layer{}_hidden{}_epoch{}_round{}.pt".format(root_file, path, layer_num, hidden_size, epoch, round)

                    # 读取文件，载入模型
                    model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
                    #########################################################################

                    ########################### 预测 #######################################
                    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
                    optimizer.zero_grad()

                    batch_size = 10000
                    data_size = x.size()[0]
                    data_index = torch.randperm(data_size)[:batch_size]

                    outputs = model(x[data_index])
                    loss = criterion(outputs, y[data_index])
                    #loss = torch.log10(loss)
                    
                losses.append(loss.item())

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

def save_raw_fig():
    data = loaddata()
    x, y, width, height = data_process(data)
    print(x.shape)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    print(y.reshape([170,-1]))
    
    root_file = 'func_video/approx2'
    

    criterion = nn.MSELoss()

    path = args.path[0]


    for hidden_size in args.hidden_size_list:
        for layer_num in args.layer_num_list:
            for round in range(1):
                epoch = 87000
                
                model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)

                # 需要读取的文件名字
                file_name = "{}/{}/model_layer{}_hidden{}_epoch{}_round{}.pt".format(root_file, path, layer_num, hidden_size, epoch, round)

                # 读取文件，载入模型
                model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
                #########################################################################

                ########################### 预测 #######################################
                optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
                optimizer.zero_grad()
                
                frame = torch.zeros((width-30, height-30))
                batch_size = width-30
                flag = 0
                for i in range(0, x.size()[0], height-30):
                    outputs = model(x[i:i+height-30])
                    #print(outputs.reshape([-1]))
                    frame[flag,:] = outputs.reshape([-1])
                    flag += 1
                print("f",frame)

                data = (frame.detach())
                # 绘制二维图
                plt.imshow(data, cmap='viridis', aspect='auto')  # cmap 是颜色映射，aspect='auto' 自动调整宽高比
                # 添加颜色条
                cbar = plt.colorbar()
                cbar.set_label('Value')  # 设置颜色条的标签
                # 添加标题和坐标轴标签
                plt.title("2D Matrix Plot")
                plt.xlabel("X-axis")
                plt.ylabel("Y-axis")
                # 显示图形
                plt.show()
                fig_name = 'fitpng'
                plt.savefig(fig_name)
                

    return 0

for i in range(1):
    #all_pic(args, device, i)
    save_raw_fig()


#######################################################################
