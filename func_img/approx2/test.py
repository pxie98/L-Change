# python库
import torch
import torch.nn as nn
import torch.optim as optim
import copy
#自定义库
from data_generation import dataGenerator     # 生成数据样本的库
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
        # 生成数据的参数
        self.func = ['0*x', '2*x+2', '-x+1', '0*x' ]     # 需要拟合的函数
        self.start = [-5,-2,0,2,5]        # 生成拟合数据的区间
        self.end = [-5,-2,0,2,5]          # 生成拟合数据的区间
        self.step = 0.001           # 生成拟合数据的步长

        # 模型参数
        self.input_size = 1           # 输入数据维度
        self.output_size = 1          # 输出数据维度
        self.layer_num_list = [2,5]            # 模型的层数，最低是1，需要是整数
        self.hidden_size_list = [20,50,100]           #range(1, 10+1, 1)  # 每一层NN的参数维度

        # 训练阶段的参数
        self.learning_rate = 0.01     # 模型训练的学习率
        self.num_epochs = 20000         # 模型训练的迭代次数
        self.path = 'func_fit/model_x07071'
        self.fig_path = 'func_fit/model_x01_png'




args = Argument()
if args.cuda_use:
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


################################ 生成数据 #########################
# 创建数据生成器实例
i = 0
for func_name in args.func:
    i += 1
    # 创建数据生成器实例
    generator = dataGenerator(func_name, args.start[i-1]+args.step/2, args.end[i], args.step)

    # 定义输入数据和目标数据
    if i==1:
        x, y = generator.generate_data()
        new_x = copy.deepcopy(x)
        new_y = copy.deepcopy(y)
    else:
        x, y = generator.generate_data()
        new_x = torch.cat((new_x, x), dim=0)
        new_y = torch.cat((new_y, y), dim=0)
    
   
x = copy.deepcopy(new_x)
y = copy.deepcopy(new_y)



#######################################################################


x = x.to(device)
y = y.to(device)
############################ 载入模型 ################################
print(x.size())
matrix = torch.zeros(20, x.size()[0])
flag = 0
for func_name in args.func:
    hidden_size = 50
    layer_num = 5
    for epoch in range(0, 20000, 1000):
        model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)

        # 需要读取的文件名字
        file_name = '{}/model_layer{}_hidden{}_epoch{}.pt'.format(args.path, layer_num, hidden_size, epoch)

        # 读取文件，载入模型
        model.load_state_dict(torch.load(file_name))
        #########################################################################


        ########################### 预测 #######################################
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        optimizer.zero_grad()
        outputs = model(x)
        criterion = nn.MSELoss()
        matrix[flag,:] = (outputs - y).reshape([-1])
        flag += 1

        fig_file_name = '{}/model_layer{}_hidden{}_epoch{}.png'.format(args.fig_path, layer_num, hidden_size, epoch)


epochs = torch.range(0, 19000, 1000)
print('x:', x.size())
print('y:', epochs.size())
print('matrix', matrix.size())
x_, y_ = np.meshgrid(x.cpu(), epochs, indexing='ij') #画图所要表现出来的主函数

z = np.log10(matrix.cpu().detach().numpy())

fig = plt.figure(figsize=(6, 6), facecolor='white')#创建图片
sub = fig.add_subplot(111, projection='3d')  # 添加子图,
surf = sub.plot_surface(y_, x_, z.T, cmap=plt.cm.brg)#绘制曲面,cmap=plt.cm.brg并设置彦
cb=fig.colorbar(surf, shrink=0.8, aspect=15)#设置颜色棒
sub.set_xlabel(r"Hidden Size")
sub.set_ylabel(r"Iteration")
sub.set_zlabel(r"MSE Loss")




sub.view_init(30, 50)
plt.show()
plt.savefig(fig_file_name)
plt.close()


plt.plot(train_data.cpu().detach().numpy(), matrix.cpu().detach().numpy())
plt.savefig(fig_file_name)

#######################################################################






'''
for func_name in args.func:
    # 创建数据生成器实例
    generator = dataGenerator(func_name, args.start, args.end, args.step)

    # 定义输入数据和目标数据
    x, y = generator.generate_data()
    data_index = torch.randperm(torch.tensor(x.size()[0]))
    train_data_size = torch.round(torch.tensor(x.size()[0]*0.8))
    train_data_size = train_data_size.int()

    x = x[data_index]
    y = y[data_index]
    print(train_data_size, x.size())
    train_data = x[:train_data_size].to(device)
    train_labels = y[:train_data_size].to(device)
    test_data = x[train_data_size:].to(device)
    test_labels = y[train_data_size:].to(device)


    for layer_num in args.layer_num_list:
        
        # 计算不同hidden size下的损失值
        for hidden_size in args.hidden_size_list:
            # 定义模型，优化器，损失函数
            model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
            criterion = nn.MSELoss()
            train = train_model(args.num_epochs)
            losses = train.train(model, optimizer, criterion, train_data, train_labels, test_data, test_labels, layer_num, hidden_size, args.path)
            print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))
            

            # 保存超参数和损失值
            #save('resulst_0627_sin5.txt', losses, hidden_size, layer_num, func_name, args)
'''
# 横坐标hidden size，纵坐标loss
#plot_figure(x=args.hidden_size_list, y=losses, xlabel='hidden_size', ylabel='loss', title='None')

