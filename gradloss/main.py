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

torch.manual_seed(1)



# 定义超参数类
class Argument:
    def __init__(self):
        self.cuda_use = True
        # 生成数据的参数
        self.name = "newlinear"
        self.func = ["torch.sin(x)", "x*x", "0.5*x", "1.*x", "2.*x", "4.*x", "0.25*x"]
        # 需要拟合的函数
        self.start = [-2]        # 生成拟合数据的区间
        self.end = [2]          # 生成拟合数据的区间
        self.step = 0.0001          # 生成拟合数据的步长
 
        # 模型参数
        self.input_size = 1           # 输入数据维度
        self.output_size = 1          # 输出数据维度
        self.layer_num_list = [2,5]            # 模型的层数，最低是1，需要是整数
        self.hidden_size_list = [20,50,100,300]           #range(1, 10+1, 1)  # 每一层NN的参数维度

        # 训练阶段的参数
        self.learning_rate = 0.01     # 模型训练的学习率
        self.num_epochs = 20001         # 模型训练的迭代次数
        self.path = ['gradloss1/model_x1','gradloss1/model_x2','gradloss1/model_x3','gradloss1/model_x4','gradloss1/model_x5','gradloss1/model_x6','gradloss1/model_x7']



args = Argument()
if args.cuda_use:
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")



def two_interval(func1, func2, start, step, end):
    # 创建数据生成器实例
    generator = dataGenerator(func1, start[0] + step / 2, end[0], step)
    x, y = generator.generate_data()
    new_x = copy.deepcopy(x)
    new_y = copy.deepcopy(y)

    generator = dataGenerator(func2, start[1] + step / 2, end[1], step)
    x, y = generator.generate_data()
    new_x = torch.cat((new_x, x), dim=0)
    new_y = torch.cat((new_y, y), dim=0)

    x = copy.deepcopy(new_x)
    y = copy.deepcopy(new_y)
    return x, y



for i in range(7):

    # 定义输入数据和目标数据
    generator = dataGenerator(args.func[i], args.start[0], args.end[0]+args.step, args.step)
    x, y, true_grad = generator.generate_data()
    

    path = args.path[i]

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
    true_grad = true_grad[:train_data_size].to(device)


    for layer_num in args.layer_num_list:
        
        # 计算不同hidden size下的损失值
        for hidden_size in args.hidden_size_list:
            # 定义模型，优化器，损失函数
            model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.MSELoss()
            train = train_model(args.num_epochs)
            losses, losses1, losses2 = train.train(model, optimizer, criterion, train_data, train_labels, true_grad, test_data, test_labels, layer_num, hidden_size, path)
            print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))
            

            # 保存超参数和损失值
            save_path0 = '{}/resulst_loss0.txt'.format(path)
            save_path1 = '{}/resulst_loss1.txt'.format(path)
            save_path2 = '{}/resulst_loss2.txt'.format(path)
            save(save_path0, losses, hidden_size, layer_num, args.func[i], args)
            save(save_path1, losses1, hidden_size, layer_num, args.func[i], args)
            save(save_path2, losses2, hidden_size, layer_num, args.func[i], args)

# 横坐标hidden size，纵坐标loss
#plot_figure(x=args.hidden_size_list, y=losses, xlabel='hidden_size', ylabel='loss', title='None')


