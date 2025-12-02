import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, RectBivariateSpline
from scipy import interpolate

def loaddata():
    mat_file_path = 'func_video/approx2/data/vorticity_small.mat'
    mat_contents = sio.loadmat(mat_file_path)
    data=mat_contents["vorticity_small"]
    return data


def data_process(data, args):
    time, width, height = data.shape
    positions = []
    labels = []
    print("time=", time)
    jiange = 1
    y_chazhi = []
    for t in range(time):

        import numpy as np
        x_data = range(0,width,jiange)
        y_data = range(0,height,jiange)
        spline = RectBivariateSpline(x_data, y_data, data[t])

        xy_random = []
        x_random = torch.randperm(width)[:int(width/jiange)]
        x_random, indices = torch.sort(x_random)
        x_random = x_random.tolist()
        y_random = torch.randperm(height)[:int(height/jiange)]
        y_random, indices = torch.sort(y_random)
        y_random = y_random.tolist()
        for x in x_random:
            for y in y_random:
                xy_random.append([x, y])
        y_new = spline(x_random, y_random)
        # 计算新的插值点

        y_new = torch.tensor(y_new, dtype=torch.float32)
        
        xy_random = torch.tensor(xy_random, dtype=torch.float32)
        
        y_new = y_new.clone().detach().reshape([-1]).tolist()
        y_chazhi.append(y_new)

        
        # 绘制二维图
        plt.imshow(data[t], cmap='viridis', aspect='auto', vmin=-40. * args.multi, vmax=40. * args.multi)  # cmap 是颜色映射，aspect='auto' 自动调整宽高比
        # 添加颜色条
        cbar = plt.colorbar()
        cbar.set_label('Value')  # 设置颜色条的标签
        # 添加标题和坐标轴标签
        plt.title("2D Matrix Plot")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        # 显示图形
        plt.show()
        fig_name = 'func_video/approx2/results/fitpngraw.png'
        plt.savefig(fig_name)
        plt.close()

        for i in range(width):
            for j in range(height):
                positions.append([t, i, j])
                labels.append(data[t, i, j])
    y_chazhi = torch.tensor(y_chazhi, dtype=torch.float32).reshape([-1,1])
    positions = torch.tensor(positions, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).reshape([-1,1])
    return positions, labels * args.multi, y_chazhi * args.multi, time, width, height

def all_data_process(data, args):
    time, width, height = data.shape
    positions = []
    labels = []
    for t in range(time):
        for i in range(width):
            for j in range(height):
                positions.append([t, i, j])
                labels.append(data[t, i, j])
    
    positions = torch.tensor(positions, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).reshape([-1,1])
    return positions, labels * args.multi, time, width, height

def data_process_sp_time(data, sp_time, args):
    time, width, height = data.shape
    positions = []
    labels = []
    for t in range(sp_time, sp_time+1):
        for i in range(width):
            for j in range(height):
                positions.append([t, i, j])
                labels.append(data[t, i, j])
    positions = torch.tensor(positions, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).reshape([-1,1])
    return positions, labels * args.multi, width, height

def pred_png(model, sp_time, epoch, train_labels, args):
    data = loaddata()
    x, y, width, height = data_process_sp_time(data, sp_time, args)
    device = torch.device("cuda")
    y = y.to(device)
    if args.chazhi_or_not:
        y_chazhi = (y - train_labels).reshape([-1])
    x = x.to(device)

    frame = torch.zeros((width, height)).to(device)
    batch_size = width
    flag = 0
    for i in range(0, x.size()[0], height):
        if args.chazhi_or_not:
            outputs = model(x[i:i+height]).reshape([-1]) + y_chazhi[i:i+height]
        else:
            outputs = model(x[i:i+height]) 
        #print(outputs.reshape([-1]))
        frame[flag,:] = outputs.reshape([-1])
        flag += 1
    #print("f",frame)

    frame = frame.to("cpu")

    data = (frame.detach())
    # 绘制二维图
    plt.imshow(data, cmap='viridis', aspect='auto', vmin=-40. * args.multi, vmax=40. * args.multi)      # cmap 是颜色映射，aspect='auto' 自动调整宽高比
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('Value')  # 设置颜色条的标签
    # 添加标题和坐标轴标签
    plt.title("2D Matrix Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # 显示图形
    plt.show()
    if args.chazhi_or_not:
        index = 1
    else:
        index = 0
    fig_name = 'func_video/approx2/results/fitpngtime{}epoch{}chazhi{}'.format(sp_time, epoch, index)
    plt.savefig(fig_name)
    plt.close()
    return 0


def save_pred_mat(model, train_data, time, width, height, epoch, args):
    device = torch.device("cuda")
    frames = torch.zeros((time, width, height)).to(device)
    batch_size = width
    flag = 0
    index = 0
    for k in range(time):
        for i in range(width):
            outputs = model(train_data[flag:flag+height])
            #print(outputs.reshape([-1]))
            frames[k,i,:] = outputs.reshape([-1])
            flag += height
    
    #print("f",frame)
    if args.chazhi_or_not:
        index = 1
    else:
        index = 0
    output_file_path = 'func_video/approx2/pred_data/dataepoch{}chazhi{}.mat'.format(epoch, index)
    sio.savemat(output_file_path, {'data': frames.to('cpu').detach().numpy()})
    return 0

def plot_loss(losses, loss_epoch_inter, num_epochs, args):
    root_file = 'func_video/approx2'
    save_file_name = '{}/resultsloss/test_loss.txt'.format(root_file)
    if args.chazhi_or_not:
        index = 1
    else:
        index = 0
    with open(save_file_name, 'a+') as fl:
        data_name = 'loss_chazhi{} = '.format(index)
        fl.write(data_name)
        fl.write('\n')
        fl.write(str(losses))
        fl.write('\n')
    
    x = range(0, num_epochs, loss_epoch_inter)
    # 创建图形
    plt.plot(x, losses, marker='o')  # marker='o' 表示在数据点上画圆圈
    # 添加标题和标签
    #plt.title("Simple 2D Line Plot")
    plt.xlabel("Iter")
    plt.ylabel("Loss")
    # 显示图形
    plt.show()
    loss_fig_name = 'func_video/approx2/resultsloss/loss_chazhi{}.png'.format(index)
    plt.savefig(loss_fig_name)
    plt.close()

    import numpy as np
    plt.plot(x, np.log10(np.array(losses)), marker='o')  # marker='o' 表示在数据点上画圆圈
    # 添加标题和标签
    #plt.title("Simple 2D Line Plot")
    plt.xlabel("Iter")
    plt.ylabel("Loss")
    # 显示图形
    plt.show()
    loss_fig_name = 'func_video/approx2/resultsloss/loss_log10_chazhi{}.png'.format(index)
    plt.savefig(loss_fig_name)
    plt.close()
    return 0

class train_model():
    def __init__(self, num_epochs, batch_size):
        self.num_epochs = num_epochs
        self.batch_size = batch_size


    def train(self, model, optimizer, criterion, train_data, train_labels, time, width, height, layer_num, hidden_size, path, round, args):
        # 训练模型
        losses = []
        batch_size = self.batch_size
        data_size = train_data.size()[0]
        

        for epoch in range(self.num_epochs):

            if epoch % 1000 == 0:
                for sp_time in range(10):
                    pred_png(model, sp_time, epoch, train_labels[616*sp_time: 616*(sp_time+1)], args)

            # 前向传播
            optimizer.zero_grad()
            data_index = torch.randperm(data_size)[:batch_size]
            outputs = model(train_data[data_index])
            loss = criterion(outputs.reshape([-1,1]), train_labels[data_index])

            # 反向传播和优化

            loss.backward()
            optimizer.step()

            if epoch % args.loss_epoch_inter == 0:
                data_index = torch.randperm(data_size)[:batch_size]
                outputs = model(train_data[data_index])
                test_loss = criterion(outputs.reshape([-1,1]), train_labels[data_index])
                losses.append(test_loss.item())

            if epoch % 1000 == 0:
                print("epoch:{}, loss is {}".format(epoch, test_loss.item()))
                

            if epoch%500==0:
                # 保存模型
                #print("epoch:{}, loss is {}".format(epoch,test_loss.item()))
                model_path = '{}/model_layer{}_hidden{}_epoch{}_round{}.pt'.format(path, layer_num, hidden_size, epoch, round)
                torch.save(model.state_dict(), model_path)
            
            if epoch%100==0:
                # 保存预测数据
                save_pred_mat(model, train_data, time, width, height, epoch, args)
        
        plot_loss(losses, args.loss_epoch_inter, self.num_epochs, args)

        return losses



