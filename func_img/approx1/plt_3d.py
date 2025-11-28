import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #绘制3D图菜
import torch



def plot_png(filename, title, fig_name, t, end_num):
    end_threshold = 20
    z_threshold = 7.0



    y = torch.tensor([2, 3, 4, 5, 8, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500])

    # Open the text file
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    for i in range(0, len(lines), 3):
        text = lines[i + 1].strip()
        list_data = eval(lines[i + 2])
        tensor_data = torch.from_numpy(np.array(list_data))
        if tensor_data.size()[0]<20000:
            tensor_data = torch.cat((tensor_data, tensor_data), dim=0)
        #print(tensor_data.size())
        data.append(tensor_data.tolist())

    # Convert to numpy array
    numpy_data = np.array(data)

    # Convert to tensor matrix
    tensor_data = torch.tensor(data)

    mask = (tensor_data > z_threshold)
    tensor_data[mask] = z_threshold



    z = tensor_data[16*t:16*(t+1), 0:end_threshold*1000:10].numpy()  # Convert to numpy array

    x = torch.arange(0, end_threshold*10000, 100).numpy()
    #x = np.tile(x, (17, 1))

    #y = np.tile(y, (100, 1))
    #y = y.T

    x = torch.tensor(x).flatten().numpy()
    y = torch.tensor(y).flatten().numpy()
    #z = torch.tensor(z).flatten().numpy()

    x_, y_ = np.meshgrid(x, y[:end_num], indexing='ij') #画图所要表现出来的主函数

    z=np.log10(z[:end_num,:])

    fig = plt.figure(figsize=(6, 6), facecolor='white')#创建图片
    sub = fig.add_subplot(111, projection='3d')  # 添加子图,
    surf = sub.plot_surface(y_, x_, z.T, cmap=plt.cm.brg)#绘制曲面,cmap=plt.cm.brg并设置彦
    cb=fig.colorbar(surf, shrink=0.8, aspect=15)#设置颜色棒
    sub.set_xlabel(r"Hidden Size")
    sub.set_ylabel(r"Iteration")
    sub.set_zlabel(r"MSE Loss")
    sub.set_title(title)
    sub.view_init(30, 50)
    plt.show()
    plt.savefig(fig_name)
    plt.close()



filename_all = 'resulst_0619_sin4.txt'
title_all = [r"f(x) = sin([2pi*x, 6pi*x]),  Layer Num = 2",
             r"f(x) = sin([2pi*x, 6pi*x]),  Layer Num = 3",
             r"f(x) = sin([2pi*x, 6pi*x]),  Layer Num = 4",
             r"f(x) = sin([2pi*x, 6pi*x]),  Layer Num = 5"]
fig_name_all = ['19-4-2.png', '19-4-3.png', '19-4-4.png', '19-4-5.png']
t_all = [0,1,2,3]
end_num = 16


for i in range(4):
    filename = filename_all
    title = title_all[i]
    fig_name = fig_name_all[i]
    t = t_all[i]
    plot_png(filename, title, fig_name, t, end_num)