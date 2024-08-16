import read_txt
import numpy as np
import torch
import matplotlib.pyplot as plt

paths = ['model_x1','model_x2','model_x3','model_x4','model_x5','model_x6','model_x7']
func_name = ["sin(x)", "x*x", "0.5*x", "1.*x", "2.*x", "4.*x", "0.25*x"]
labels = ['layer=2, para=20', 'layer=2, para=50', 'layer=2, para=100', 'layer=2, para=300',
         'layer=5, para=20', 'layer=5, para=50', 'layer=5, para=100', 'layer=5, para=300']

def all_pic(paths, labels, func_name, start):
    for index in range(7):
        path = '{}/resulst_loss2.txt'.format(paths[index])
        pic_name = 'png3/{}.png'.format(index)
        data = read_txt.read_txt_data(path)

        x = np.arange(0, 20001, 500)


        for i in range(7):
            plt.plot(x[start:], data[i][start:], label=labels[i])


        plt.legend()
        plt.title(func_name[index])
        plt.savefig(pic_name)
        plt.close()

start = 4
all_pic(paths, labels, func_name, start)
index = 6
path = '{}/resulst_loss0.txt'.format(paths[index])
pic_name = '{}.png'.format(index)
data = read_txt.read_txt_data(path)

x = np.arange(0, 20001, 500)


for i in range(7):
    plt.plot(x, data[i], label=labels[i])


plt.legend()
plt.title(func_name[index])
plt.savefig(pic_name)
plt.close()