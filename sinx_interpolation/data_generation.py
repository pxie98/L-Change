import torch
import numpy as np

class dataGenerator:
    '''
    func: 需要拟合的函数；
    [start, end]：生成数据样本的区间；
    step：生成数据的步长
    example：
    输入：func: f(x), start=-1, end=1, step=0.5,
    输出：x就是tensor[-1, -0.5, 0, 0.5, 1], y就是 tensor[f(-1), f(-0.5), f(0), f(0.5), f(1)]
    '''
    def __init__(self, func, start, end, num_points):
        self.func = func
        self.start = start
        self.end = end
        self.num_points = num_points
        self.step = (self.end - self.start) / (self.num_points - 1)

    def function(self, x):
        return eval(self.func)

    def generate_data(self):
        x = torch.arange(self.start, self.end + self.step, self.step)
        y = self.function(x)
        # 转换数据为PyTorch的张量
        x = torch.tensor(x).unsqueeze(dim=1).float()
        y = torch.tensor(y).unsqueeze(dim=1).float()
        return x, y