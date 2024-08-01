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
    def __init__(self, func, start, end, step, num_points, coff):
        self.func = func
        self.start = start
        self.end = end
        self.step = step
        self.num_points = num_points
        self.coff = coff
        

    def function(self, x1, x2):
        return self.coff*x1 + self.coff*x2

    def generate_data(self):
        x1_vals = np.arange(-1, 1 + self.step, self.step)
        x2_vals = np.arange(-1, 1 + self.step, self.step)
        points = np.array([[x1, x2] for x1 in x1_vals for x2 in x2_vals])

        # 生成训练数据
        X = torch.tensor(points, dtype=torch.float)
        y = torch.tensor([self.function(x[0], x[1]) for x in points], dtype=torch.float).view(-1, 1)
        return X, y