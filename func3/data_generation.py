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
    def __init__(self, func, start, end, num_points, coff):
        self.func = func
        self.start = start
        self.end = end
        self.num_points = num_points
        self.coff = coff
        self.x_dim = len(coff)

    def function(self, x):
        return eval(self.func)

    def generate_data(self):
        X = np.random.uniform(self.start, self.end, (self.num_points, self.x_dim))
        y = X[:,-1]
        for i in range(self.x_dim-1):
            y = y + self.coff[i]*X[:, i] 
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(-1, 1)
        return X, y