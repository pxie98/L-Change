import torch


class dataGenerator:
    '''
    func: 需要拟合的函数；
    [start, end]：生成数据样本的区间；
    step：生成数据的步长
    example：
    输入：func: f(x), start=-1, end=1, step=0.5,
    输出：x就是tensor[-1, -0.5, 0, 0.5, 1], y就是 tensor[f(-1), f(-0.5), f(0), f(0.5), f(1)]
    '''
    def __init__(self, func, start, end, step):
        self.func = func
        self.start = start
        self.end = end
        self.step = step

    def function(self, x):
        return eval(self.func)

    def generate_data(self):
        x = torch.arange(self.start, self.end, self.step)
        x = torch.tensor(x, requires_grad=True)
        y = self.function(x)
        y.sum().backward()
        grad = x.grad.clone()
        # 转换数据为PyTorch的张量
        print(y)
        x = torch.tensor(x).unsqueeze(dim=1).float()
        y = torch.tensor(y).unsqueeze(dim=1).float()
        return x, y, grad