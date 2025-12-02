import torch
import numpy as np

class dataGenerator:
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
        y = y * 0
        for i in range(self.x_dim):
            y = y + self.coff[i]*X[:, i] 
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(-1, 1)
        return X, y
    
    def generate_data1(self):
        X = np.random.uniform(self.start, self.end, (self.num_points, self.x_dim))
        y = np.sin(X[:,0]) + np.sin(X[:,1])
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(-1, 1)
        
        g = X[:,-1]
        g = g * 0
        for i in range(self.x_dim):
            g = g + self.coff[i]*X[:, i] 
        g = torch.tensor(g, dtype=torch.float).view(-1, 1)
        return X, y, g