import torch
import numpy as np

class dataGenerator:
    def __init__(self, start, end, num_points, coff):
        self.start = start
        self.end = end
        self.num_points = num_points
        self.coff = coff
        self.step = (self.end - self.start) / (self.num_points - 1)

    def generate_data(self):
        #X = np.random.uniform(self.start, self.end, (self.num_points, self.x_dim))
        X = np.arange(self.start, self.end + self.step, self.step)
        y = self.coff * X
        X = torch.tensor(X, dtype=torch.float).view(-1, 1)
        y = torch.tensor(y, dtype=torch.float).view(-1, 1)
        return X, y