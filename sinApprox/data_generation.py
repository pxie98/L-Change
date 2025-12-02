import torch
import numpy as np

class dataGenerator:
    '''
    func: Target function to be approximated;
    [start, end]: Interval for generating data samples;
    step: Step size for data generation
    example:
    Input: func: f(x), start=-1, end=1, step=0.5,
    Output: x is tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]]), y is tensor([[f(-1)], [f(-0.5)], [f(0)], [f(0.5)], [f(1)]])
    '''
    def __init__(self, func, start, end, step):
        # Mathematical function expression to be approximated
        self.func = func
        # Start point of the sampling interval
        self.start = start
        # End point of the sampling interval
        self.end = end
        # Sampling resolution step size
        self.step = step

    def function(self, x):
        """
        Evaluate the mathematical function at given input points
        
        Args:
            x: Input tensor containing x-values
            
        Returns:
            Evaluated function values at input points
        """
        return eval(self.func)

    def generate_data(self):
        """
        Generate synthetic dataset by sampling the function over specified interval
        
        Returns:
            x: Input features tensor of shape (n_samples, 1)
            y: Target values tensor of shape (n_samples, 1)
        """
        # Generate linearly spaced input values across specified interval
        x = torch.arange(self.start, self.end, self.step)
        # Compute corresponding function outputs
        y = self.function(x)
        # Print generated y values for verification
        # Convert to PyTorch tensors and reshape to 2D format
        # unsqueeze(dim=1) adds feature dimension: (n_samples,) -> (n_samples, 1)
        x = torch.tensor(x).unsqueeze(dim=1).float()
        y = torch.tensor(y).unsqueeze(dim=1).float()
        return x, y