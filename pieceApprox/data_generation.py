import torch

class dataGenerator:
    '''
    Data generator for creating synthetic datasets from mathematical functions.
    
    This class generates paired (x, y) samples by evaluating a given mathematical
    function over a specified interval with defined sampling resolution.
    
    Args:
        func (str): Mathematical function expression to be fitted (e.g., "2*x+2")
        start (float): Starting point of the sampling interval
        end (float): Ending point of the sampling interval (exclusive)
        step (float): Sampling resolution step size
    
    Example:
        >>> generator = DataGenerator("x**2", -1, 1, 0.5)
        >>> x, y = generator.generate_data()
        # x: tensor([[-1.0], [-0.5], [0.0], [0.5]])
        # y: tensor([[1.0], [0.25], [0.0], [0.25]])
    '''
    
    def __init__(self, func, start, end, step):
        """
        Initialize the data generator with function specification and sampling parameters.
        
        Args:
            func (str): Mathematical expression in terms of 'x'
            start (float): Lower bound of sampling interval
            end (float): Upper bound of sampling interval
            step (float): Discretization step size
        """
        self.func = func  # Mathematical function expression
        self.start = start  # Interval start point
        self.end = end  # Interval end point
        self.step = step  # Sampling step size

    def function(self, x):
        """
        Evaluate the mathematical function at given input points.
        
        Args:
            x (torch.Tensor): Input tensor containing x-values
            
        Returns:
            torch.Tensor: Output tensor with evaluated function values
            
        Note:
            Uses eval() for dynamic expression evaluation - ensure input safety
            in production environments.
        """
        return eval(self.func)  # Dynamically evaluate the mathematical expression

    def generate_data(self):
        """
        Generate synthetic dataset by sampling the function over specified interval.
        
        Returns:
            tuple: (x, y) where:
                x (torch.Tensor): Input features of shape (n_samples, 1)
                y (torch.Tensor): Target values of shape (n_samples, 1)
                
        Steps:
            1. Create linearly spaced x-values using torch.arange
            2. Evaluate function at each x-point
            3. Reshape tensors to 2D format (samples, features)
            4. Convert to float32 precision for neural network compatibility
        """
        # Generate linearly spaced input values
        x = torch.arange(self.start, self.end, self.step)
        
        # Compute corresponding output values through function evaluation
        y = self.function(x)
        
        # Reshape and convert data to PyTorch tensors with proper dimensions
        # unsqueeze(dim=1) converts 1D tensor to 2D: (n_samples,) -> (n_samples, 1)
        x = torch.tensor(x).unsqueeze(dim=1).float()  # Input features
        y = torch.tensor(y).unsqueeze(dim=1).float()  # Target values
        
        return x, y