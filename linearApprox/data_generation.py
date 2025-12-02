import torch
import numpy as np

class dataGenerator:
    '''
    Synthetic data generator for multi-dimensional function sampling.
    
    Parameters:
    -----------
    start : float
        Lower bound of the sampling interval for all dimensions
    end : float
        Upper bound of the sampling interval for all dimensions  
    step : float
        Sampling resolution (step size between consecutive points)
    coff : float
        Coefficient scaling factor for the linear function
    '''
    def __init__(self, start, end, step, coff):
        self.start = start
        self.end = end
        self.step = step
        self.coff = coff
        
    def function(self, x1, x2, x3):
        """
        Defines the target mathematical function for data generation.
        
        Args:
            x1, x2, x3: Input variables representing 3D coordinates
            
        Returns:
            Linear combination: coff*x1 + coff*x2 + coff*x3
            This represents a 3D linear function with equal coefficients
        """
        return self.coff * x1 + self.coff * x2 + self.coff * x3

    def generate_data(self):
        """
        Generates a comprehensive 3D grid dataset with corresponding function values.
        
        Methodology:
        - Creates equidistant points along each dimension using np.arange
        - Generates full factorial design using Cartesian product
        - Computes target values using the defined linear function
        - Converts results to PyTorch tensors for ML pipeline compatibility
        
        Returns:
            X : torch.Tensor 
                Input feature matrix of shape (n_samples, 3) containing 3D coordinates
            y : torch.Tensor
                Target values of shape (n_samples, 1) containing function outputs
        """
        # Generate equidistant sampling points for each dimension
        x1_vals = np.arange(-1, 1 + self.step, self.step)
        x2_vals = np.arange(-1, 1 + self.step, self.step)
        x3_vals = np.arange(-1, 1 + self.step, self.step)
        
        # Create full 3D grid using Cartesian product of all dimension values
        points = np.array([[x1, x2, x3] 
                          for x1 in x1_vals 
                          for x2 in x2_vals 
                          for x3 in x3_vals])

        # Convert to PyTorch tensors with float precision for neural network compatibility
        X = torch.tensor(points, dtype=torch.float32)
        
        # Compute target values and reshape to column vector (n_samples, 1)
        y = torch.tensor([self.function(x[0], x[1], x[2]) for x in points], 
                        dtype=torch.float32).view(-1, 1)
        
        return X, y