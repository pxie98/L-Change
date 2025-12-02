# python package
import torch
import torch.nn as nn
import torch.optim as optim
import copy

#自定义库
from data_generation import dataGenerator     # 生成数据样本的库
from NNnetwork import NeuralNetwork
from train_class import train_model
from save_data import save

torch.manual_seed(1)
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.interpolate import lagrange, Rbf

# Hyperparameter configuration class for interpolation-based function approximation
class Argument:
    def __init__(self):
        # Hardware configuration
        self.cuda_use = True

        # Data generation parameters
        self.func = ["torch.sin(x)"]  # Target function to approximate
        # Function fitting parameters
        self.start = [-torch.pi]  # Data generation interval start
        self.end = [torch.pi]  # Data generation interval end
        self.num_points = 201  # Number of data points to generate
        self.pol_num_points = 7  # Number of points for polynomial fitting
        self.times = [1, 2, 3]  # Polynomial degrees for interpolation

        # Neural network architecture parameters
        self.input_size = 1  # Input feature dimension
        self.output_size = 1  # Output prediction dimension
        self.layer_num_list = [5]  # Number of hidden layers in network
        self.hidden_size_list = [50]  # Number of neurons per hidden layer

        # Training configuration
        self.learning_rate = 0.001  # Optimization learning rate
        self.num_epochs = 50001  # Total training epochs
        self.path_root = '.'  # Root directory for results
        # Model storage paths for different interpolation methods
        self.path = ['model_x1', 'model_x2', 'model_x3', 'model_x4', 'model_x5', 'model_x6', 'model_x7', 'model_x8']


# Global interpolation methods mapping:
# x3: Linear interpolation
# x4: Quadratic interpolation
# x5: Cubic interpolation (i=2)
# Corresponding files: test_loss_2, test_loss_3, test_loss_4


# Initialize hyperparameter configuration
args = Argument()

# Device configuration for computation
if args.cuda_use:
    # Check GPU availability and set appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def poy_fun(args, pol_num_points, times, x):
    """
    Perform global polynomial interpolation

    Args:
        args: Hyperparameter configuration
        pol_num_points: Number of points for polynomial fitting
        times: Polynomial degree for interpolation
        x: Input tensor for prediction

    Returns:
        y_new: Polynomial approximation of target function
    """
    # Generate polynomial fitting points
    generator = dataGenerator(args.func[0], args.start[0], args.end[0], pol_num_points)
    pol_x, pol_y = generator.generate_data()

    # Fit polynomial to the entire dataset
    coefficients = np.polyfit(pol_x.reshape([-1]), pol_y.reshape([-1]), times)

    # Evaluate polynomial at input points
    y_new = np.polyval(coefficients, x)
    y_new = torch.tensor(y_new)
    return y_new


def yangtiao(args, pol_num_points, times, x):
    """
    Perform spline interpolation using scipy

    Args:
        args: Hyperparameter configuration
        pol_num_points: Number of points for spline fitting
        times: Unused parameter (maintained for interface consistency)
        x: Input tensor for prediction

    Returns:
        y_new: Spline interpolation of target function
    """
    generator = dataGenerator(args.func[0], args.start[0], args.end[0], pol_num_points)
    pol_x, pol_y = generator.generate_data()

    # Create spline interpolation representation
    tck = splrep(pol_x, pol_y)

    # Evaluate spline at input points
    y_new = splev(x, tck)
    y_new = torch.tensor(y_new, dtype=torch.float32)
    return y_new


def jingxiangji(args, pol_num_points, times, x):
    """
    Perform radial basis function interpolation

    Args:
        args: Hyperparameter configuration
        pol_num_points: Number of points for RBF fitting
        times: Unused parameter (maintained for interface consistency)
        x: Input tensor for prediction

    Returns:
        y_new: RBF interpolation of target function
    """
    generator = dataGenerator(args.func[0], args.start[0], args.end[0], pol_num_points)
    pol_x, pol_y = generator.generate_data()

    # Define radial basis function interpolator with Gaussian kernel
    rbf = Rbf(pol_x, pol_y, function='gaussian')

    # Perform RBF interpolation prediction
    y_new = rbf(x)
    y_new = torch.tensor(y_new, dtype=torch.float32)
    return y_new


def langariange(args, pol_num_points, times, x):
    """
    Perform Lagrange polynomial interpolation

    Args:
        args: Hyperparameter configuration
        pol_num_points: Number of points for Lagrange interpolation
        times: Unused parameter (maintained for interface consistency)
        x: Input tensor for prediction

    Returns:
        y_new: Lagrange interpolation of target function
    """
    generator = dataGenerator(args.func[0], args.start[0], args.end[0], pol_num_points)
    pol_x, pol_y = generator.generate_data()

    # Construct Lagrange polynomial interpolator
    poly = lagrange(pol_x.numpy().flatten(), pol_y.numpy().flatten())

    # Perform Lagrange interpolation prediction
    y_new = poly(x)
    y_new = torch.tensor(y_new, dtype=torch.float32)
    return y_new


# Main experimental loop for different interpolation methods
for i in range(2, 5):
    # Generate base dataset
    generator = dataGenerator(args.func[0], args.start[0], args.end[0], args.num_points)
    x, y = generator.generate_data()
    path = '{}/{}'.format(args.path_root, args.path[i])

    # Select interpolation method based on index
    if i > 1 and i < 4.5:
        # Polynomial interpolation: linear, quadratic, cubic
        pol_y = poy_fun(args, args.pol_num_points, args.times[i - 2], x)
    elif i == 5:
        # Spline interpolation
        pol_y = yangtiao(args, args.pol_num_points, args.times[0], x)
    elif i == 6:
        # Radial basis function interpolation
        pol_y = jingxiangji(args, args.pol_num_points, args.times[0], x)
    elif i == 7:
        # Lagrange interpolation
        pol_y = langariange(args, args.pol_num_points, args.times[0], x)

    print("x", x.dtype)
    print("y", pol_y.dtype)

    # Compute residual for neural network to learn
    new_y = y - pol_y

    # Move data to computation device
    x = x.to(device)
    new_y = new_y.to(device)

    # Architecture hyperparameter grid search
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            # Multiple training rounds for statistical significance
            for round in range(1):
                # Model initialization and training setup
                model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                criterion = nn.MSELoss()
                train = train_model(args.num_epochs)

                # Execute training process on residual data
                losses = train.train(model, optimizer, criterion, x, new_y, layer_num, hidden_size, path, round)
                print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))

                # Save training results to file
                save_file_name = '{}/results/test_loss_{}.txt'.format(args.path_root, i)
                with open(save_file_name, 'a+') as fl:
                    data_name = 'loss_model_layer{}_hidden{} = '.format(layer_num, hidden_size)
                    fl.write(data_name)
                    fl.write('\n')
                    fl.write(str(losses))
                    fl.write('\n')




