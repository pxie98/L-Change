# python库
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from scipy.interpolate import RectBivariateSpline

#自定义库
from data_generation import dataGenerator     # 生成数据样本的库
from NNnetwork import NeuralNetwork
from train_class import train_model
from save_data import save
import scipy.io as sio
torch.manual_seed(1)

'''
Training strategy: First train on y/2, then fine-tune on full y
'''
# Hyperparameter configuration class for video data approximation experiments
class Argument:
    def __init__(self):
        # Hardware configuration
        self.cuda_use = True

        # Data generation parameters
        self.sample = 1  # Sampling method: 1 for random sampling, 2 for sequential interval sampling

        # Neural network architecture parameters
        self.input_size = 2  # Input feature dimension (spatial coordinates)
        self.output_size = 1  # Output dimension (vorticity values)
        self.layer_num_list = [4]  # Number of hidden layers in network
        self.hidden_size_list = [100]  # Number of neurons per hidden layer

        # Training configuration
        self.learning_rate = 0.00001  # Optimization learning rate
        self.num_epochs = 1000001  # Total training epochs
        # Model storage paths
        self.path = ['./model_x1']

# Initialize hyperparameter configuration
args = Argument()

# Device configuration for computation
if args.cuda_use:
    # Check GPU availability and set appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def loaddata():
    """
    Load vorticity data from MATLAB file

    Returns:
        data: 3D tensor containing vorticity field data (time, width, height)
    """
    mat_file_path = 'func_video/approx2/data/vorticity_small.mat'
    mat_contents = sio.loadmat(mat_file_path)
    data = mat_contents["vorticity_small"]
    return data


def data_process(data):
    """
    Process 3D vorticity data into coordinate-value pairs

    Args:
        data: 3D vorticity field tensor (time, width, height)

    Returns:
        positions: Coordinate tensor of shape (n_samples, 2)
        labels: Vorticity values tensor of shape (n_samples, 1)
    """
    time, width, height = data.shape
    positions = []
    labels = []
    time = 1  # Use only first time step
    # Extract spatial coordinates and corresponding vorticity values
    for t in range(time):
        for i in range(15, width - 15):  # Exclude boundary regions
            for j in range(15, height - 15):
                positions.append([i - 15, j - 15])
                labels.append(data[t, i, j])
    positions = torch.tensor(positions)
    labels = torch.tensor(labels).reshape([-1, 1])
    return positions, labels


def yangtiao(args, img_data):
    """
    Perform 2D spline interpolation with random grid sampling

    Args:
        args: Hyperparameter configuration
        img_data: Input image tensor

    Returns:
        xy: Regular grid coordinate tensor
        data: Original values at regular grid points
        xy_random: Randomly sampled coordinate tensor
        y_new: Spline-interpolated values at random points
    """
    jiange = 20  # Grid spacing for interpolation
    # Extract image dimensions
    width, height = img_data.size()

    xy = []
    # Initialize data tensor for regular grid
    data = torch.zeros([int(width / jiange), int(height / jiange)])
    # Sample data at regular grid points
    for x in range(0, width, jiange):
        y_l = 0
        for y in range(0, height, jiange):
            xy.append([x, y])
            data[x_l, y_l] = img_data[x, y]
            y_l = y_l + 1
        x_l = x_l + 1

    import numpy as np
    # Create coordinate arrays for spline interpolation
    x_data = range(0, width, jiange)
    y_data = range(0, width, jiange)
    spline = RectBivariateSpline(x_data, y_data, data)

    # Generate random sampling points
    xy_random = []
    x_random = torch.randperm(width)[:int(width / jiange)]
    x_random, indices = torch.sort(x_random)
    x_random = x_random.tolist()
    y_random = torch.randperm(height)[:int(height / jiange)]
    y_random, indices = torch.sort(y_random)
    y_random = y_random.tolist()
    for x in x_random:
        for y in y_random:
            xy_random.append([x, y])

    # Evaluate spline at random points
    y_new = spline(x_random, y_random)

    # Convert results to PyTorch tensors
    y_new = torch.tensor(y_new, dtype=torch.float32)
    xy = torch.tensor(xy)
    xy_random = torch.tensor(xy_random, dtype=torch.float32)
    data = data.clone().detach().reshape([-1, 1])
    y_new = y_new.clone().detach().reshape([-1, 1])
    return xy, data, xy_random, y_new


# Training round identifier
round = 0

# Main training loop
for i in range(1):
    # Load and process vorticity data
    data = loaddata()
    x, y = data_process(data)
    print(x.shape)

    # Convert data types and move to computation device
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    x = x.to(device)
    y = y.to(device)

    print(y.size())
    path = args.path[i]

    # Architecture hyperparameter grid search
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            # Model initialization and training setup
            model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.MSELoss()
            train = train_model(args.num_epochs)

            # Execute training process (note: both phases use same data in this implementation)
            losses = train.train(model, optimizer, criterion, x, y, x, y, layer_num, hidden_size, path, round)
            print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))

