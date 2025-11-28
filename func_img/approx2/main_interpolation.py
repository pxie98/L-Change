# python库
import torch
import torch.nn as nn
import torch.optim as optim
import copy

#自定义库
from data_generation import dataGenerator     # 生成数据样本的库
from NNnetwork import NeuralNetwork
from train_class import train_model
from save_data import save
import numpy as np

torch.manual_seed(1)
from scipy.interpolate import splrep, splev, RectBivariateSpline


# Hyperparameter configuration class for image approximation with spline interpolation
class Argument:
    def __init__(self):
        # Hardware configuration
        self.cuda_use = True

        # Data generation parameters
        self.name = "newlinear"
        self.step = 0.01  # Step size for data generation
        self.sample = 1  # Sampling method: 1 for random sampling, 2 for sequential interval sampling

        # Neural network architecture parameters
        self.input_size = 2  # Input feature dimension (pixel coordinates)
        self.output_size = 1  # Output dimension (pixel intensity)
        self.layer_num_list = [5]  # Number of hidden layers in network
        self.hidden_size_list = [50, 300]  # Hidden layer neuron counts
        self.jianges = [2, 4, 8, 16, 32, 64, 128]  # Grid intervals for spline interpolation

        # Training configuration
        self.learning_rate = 0.0005  # Optimization learning rate
        self.num_epochs = 200001  # Total training epochs
        # Model storage paths for different grid intervals
        self.path = ['func_img/approx2/model_x1', 'func_img/approx2/model_x2', 'func_img/approx2/model_x3',
                     'func_img/approx2/model_x4', 'func_img/approx2/model_x5', 'func_img/approx2/model_x6',
                     'func_img/approx2/model_x7']


# Initialize hyperparameter configuration
args = Argument()

# Device configuration for computation
if args.cuda_use:
    # Check GPU availability and set appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def data_process(img_data):
    """
    Convert image data to coordinate-pixel value pairs

    Args:
        img_data: Input image tensor

    Returns:
        positions: Coordinate tensor of shape (n_pixels, 2)
        pixels: Pixel intensity tensor of shape (n_pixels, 1)
    """
    width, height = img_data.size()
    positions = []
    pixes = []
    # Extract all pixel coordinates and intensity values
    for i in range(width):
        for j in range(height):
            positions.append([i, j])
            pixes.append(img_data[i, j])
    positions = torch.tensor(positions)
    pixels = torch.tensor(pixes).reshape([-1, 1])
    return positions, pixels


def yangtiao(args, img_data, jiange):
    """
    Perform 2D spline interpolation on image data with specified grid spacing

    Args:
        args: Hyperparameter configuration
        img_data: Input image tensor
        jiange: Grid spacing for spline interpolation

    Returns:
        xy: Coordinate tensor of sampled grid points
        data: Original pixel values at grid points
        y_new: Spline-interpolated values at grid points
    """
    # Extract image dimensions
    width, height = img_data.size()

    xy = []
    # Calculate grid dimensions
    x_l = 0
    for x in range(0, width, jiange):
        y_l = 0
        for y in range(0, height, jiange):
            y_l = y_l + 1
        x_l = x_l + 1

    # Initialize data tensor for grid points
    data = torch.zeros([x_l, y_l])
    x_l = 0
    # Sample image data at grid points
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

    # Perform bivariate spline interpolation
    spline = RectBivariateSpline(x_data, y_data, data)
    y_new = spline(x_data, y_data)

    # Convert results to PyTorch tensors
    y_new = torch.tensor(y_new, dtype=torch.float32)
    xy = torch.tensor(xy)
    data = data.clone().detach().reshape([-1, 1])
    y_new = y_new.clone().detach().reshape([-1, 1])
    return xy, data, y_new


# Training round identifier
round = 0

# Main experimental loop across different grid intervals
for i in range(4, 7):
    # Generate and process image data
    generator = dataGenerator()
    img_data = generator.generate_data()
    x_raw, y_raw = data_process(img_data)
    print("raw", x_raw.dtype, y_raw.size())

    # Apply spline interpolation with current grid interval
    jiange = args.jianges[i]
    print("jiange", jiange)
    x, y, y_new = yangtiao(args, img_data, jiange)
    print("train", x.dtype, y.size(), y_new.size())

    # Data normalization
    x = x * 1.
    y = y * 1.
    y = y / 255  # Normalize pixel values to [0,1] range

    # Prepare training data based on sampling strategy
    if args.sample == 1:
        # Random sampling: select 5% of data points randomly
        data_size = y_raw.size()[0]
        sample_index = torch.randperm(data_size)[:int(torch.floor(torch.tensor(data_size / 20)))]
        x_train = (x_raw[sample_index, :] * 1.).to(device)
        y_train = (y_raw[sample_index, :] * 1. / 255).to(device)
    else:
        # Sequential interval sampling: select every 20th data point
        x_train = (x_raw[::20, :] * 1.).to(device)
        y_train = (y_raw[::20, :] * 1. / 255).to(device)

    # Move all data to computation device
    y_new = (y_new * 1. / 255).to(device)
    x = x.to(device)
    y = y.to(device)
    x_raw = x_raw.to(device)
    path = args.path[i]

    # Architecture hyperparameter grid search
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            # Model initialization and training setup
            model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.MSELoss()
            train = train_model(args.num_epochs)

            # Execute two-phase training: first on spline data, then on sampled image data
            losses = train.train(model, optimizer, criterion, x, y_new, x_train, y_train,
                                 layer_num, hidden_size, path, round, jiange)
            print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))

            # Note: Image generation code is currently commented out
            # pred_img = model(x_raw)*255
            # pred_img = (torch.round(pred_img.data)).int().reshape([-1]).tolist()
            # img_name = 'func_img/approx2/results/ln{}_hs{}_lr{}.png'.format(layer_num, hidden_size, args.learning_rate)
            # a = generator.out_img(img_name, pred_img)