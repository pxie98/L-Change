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
import scipy.io as sio
torch.manual_seed(1)
from scipy.interpolate import splrep, splev, RectBivariateSpline
from scipy import interpolate

from train_class import loaddata, data_process

'''
Training modes:
1. Original data minus interpolated data
2. Original data directly
'''


# Hyperparameter configuration class for video data approximation with interpolation options
class Argument:
    def __init__(self):
        # Hardware configuration
        self.cuda_use = True

        # Data generation parameters
        self.sample = 1  # Sampling method: 1 for random sampling, 2 for sequential interval sampling

        # Neural network architecture parameters
        self.input_size = 3  # Input feature dimension (spatial coordinates + time)
        self.output_size = 1  # Output dimension (vorticity values)
        self.layer_num_list = [8]  # Number of hidden layers in network
        self.hidden_size_list = [50]  # Number of neurons per hidden layer

        # Training configuration
        self.learning_rate = 0.01  # Optimization learning rate
        self.batch_size = 512  # Mini-batch size for training
        self.num_epochs = 10001  # Total training epochs

        self.loss_epoch_inter = 100  # Epoch interval for loss value saving
        self.chazhi_or_not = False  # Training mode: True for residual (original - interpolated), False for original data
        self.multi = 10  # Multiplier factor (usage context depends on implementation)
        # Model storage paths
        self.path = ['./model_x1', './model_x2', './model_x3']


# Training strategy: First train on y/2, then fine-tune on full y

# Initialize hyperparameter configuration
args = Argument()

# Device configuration for computation
if args.cuda_use:
    # Check GPU availability and set appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Note: GPU cache clearing (commented out)
# torch.cuda.empty_cache()  # Clear GPU cache

# Training round identifier
round = 0

# Main training loop
for i in range(1):
    # Load and process vorticity data
    data = loaddata()
    x, y, y_chazhi, time, width, height = data_process(data, args)
    print(x.shape)

    # Move data to computation device
    x = x.to(device)
    y = y.to(device)
    y_chazhi = y_chazhi.to(device)

    # Save interpolated data to MATLAB file for analysis
    output_file_path = 'chazhidata/chazhidata.mat'
    sio.savemat(output_file_path, {'chazhi_data': y_chazhi.to('cpu').detach().numpy()})
    print("saved")

    # Compute residual between original data and interpolated data
    y_process = y - y_chazhi  # Difference between true data and interpolated data for each training round

    # Print data characteristics for verification
    print(y)
    print(y_process)
    print(y_chazhi)
    path = args.path[i]

    # Architecture hyperparameter grid search
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            # Model initialization and training setup
            model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.MSELoss()
            train = train_model(args.num_epochs, args.batch_size)

            # Execute training based on selected mode
            if args.chazhi_or_not:
                # Train on residual data (original - interpolated)
                losses = train.train(model, optimizer, criterion, x, y_process, time, width, height,
                                     layer_num, hidden_size, path, round, args)
            else:
                # Train directly on original data
                losses = train.train(model, optimizer, criterion, x, y, time, width, height,
                                     layer_num, hidden_size, path, round, args)
            print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))