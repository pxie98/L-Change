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

torch.manual_seed(1)


# Hyperparameter configuration class for one-dimensional function approximation experiments
class Argument:
    def __init__(self):
        # Hardware acceleration settings
        self.cuda_use = True

        # Data generation parameters
        self.start = [-1, -1]  # Data generation intervals
        self.end = [1, 1]  # Data generation intervals
        self.num_points = 201  # Number of data points to generate
        self.coff = [-100, 100, -10, 10]  # Coefficient sets for linear transformations

        # Neural network architecture parameters
        self.input_size = 1  # Input feature dimension
        self.output_size = 1  # Output prediction dimension
        self.layer_num_list = [2, 5]  # Number of hidden layers to experiment with
        self.hidden_size_list = [20, 50, 100, 300]  # Hidden layer neuron counts

        # Training configuration
        self.learning_rate = 0.001  # Optimization learning rate
        self.num_epochs = 20001  # Total training epochs
        self.path_root = '.'  # Root directory for results
        # Model storage paths for different experimental conditions
        self.path = ['model_x1', 'model_x2', 'model_x3', 'model_x4', 'model_x5', 'model_x6', 'model_x7', 'model_x8']


# Initialize hyperparameter configuration
args = Argument()

# Device configuration for computation
if args.cuda_use:
    # Check GPU availability and set appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Main experimental loop across different coefficient configurations
for i in range(3):
    input_size = args.input_size

    # Generate training dataset with current coefficient
    generator = dataGenerator(args.start[0], args.end[0], args.num_points, args.coff[i])
    x1, y1 = generator.generate_data()

    # Generate testing/validation dataset with reference coefficient
    generator = dataGenerator(args.start[0], args.end[0], args.num_points, args.coff[3])
    x2, y2 = generator.generate_data()

    # Set model storage path for current experiment
    path = '{}/{}'.format(args.path_root, args.path[i])

    # Move datasets to computation device
    x1 = x1.to(device)
    y1 = y1.to(device)
    x2 = x2.to(device)
    y2 = y2.to(device)

    # Architecture hyperparameter grid search
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            # Multiple training rounds for statistical significance
            for round in range(1):
                # Model initialization and training setup
                model = NeuralNetwork(input_size, hidden_size, args.output_size, layer_num).to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                criterion = nn.MSELoss()
                train = train_model(args.num_epochs)

                # Execute training process with train/test datasets
                losses = train.train(model, optimizer, criterion, x1, y1, x2, y2, layer_num, hidden_size, path, round)
                print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))

                # Save training results to file for analysis
                save_file_name = '{}/results/test_loss_{}.txt'.format(args.path_root, i)
                with open(save_file_name, 'a+') as fl:
                    data_name = 'loss_model_layer{}_hidden{} = '.format(layer_num, hidden_size)
                    fl.write(data_name)
                    fl.write('\n')
                    fl.write(str(losses))
                    fl.write('\n')

