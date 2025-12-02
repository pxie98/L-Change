# python库
import torch
import torch.nn as nn
import torch.optim as optim

#自定义库
from data_generation import dataGenerator     # 生成数据样本的库
from NNnetwork import NeuralNetwork
from train_class import train_model
from save_data import save
import matplotlib.pyplot as plt
import numpy as np
import copy

from scipy.io import savemat
torch.manual_seed(1)


# Hyperparameter configuration class for neural network training experiments
class Argument:
    def __init__(self):
        # Hardware configuration
        self.cuda_use = True  # Flag for CUDA GPU acceleration
        
        # Data generation parameters
        self.name = "newlinear"  # Experiment identifier
        self.func = ["2*x+2", "0*x", "-x+1"]  # Mathematical functions to approximate
        self.start = [-2., 0., 2., 5]  # Start points for data intervals
        self.end = [0., 2., 5.]  # End points for data intervals
        self.step = 0.01  # Sampling step size for data generation

        # Neural network architecture parameters
        self.input_size = 1  # Dimension of input features
        self.output_size = 1  # Dimension of output predictions
        self.layer_num_list = [2,5]  # Number of hidden layers to experiment with
        self.hidden_size_list = [20,50,100,300]  # Number of neurons per hidden layer

        # Training configuration
        self.learning_rate = 0.01  # Optimization learning rate
        self.num_epochs = 20001  # Total number of training epochs
        self.path = ['./model_x1','./model_x2','./model_x3']  # Model save paths

# Note: test_loss_0_1 naming convention:
# First digit represents function index, second digit represents interval index
# Example: test_loss_0_1 - first 0 represents "0*x" function, 
# second 1 represents interval [-1.5, -0.5]

# Initialize hyperparameters and device configuration
args = Argument()
device = torch.device("cpu")  # Force CPU computation


################################ Data Generation Functions #########################
def multi_interval(func1, func2, start, step, end):
    """
    Generate training data from multiple function intervals
    
    Args:
        func1: Primary mathematical function expression
        func2: Secondary mathematical function expression  
        start: List of interval start points
        step: Sampling step size
        end: List of interval end points
        
    Returns:
        x: Combined input tensor from both intervals
        y: Combined output tensor from both intervals
    """
    # Create data generator for first function interval
    generator = dataGenerator(func1, start[0]+step/2, end[0], step)
    x, y = generator.generate_data()

    # Initialize combined datasets
    new_x = copy.deepcopy(x)
    new_y = copy.deepcopy(y)

    # Generate data for second function interval
    generator = dataGenerator(func2, start[1]+step/2, end[1], step)
    x, y = generator.generate_data()
    
    # Concatenate datasets from both intervals
    new_x = torch.cat((new_x, x), dim=0)
    new_y = torch.cat((new_y, y), dim=0)

    x = copy.deepcopy(new_x)
    y = copy.deepcopy(new_y)
    
    return x, y


def all_pic(args, device, i):
    """
    Generate predictions across multiple epochs and save results
    
    Args:
        args: Hyperparameter configuration object
        device: Computation device (CPU/GPU)
        i: Function index for model selection
    """
    # Generate multi-interval training data
    x, y = multi_interval(args.func[0], args.func[i+1], args.start, args.step, args.end)
    
    # Initialize loss function
    criterion = nn.MSELoss()
    
    # Set model storage path
    path = args.path[i]
    
    # Iterate over different network architectures
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            outputs_all = []  # Store predictions across epochs

            # Sample predictions at regular epoch intervals
            for epoch in range(0, 1001, 5):
                round = 0  # Training round identifier
                    
                # Initialize neural network model
                model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)

                # Construct model filename for loading
                file_name = "{}/model_layer{}_hidden{}_epoch{}_round{}.pt".format(path, layer_num, hidden_size, epoch, round)

                # Load pre-trained model weights
                model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
                #########################################################################

                ########################### Prediction Phase #######################################
                # Initialize optimizer (required for model execution context)
                optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
                optimizer.zero_grad()  # Clear any existing gradients
                
                # Generate model predictions
                outputs = model(x)

                # Store detachted predictions for analysis
                outputs_all.append(outputs.detach().clone().reshape([-1]).tolist())

            # Save predictions to MATLAB format for further analysis
            mat_file_name = 'pieceApprox/matfile/layer{}hz{}f{}.mat'.format(layer_num, hidden_size,i)
            savemat(mat_file_name, {'data': outputs_all})
    
    # Note: Visualization code is currently commented out
    #fig_file_name = 'png/{}/loss_model_interval_1_0.png'.format(fig_path)
    #plt.legend()
    #plt.show()
    #plt.savefig(fig_file_name)
    #plt.close()


#######################################################################
# Main execution loop for all function combinations
for i in range(2):
    all_pic(args, device, i)


