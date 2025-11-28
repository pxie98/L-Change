import torch
import torch.nn as nn
import torch.optim as optim
import copy

from data_generation import dataGenerator    
from NNnetwork import NeuralNetwork
from train_class import train_model
from save_data import save

torch.manual_seed(1)


# Hyperparameter configuration class for neural network training
class Argument:
    def __init__(self):
        # Hardware acceleration settings
        self.cuda_use = True
        
        # Data generation parameters
        # Trigonometric functions with varying frequencies and combinations
        self.func = ["torch.sin(2*x)", "torch.sin(6*x)", "torch.sin(10*x)", "torch.sin(18*x)", 
                    "torch.sin(2*x)+torch.sin(6*x)", "torch.sin(2*x)+torch.sin(6*x)+torch.sin(10*x)", 
                    "torch.sin(2*x)+torch.sin(6*x)+torch.sin(10*x)+torch.sin(18*x)"]
        # Target functions to approximate
        self.start = [-2, -1]        # Data generation intervals
        self.end = [2, 1]            # Data generation intervals
        self.step = 0.001            # Sampling resolution

        # Neural network architecture parameters
        self.input_size = 1           # Input feature dimension
        self.output_size = 1          # Output prediction dimension
        self.layer_num_list = [2,5]   # Number of hidden layers to experiment with
        self.hidden_size_list = [20,50,100,300]  # Hidden layer neuron counts

        # Training configuration
        self.learning_rate = 0.01     # Optimization learning rate
        self.num_epochs = 20001       # Total training epochs
        # Model checkpoint storage paths
        self.path = ['./model_x1','./model_x2','./model_x3',
                    './model_x4','./model_x5','./model_x6',
                    './model_x7']

# Initialize hyperparameter configuration
args = Argument()

# Device configuration for computation
if args.cuda_use:
    # Check GPU availability and set appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Main training loop across all target functions
for i in range(7):
    # Generate training data for current target function
    generator = dataGenerator(args.func[i], args.start[0], args.end[0], args.step)
    x, y = generator.generate_data()
    
    # Set model storage path for current function
    path = args.path[i]

    # Dataset shuffling and train-test split
    data_index = torch.randperm(torch.tensor(x.size()[0]))
    train_data_size = torch.round(torch.tensor(x.size()[0] * 0.8))
    train_data_size = train_data_size.int()

    # Apply shuffling to dataset
    x = x[data_index]
    y = y[data_index]
    print(f"Training set size: {train_data_size}, Original data shape: {x.size()}")
    
    # Split data into training and testing sets
    train_data = x[:train_data_size].to(device)
    train_labels = y[:train_data_size].to(device)
    test_data = x[train_data_size:].to(device)
    test_labels = y[train_data_size:].to(device)

    # Architecture hyperparameter grid search
    for layer_num in args.layer_num_list:
        # Evaluate different hidden layer sizes
        for hidden_size in args.hidden_size_list:
            # Model initialization with current architecture
            model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)
            # Adam optimizer with specified learning rate
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            # Mean Squared Error loss function for regression
            criterion = nn.MSELoss()
            
            # Initialize training module and execute training
            train = train_model(args.num_epochs)
            losses = train.train(model, optimizer, criterion, train_data, train_labels, 
                               test_data, test_labels, layer_num, hidden_size, path)
            
            # Print final training loss for current configuration
            print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))