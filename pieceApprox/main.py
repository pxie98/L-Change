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



# Hyperparameter configuration class for neural network training
class Argument:
    def __init__(self):
        # Hardware acceleration settings
        self.cuda_use = True  # Flag for GPU utilization
        
        # Data generation parameters
        self.func = ["2*x+2", "0*x", "-x+1"]  # Mathematical functions for data generation
        self.start = [-2., 0., 2., 5]  # Starting points of data intervals
        self.end = [0., 2., 5.]       # Ending points of data intervals  
        self.step = 0.01              # Sampling step size for data generation

        # Neural network architecture parameters
        self.input_size = 1           # Dimensionality of input features
        self.output_size = 1          # Dimensionality of output predictions
        self.layer_num_list = [2,5]   # Number of hidden layers to experiment with
        self.hidden_size_list = [20,50,100,300]  # Number of neurons per hidden layer

        # Training configuration
        self.learning_rate = 0.01     # Optimization step size
        self.num_epochs = 1001        # Total training iterations
        self.path = ['./model_x1','./model_x2','./model_x3']  # Model save paths


# Initialize hyperparameters
args = Argument()

# Device configuration for GPU/CPU computation
if args.cuda_use:
    # Check GPU availability and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def multi_interval(func1, func2, start, step, end, i):
    """
    Generate multi-interval training data with stratified sampling
    
    Args:
        func1: Primary mathematical function for data generation
        func2: Secondary mathematical function for data generation  
        start: Interval starting points
        step: Data sampling resolution
        end: Interval ending points
        i: Interval index for sampling strategy selection
    
    Returns:
        x: Combined feature tensor across intervals
        y: Combined target tensor across intervals
    """
    # Generate data from primary function in first interval
    generator = dataGenerator(func1, start[0] + step / 2, end[0], 0.02)
    x, y = generator.generate_data()
    
    # Apply stratified sampling based on interval index
    if i == 1:
        # Random subsampling for 40% of data
        data_index = torch.randperm(torch.tensor(x.size()[0]))
        x_size = torch.round(torch.tensor(x.size()[0] * 0.4)).int()
        x = x[data_index][:x_size]
        y = y[data_index][:x_size]

    if i == 0:
        # Random subsampling for 70% of data  
        data_index = torch.randperm(torch.tensor(x.size()[0]))
        x_size = torch.round(torch.tensor(x.size()[0] * 0.7)).int()
        x = x[data_index][:x_size]
        y = y[data_index][:x_size]

    # Initialize combined datasets
    new_x = copy.deepcopy(x)
    new_y = copy.deepcopy(y)

    # Generate data from secondary function in subsequent interval
    generator = dataGenerator(func2, start[1] + step / 2, end[1], step)
    x, y = generator.generate_data()

    if i == 0:
        # Additional sampling for first interval case
        data_index = torch.randperm(torch.tensor(x.size()[0]))
        x_size = torch.round(torch.tensor(x.size()[0] * 0.7)).int()
        x = x[data_index][:x_size]
        y = y[data_index][:x_size]
        
    # Concatenate datasets from both intervals
    new_x = torch.cat((new_x, x), dim=0)
    new_y = torch.cat((new_y, y), dim=0)

    return new_x, new_y


# Main training loop
round = 0
for i in range(2):
    # Generate multi-interval training data
    x, y = multi_interval(args.func[0], args.func[i+1], args.start, args.step, args.end, i)
    print("Feature tensor x:", x.reshape([-1]))
    print("Target tensor y:", y.reshape([-1]))
    
    # Configure model save path
    path = args.path[i]

    # Dataset shuffling and train-test split
    data_index = torch.randperm(torch.tensor(x.size()[0]))
    train_data_size = torch.round(torch.tensor(x.size()[0] * 1.0)).int()

    x = x[data_index]
    y = y[data_index]
    print(f"Training set size: {train_data_size}, Original data shape: {x.size()}")
    
    # Move data to appropriate device (GPU/CPU)
    train_data = x[:train_data_size].to(device)
    train_labels = y[:train_data_size].to(device)
    test_data = x[train_data_size:].to(device)
    test_labels = y[train_data_size:].to(device)

    # Architecture hyperparameter grid search
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            # Model initialization
            model = NeuralNetwork(args.input_size, hidden_size, args.output_size, layer_num).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.MSELoss()  # Mean Squared Error loss for regression
            
            # Training execution
            train = train_model(args.num_epochs)
            losses = train.train(model, optimizer, criterion, train_data, train_labels, 
                               test_data, test_labels, layer_num, hidden_size, path, round)
            
            print(f'Layer num: {layer_num}, Hidden size: {hidden_size}, Final loss: {losses[-1]:.6f}')

            # Save training results and hyperparameters
            # save('resulst_0627_sin5.txt', losses, hidden_size, layer_num, args.func[i+1], args)

# Visualization of results (commented out)
# plot_figure(x=args.hidden_size_list, y=losses, xlabel='hidden_size', 
#            ylabel='loss', title='None')