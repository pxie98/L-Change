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

# 拟合三位函数，系数是coff

# Define hyperparameter class
class Argument:
    def __init__(self):
        self.cuda_use = True
        # Parameters for generating data
        self.name = "newlinear"
        self.func = ["0*x"]
        # Function to be fitted
        self.start = [-1, -1]        # Interval for generating fitting data
        self.end = [1, 1]          # Interval for generating fitting data
        self.step = 0.05
        self.num_points = 201          # Step size for generating fitting data
        self.coff = [1,10]
 
        # Model parameters
        self.input_size = [3]           # Input data dimension
        self.output_size = 1          # Output data dimension
        self.layer_num_list = [2,5]            # Number of model layers, minimum is 1, must be integer
        self.hidden_size_list = [20,50,100,300]          #range(1, 10+1, 1)  # Parameter dimension for each NN layer

        # Training phase parameters
        self.learning_rate = 0.01     # Learning rate for model training
        self.num_epochs = 20001         # Number of iterations for model training
        self.path = ['./model_x1','./model_x2']

args = Argument()
if args.cuda_use:
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")



# Outer loop for multiple experimental runs with different coefficients
for i in range(2):
    # Extract input dimension from hyperparameters
    input_size = args.input_size[0]

    # Initialize data generator and create dataset
    # Generates synthetic data based on mathematical function and parameters
    generator = dataGenerator(args.func[0], args.start[0], args.end[0], 
                             args.step, args.num_points, args.coff[i])
    x, y = generator.generate_data()
    
    # Set model storage path for current experiment
    path = args.path[i]
    print(f"Input data dimensions: {x.size()}")
    
    # Shuffle dataset and split into training/test sets (80/20 split)
    data_index = torch.randperm(torch.tensor(x.size()[0]))
    train_data_size = torch.round(torch.tensor(x.size()[0] * 0.8)).int()

    # Reorder data using shuffled indices
    x = x[data_index]
    y = y[data_index]
    
    print(f"Training samples: {train_data_size}, Total data shape: {x.size()}")
    
    # Move data to appropriate device (CPU/GPU)
    train_data = x[:train_data_size].to(device)
    train_labels = y[:train_data_size].to(device)
    test_data = x[train_data_size:].to(device)
    test_labels = y[train_data_size:].to(device)

    # Hyperparameter grid search over network architectures
    for layer_num in args.layer_num_list:
        # Evaluate different hidden layer sizes for current layer count
        for hidden_size in args.hidden_size_list:
            # Initialize neural network with current architecture
            model = NeuralNetwork(input_size, hidden_size, args.output_size, layer_num).to(device)
            
            # Configure Adam optimizer with specified learning rate
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            
            # Use Mean Squared Error as loss function for regression task
            criterion = nn.MSELoss()
            
            # Initialize training module and execute training procedure
            train = train_model(args.num_epochs)
            losses = train.train(model, optimizer, criterion, train_data, train_labels, 
                               test_data, test_labels, layer_num, hidden_size, path)
            
            # Log performance metrics for current configuration
            print(f'Layer num: {layer_num}, Hidden size: {hidden_size}, Final loss: {losses[-1]:.6f}')

        # Optional: Save hyperparameters and loss trajectories
        # save('results_0627_sin5.txt', losses, hidden_size, layer_num, args.func[i+1], args)

# Visualization code (commented out) for plotting loss vs hidden size
# plot_figure(x=args.hidden_size_list, y=losses, xlabel='hidden_size', ylabel='loss', title='Model Performance vs Hidden Layer Size')