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



# Hyperparameter configuration class for model comparison experiments
class Argument:
    def __init__(self):
        # Hardware configuration
        self.cuda_use = True
        
        # Data generation parameters
        self.name = "newlinear"
        self.func = ["0*x"]  # Base function template
        # Function fitting parameters
        self.start = [-1, -1]        # Data generation intervals
        self.end = [1, 1]            # Data generation intervals
        self.num_points = 201        # Number of data points to generate
        self.coff = [[5]]            # Coefficient sets for linear functions

        # Neural network architecture parameters
        self.input_size = [3]        # Input feature dimension
        self.output_size = 1         # Output prediction dimension
        self.layer_num_list = [2,5]  # Number of hidden layers to experiment with
        self.hidden_size_list = [20,50,100,300]  # Hidden layer neuron counts

        # Training configuration
        self.learning_rate = 0.001   # Optimization learning rate
        self.num_epochs = 20001      # Total training epochs
        # Model storage paths for different training strategies
        self.path = ['func3_pretrain2/model_x1','func3_pretrain2/model_x2', 'func3_pretrain2/model_x3',
                    'func3_pretrain2/model_x4','func3_pretrain2/model_x5','func3_pretrain2/model_x6']

# Experimental strategy explanations:
# x1: Pretrain on 5x+5y+5z then fine-tune on 10x+10y+10z
# x2: Direct training on 10x+10y+10z
# x3: Additional function: 10x+10y+10z
# x4: Additional function: 5x+5y+5z
# x5: Compare f=sin(x)+sin(y) with f-g where g=1/pi*x+1/pi*y

# Initialize hyperparameter configuration
args = Argument()

# Device configuration for computation
if args.cuda_use:
    # Check GPU availability and set appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Training round identifier
round = 0

def adjust_model(model, x, y):
    """
    Adjust model output scale to match target distribution
    
    Args:
        model: Neural network model to adjust
        x: Input features for scale calibration
        y: Target values for scale reference
        
    Returns:
        model: Model with rescaled parameters
    """
    # Calculate value ranges for target and predictions
    max_true_y = torch.max(y)
    min_true_y = torch.min(y)
    pred = model(x)
    max_pred_y = torch.max(pred)
    min_pred_y = torch.min(pred)
    print("max", max_true_y, max_pred_y)

    # Compute scaling ratio based on value ranges
    ratio = (max_true_y - min_true_y) / (max_pred_y - min_pred_y)
    
    # Apply scaling to all model parameters
    for param in model.parameters():
        param.data = param.data * ratio

    return model

# Extract network architecture parameters
input_size = args.input_size[0]
coffs = args.coff[0]
i = 1
i = i - 1

# Data generation and preparation loop
for coff in coffs:
    # Initialize coefficients for linear function
    coff = [coff for i in range(input_size)]
    print(input_size, coff)
    
    # Generate training data with specified coefficients
    generator = dataGenerator(args.func[0], args.start[0], args.end[0], args.num_points, coff)
    x, y = generator.generate_data()

    # Create scaled version of target values
    y2 = y.detach().clone() * 2
    y2 = y2.to(device)
    
    # Print data characteristics for verification
    print(x)
    print(y.reshape([-1]))
    print(y2.reshape([-1]))

    # Dataset shuffling and preparation
    data_index = torch.randperm(torch.tensor(x.size()[0]))
    train_data_size = torch.round(torch.tensor(x.size()[0] * 1.0))
    train_data_size = train_data_size.int()

    x = x[data_index]
    y = y[data_index]
    print(train_data_size, x.size())
    
    # Move data to computation device
    x = x.to(device)
    y = y.to(device)
    
    # Note: Residual computation setup (commented out)
    #g = copy.deepcopy(y.detch().clone()).to(device)
    #g = g * 0

# Fixed architecture parameters for model comparison
hidden_size = 20
layer_num = 2
epoch = 20000
path = 'func3_pretrain2'

# Load pre-trained model trained on 5x+5y+5z function
path_5x = 'model_x4'
model1 = NeuralNetwork(input_size, hidden_size, args.output_size, layer_num).to(device)
# Construct model checkpoint filename
file_name = "{}/{}/model_layer{}_hidden{}_epoch{}_round{}.pt".format(path, path_5x, layer_num, hidden_size, epoch, round)
# Load pre-trained model weights from disk
model1.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))

# Load pre-trained model trained on 10x+10y+10z function
path_5x = 'model_x2'
model2 = NeuralNetwork(input_size, hidden_size, args.output_size, layer_num).to(device)
# Construct model checkpoint filename
file_name = "{}/{}/model_layer{}_hidden{}_epoch{}_round{}.pt".format(path, path_5x, layer_num, hidden_size, epoch, round)
# Load pre-trained model weights from disk
model2.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))

# Initialize untrained model for baseline comparison
model3 = NeuralNetwork(input_size, hidden_size, args.output_size, layer_num).to(device)

# Compute parameter distance between models trained on 5x and 10x functions
print("the distane of 5x and 10x")
for p1, p2 in zip(model1.parameters(), model2.parameters()):
    # Calculate L1 distance between corresponding parameter tensors
    print(torch.sum(torch.abs(p1.data - p2.data)))

# Compute parameter distance between untrained model and 10x-trained model
print("the distane of 0 and 10x")
for p1, p2 in zip(model3.parameters(), model2.parameters()):
    # Calculate L1 distance between corresponding parameter tensors
    print(torch.sum(torch.abs(p1.data - p2.data)))