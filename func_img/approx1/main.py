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

# Training strategy: First train on y/2, then fine-tune on full y

# Hyperparameter configuration class for image approximation experiments
class Argument:
    def __init__(self):
        # Hardware configuration
        self.cuda_use = True

        # Data generation parameters
        self.sample = 1  # Sampling method: 1 for random sampling, 2 for sequential interval sampling

        # Neural network architecture parameters
        self.input_size = 2  # Input feature dimension (pixel coordinates)
        self.output_size = 1  # Output dimension (pixel intensity)
        self.layer_num_list = [5]  # Number of hidden layers in network
        self.hidden_size_list = [20, 50, 100, 300]  # Hidden layer neuron counts

        # Training configuration
        self.learning_rate = 0.0005  # Optimization learning rate
        self.num_epochs = 200001  # Total training epochs
        # Model storage paths for different experimental conditions
        self.path = ['func_img/approx1/model_x1', 'func_img/model_x2', 'func_img/model_x3',
                     'func_img/model_x07281', 'func_linear/model_x07282', 'func_linear/model_x07284',
                     'func_linear/model_x07288']


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


# Training round identifier
round = 0

# Main training loop
for i in range(1):
    # Generate and process image data
    generator = dataGenerator()
    img_data = generator.generate_data()
    x, y = data_process(img_data)

    # Data normalization and device transfer
    x = x * 1.
    y = y * 1.
    x = x.to(device)
    y = y.to(device) / 255  # Normalize pixel values to [0,1] range
    x_raw = x  # Store original full dataset
    y_raw = y

    # Apply sampling strategy to reduce dataset size
    if args.sample == 1:
        # Random sampling: select 5% of data points randomly
        data_size = y_raw.size()[0]
        sample_index = torch.randperm(data_size)[:int(torch.floor(torch.tensor(data_size / 20)))]
        x = x[sample_index, :]
        y = y[sample_index, :]
    else:
        # Sequential interval sampling: select every 20th data point
        x = x[::20, :]
        y = y[::20, :]

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

            # Execute two-phase training: first on y/2, then on full y
            losses = train.train(model, optimizer, criterion, x, y / 2, x, y, layer_num, hidden_size, path, round)
            print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))

            # Generate and save output image using trained model
            pred_img = model(x_raw) * 255  # Scale predictions back to [0,255] range
            pred_img = (torch.round(pred_img.data)).int().reshape([-1]).tolist()
            img_name = 'func_img/approx1/results/ln{}_hs{}_lr{}.png'.format(layer_num, hidden_size, args.learning_rate)
            a = generator.out_img(img_name, pred_img)


