import torch
import torch.nn as nn
import torch.optim as optim
import copy


from data_generation import dataGenerator
from NNnetwork import NeuralNetwork
from train_class import train_model


torch.manual_seed(1)



# hyper-parameter
class Argument:
    def __init__(self):
        self.cuda_use = True

        self.name = "newlinear"
        self.func = ["0*x"]

        self.start = [-3.1415926, -1]
        self.end = [3.1415916, 1]
        self.num_points = 201
        self.coff = [[1/3.1415926]]

        self.input_size = [3]
        self.output_size = 1
        self.layer_num_list = [2,5]
        self.hidden_size_list = [20,50,100,300]

        self.learning_rate = 0.001
        self.num_epochs = 20001
        self.path = ['model_x5','model_x6', 'model_x3','model_x4','model_x5','model_x6']

args = Argument()
if args.cuda_use:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


round = 0

def adjust_model(model, x, y):
    max_true_y = torch.max(y)
    min_true_y = torch.min(y)
    pred = model(x)
    max_pred_y = torch.max(pred)
    min_pred_y = torch.min(pred)
    print("max", max_true_y, max_pred_y)

    ratio = (max_true_y - min_true_y) / (max_pred_y - min_pred_y)
    for param in model.parameters():
        param.data = param.data *ratio

    return model

input_size = args.input_size[0]
coffs = args.coff[0]
i = 1


for coff in coffs:
    coff = [coff for i in range(input_size)]
     # generate data
    generator = dataGenerator(args.func[0], args.start[0], args.end[0], args.num_points, coff)
    x, y, g = generator.generate_data()

    data_index = torch.randperm(torch.tensor(x.size()[0]))
    train_data_size = torch.round(torch.tensor(x.size()[0]*1.0))
    train_data_size = train_data_size.int()

    x = x[data_index]
    y = y[data_index]
    print(train_data_size, x.size())
    x = x.to(device)
    y = y.to(device)
    g = copy.deepcopy(y.detach().clone()).to(device)
    if i==1:
        g = g * 0
    
    train_data = x[:train_data_size].to(device)
    train_labels = y[:train_data_size].to(device)
    test_data = x[train_data_size:].to(device)
    test_labels = y[train_data_size:].to(device)

    
    for layer_num in args.layer_num_list:
        for hidden_size in args.hidden_size_list:
            # Define model, optimizer, loss function
            model = NeuralNetwork(input_size, hidden_size, args.output_size, layer_num).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.MSELoss()
            train = train_model(args.num_epochs)
            losses = train.train(model, optimizer, criterion, x, y, g, test_labels, layer_num, hidden_size, args.path[i], round)
            print('Layer num is {}, hidden_size is {}, loss is {}'.format(layer_num, hidden_size, losses[-1]))
