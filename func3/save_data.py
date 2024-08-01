

def save(file_name, para, hidden_size, layer_num, func_name, args):
    with open(file_name, 'a+') as file:
        file.write('\n')
        file.write("# Hyperparameters: func is {}, layer num is {}, hidden_size is {}, interval is [{}, {}], step is {}, lr is {}, epoch is {}".format(
            func_name, layer_num, hidden_size, args.start, args.end, args.step, args.learning_rate, args.num_epochs
        ))
        file.write('\n')
        file.write(str(para) + '\n')

