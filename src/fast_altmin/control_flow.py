import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim

import fast_altmin
import unittest

#These two functions need to be moved to control flow
def conv_model_to_class(model, criterion, batch_size, n_iter_weights=5):
    if isinstance(criterion, nn.BCELoss):
        neural_network = fast_altmin.NeuralNetwork(fast_altmin.NeuralNetwork.BCELoss, n_iter_weights, batch_size, model[0].weight.size(1))
    elif isinstance(criterion, nn.MSELoss):
        neural_network = fast_altmin.NeuralNetwork(fast_altmin.NeuralNetwork.MSELoss, n_iter_weights, batch_size, model[0].weight.size(1))
    elif isinstance(criterion, nn.CrossEntropyLoss):
        neural_network = fast_altmin.NeuralNetwork(fast_altmin.NeuralNetwork.CrossEntropyLoss, n_iter_weights, batch_size, model[0].weight.size(1))
    else:
        print("loss not imp yet")

    create_model_class(model, neural_network, batch_size, 0)
    neural_network.construct_pairs()
    return neural_network


def create_model_class(model, neural_network, batch_size, n,  has_codes = True):
    #print(model)
    for mod in model:
        #print(mod)
        if isinstance(mod, nn.ReLU):
            neural_network.push_back_non_lin_layer(fast_altmin.Layer.relu, batch_size, n)
        elif isinstance(mod, nn.Linear):
            n = mod.weight.size(0)
            neural_network.push_back_lin_layer(fast_altmin.Layer.linear, batch_size, n, mod.weight.size(1), mod.weight.data, mod.bias.data, has_codes)
        elif isinstance(mod, nn.Sigmoid):
            neural_network.push_back_non_lin_layer(fast_altmin.Layer.sigmoid, batch_size, n)
        elif isinstance(mod, nn.Sequential):
            layer = create_model_class(mod, neural_network, batch_size, n, False)
        else:
            print("layer not imp yet")
    


