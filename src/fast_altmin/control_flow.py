import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim


import fast_altmin
import unittest

def create_model_class(model, neural_network, batch_size, n, last_layer = False, lr=0.008):
    for mod in model:
        #print(mod)
        if isinstance(mod, nn.ReLU):
            neural_network.addReluLayer(n, batch_size)
        elif isinstance(mod, nn.Linear):
            n = mod.weight.size(0)
            if last_layer:
                neural_network.addLastLinearLayer(batch_size, mod.weight.data, mod.bias.data, lr)
            else:
                neural_network.addLinearLayer(batch_size, mod.weight.data, mod.bias.data, lr)
        elif isinstance(mod, nn.Sigmoid):
            neural_network.addSigmoidLayer(n, batch_size)
        elif isinstance(mod, nn.Sequential):
            create_model_class(mod, neural_network, batch_size, n, True, lr)
        else:
            print("layer not imp yet")

    #print("J")
    