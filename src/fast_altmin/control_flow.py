import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim

from altmin import Flatten, get_mods
import fast_altmin
import unittest

def create_model_class(model, neural_network, batch_size, n, lr=0.008, lr_codes = 0.3,  last_layer = False):
    for mod in model:
        #print(mod)
        if isinstance(mod, nn.ReLU):
            neural_network.addReluLayer(n, batch_size)
        elif isinstance(mod, nn.Linear):
            n = mod.weight.size(0)
            if last_layer:
                neural_network.addLastLinearLayer(batch_size, mod.weight.data, mod.bias.data, lr)
            else:
                neural_network.addLinearLayer(batch_size, mod.weight.data, mod.bias.data, lr, lr_codes)
        elif isinstance(mod, nn.Sigmoid):
            neural_network.addSigmoidLayer(n, batch_size)
        elif isinstance(mod, nn.Sequential):
            create_model_class(mod, neural_network, batch_size, n,  lr, lr_codes, True)
        else:
            print("layer not imp yet")


def create_CNN(model, neural_network, N, C, H, W):
    for mod in model:
        #print(mod)
        if isinstance(mod, nn.ReLU):
            neural_network.addReluCNNLayer(N,C,H,W)
        elif isinstance(mod, nn.Conv2d):
            kernel = mod.weight.data
            kernel_list = [] 
            C_out = kernel.shape[0]
            for c_out in range(C_out):
                tmp = [] 
                for c_in in range(C):
                    tmp.append(kernel[c_out,c_in].numpy())
                kernel_list.append(tmp)

            bias = mod.bias.data 
            H =(int) (1 + ((H+2*mod.padding[0] - mod.dilation[0] *(mod.kernel_size[0]-1)-1)/mod.stride[0]))
            W = (int) (1 + ((W+2*mod.padding[1] - mod.dilation[1] *(mod.kernel_size[1]-1)-1)/mod.stride[1]))
            neural_network.addConv2dLayer(kernel_list, bias, N, C, H, W)
            C = C_out
        elif isinstance(mod, nn.MaxPool2d):
            H = (int) (1 + ((H+2*mod.padding - mod.dilation *(mod.kernel_size-1)-1)/mod.stride))
            W = (int) (1 + ((W+2*mod.padding - mod.dilation *(mod.kernel_size-1)-1)/mod.stride))
            neural_network.addMaxPool2dLayer(mod.kernel_size, mod.stride, N, C, H, W)
        elif isinstance(mod, nn.Sequential):
            C, H, W = create_CNN(mod, neural_network, N, C, H, W)
        else:
            if not isinstance(mod, Flatten):
                print("layer not imp yet")

    return C,H,W



def convert_feed_forward_pytorch_model(model, criterion, batch_size, lr_weights=0.008, lr_codes=0.3, file_path = "-1"):
    model = model.double()
    model = get_mods(model)
    model = model[1:]
    if file_path != "-1":
        model.load_state_dict(torch.load(file_path))

    if isinstance(criterion, nn.BCELoss):
        neural_network = fast_altmin.VariantNeuralNetworkBCE()
        fast_altmin.create_model_class(model, neural_network, batch_size, 0, lr_weights, lr_codes)
        neural_network.construct_pairs()
        return neural_network
    elif isinstance(criterion, nn.MSELoss):
        neural_network = fast_altmin.VariantNeuralNetworkMSE()
        fast_altmin.create_model_class(model, neural_network, batch_size, 0, lr_weights, lr_codes)
        neural_network.construct_pairs()
        return neural_network
    elif isinstance(criterion, nn.CrossEntropyLoss):
        neural_network = fast_altmin.VariantNeuralNetworkCrossEntropy()
        fast_altmin.create_model_class(model, neural_network, batch_size, 0, lr_weights, lr_codes)
        neural_network.construct_pairs()
        return neural_network
    else:
        print("Invalid criterion: Please choose either BCELoss, MSELoss or CrossEntropyLoss")

    


def convert_cnn_pytorch_model(model, criterion, batch_size, C_in, height, width):
    model = get_mods(model)
    x = 0
    for m in model:
        if isinstance(m, nn.Linear):
            break
        x+=1

    model_cnn = model[0:x]
    model_nn = model[x:]

    convolutional_neural_network = fast_altmin.VariantCNNCrossEntropy()
    
    C_out, H_out, W_out  = fast_altmin.create_CNN(model_cnn, convolutional_neural_network, batch_size, C_in, height, width)
    neural_network = fast_altmin.VariantNeuralNetworkCrossEntropy()
    fast_altmin.create_model_class(model_nn, neural_network, batch_size, 0)
    neural_network.construct_pairs()
    matrix_size = model_cnn[-1][1].kernel_size*2
    lenet = fast_altmin.LeNetCrossEntropy(batch_size, C_out, matrix_size)
    lenet.AddCNN(convolutional_neural_network)
    lenet.AddFeedForwardNN(neural_network)
    return lenet
