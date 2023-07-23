import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim

from altmin import Flatten
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



    