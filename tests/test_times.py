import torch
import numpy as np

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import fast_altmin
from fast_altmin import conv_model_to_class
from altmin import simpleNN, get_mods, get_codes, update_codes, update_hidden_weights_adam_, update_last_layer_
import time

from log_approximator import *

def sgd_forward(model, X, Y, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    time_sgd_foward = 0.0
    time_sgd_backwards = 0.0
    
    for epoch in range(epochs):
        for index in range(0,99):
            data = torch.tensor(np.array(X[index])).double()
            targets = torch.tensor(np.array(Y[index])).double()
            targets.reshape(1,1)
            optimizer.zero_grad()
            start = time.time()    
            output = model(data)
            end = time.time()
            time_sgd_foward += (end-start)
            start = time.time()    
            loss = nn.MSELoss()(output,targets)
            #print("loss "+str(loss))
            loss.backward()
            optimizer.step()
            end = time.time() 
            time_sgd_backwards += (end-start)
    
    print("time for sgd forward: "+str(time_sgd_foward))
    print("time for sgd backwards: "+str(time_sgd_backwards))
    print(" ")

def altmin_forward(model, X, Y, epochs):
    time_altmin_forward = 0.0
    time_altmin_codes = 0.0
    time_altmin_weights = 0.0

    for epoch in range(epochs):
        for index in range(0,99):
            data = torch.tensor(np.array(X[index])).double()
            targets = torch.tensor(np.array(Y[index])).double()
            targets = targets.reshape(1,1)
            data = data.reshape(1,1)
            model.train()
            start = time.time()    
            with torch.no_grad():
                outputs, codes = get_codes(model, data)
            end = time.time()
            time_altmin_forward += (end-start)

            start = time.time()
            codes = update_codes(codes, model, targets, nn.MSELoss(), 0.003, lambda_c=0, n_iter=1, lr=0.001)
            end = time.time()
            time_altmin_codes += (end-start)

            start = time.time()
            update_last_layer_(model[-1], codes[-1], targets, nn.MSELoss(), n_iter=1)
            update_hidden_weights_adam_(model, data, codes, lambda_w=0, n_iter=1)
            end = time.time()
            time_altmin_weights += (end-start)
    
    time_altmin_forward = end - start 
    print("time for altmin forward: "+str(time_altmin_forward))
    print("time for altmin codes: "+str(time_altmin_codes))
    print("time for altmin weights: "+str(time_altmin_weights))
    print("time for altmin backwards: "+str(time_altmin_codes+time_altmin_weights))
    print(" ")
            
            
def cpp_forward(neural_network, X, Y, epochs):
    time_cpp_forward = 0.0
    time_cpp_codes = 0.0
    time_cpp_weights = 0.0
    
    for epoch in range(epochs):
        for index in range(0,99):
            data = torch.tensor(np.array(X[index])).double()
            targets = torch.tensor(np.array(Y[index])).double()
            targets = targets.reshape(1,1)
            data = data.reshape(1,1)
            start=time.time()
            neural_network.get_codes(data, True)
            end = time.time()
            time_cpp_forward+= (end-start)

            start=time.time()
            neural_network.update_codes(targets)
            end = time.time()
            time_cpp_codes += (end-start)

            start=time.time()
            neural_network.update_weights(data,targets)
            end = time.time()
            time_cpp_weights += (end-start)

     
    print("time for cpp forward: "+str(time_cpp_forward))
    print("time for cpp codes: "+str(time_cpp_codes))
    print("time for cpp weights: "+str(time_cpp_weights))
    print("time for cpp backwards: "+str(time_cpp_codes+time_cpp_weights))
            

def test_log_approx():
    criterion = nn.MSELoss()
    X = np.arange(0.0,10.0,0.1)
    #Avoid divide by 0
    X = X[1:]
    X = X.reshape(99,1)
    Y = np.log(X)
    epochs = 100

    model = LogApproximator(100)
    model = get_mods(model)
    model = model[1:]
    sgd_forward(model, X, Y, epochs)


    model = LogApproximator(100)
    model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.001})
    model[-1].optimizer.param_groups[0]['lr'] = 0.001
    model = model[1:]
    altmin_forward(model, X, Y, epochs)


    model = LogApproximator(100)
    model = get_mods(model)
    model = model[1:]
    neural_network = conv_model_to_class(model, nn.MSELoss(),  1, 1, 1, 0.001, 0.003, 0.9, 0.001)
    cpp_forward(neural_network, X, Y, epochs)

test_log_approx()
