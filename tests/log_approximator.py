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



class LogApproximator(nn.Module):
    def __init__(self):
        super(LogApproximator, self).__init__()

        self.features = ()
        self.classifier = nn.Sequential()
        self.classifier.add_module("-1", nn.Linear(1,100))
        self.classifier.add_module("0", nn.ReLU())
        self.classifier.add_module("1",nn.Linear(100,100)) 
        self.classifier.add_module("2",nn.ReLU()) 
        self.classifier.add_module("3",nn.Linear(100,100)) 
        self.classifier.add_module("4",nn.ReLU()) 
        self.classifier.add_module("5",nn.Linear(100,1)) 
        self.classifier.double()
    def forward(self, x):
        x = self.classifier(x)
        return x

            


def test_set(model):
    X = np.arange(10.0,12.0,0.1)
    X = X.reshape(20,1)
    Y = np.log(X)
    outputs = model(torch.tensor(np.array(X)).double())
    res = np.square(np.subtract(outputs.detach(), Y)).mean()
    return res

def test_nn(network):
    X = np.arange(10.0,12.0,0.1)
    X = X.reshape(20,1)
    Y = np.log(X)
    outputs = network.get_codes(torch.tensor(np.array(X)).double(),False)
    res = np.square(np.subtract(outputs, Y)).mean()
    return res




# def sgd(model, X, Y, epochs):
#     losses = []
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     for epoch in range(epochs):

#         for index in range(0,99):
#             data = torch.tensor(np.array(X[index])).double()
#             targets = torch.tensor(np.array(Y[index])).double()
#             targets.reshape(1,1)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output,targets)
#             #print("loss "+str(loss))
#             loss.backward()
#             optimizer.step()
#         loss = test_set(model).item() 
#         losses.append(loss)
#         print("Epoch: " + str(epoch) + "res "+ str(loss))

#     outputs = model(torch.tensor(np.array(X)).double())
#     return losses, outputs

def run_altmin(model, X, Y, epochs):
    losses= [] 
    for epoch in range(epochs):
        for index in range(0,99):
            data = torch.tensor(np.array(X[index])).double()
            targets = torch.tensor(np.array(Y[index])).double()
            targets = targets.reshape(1,1)
            data = data.reshape(1,1)
            model.train()
            with torch.no_grad():
                outputs, codes = get_codes(model, data)

            
            codes = update_codes(codes, model, targets, criterion, 0.003, lambda_c=0, n_iter=1, lr=0.001)

            update_last_layer_(model[-1], codes[-1], targets, criterion, n_iter=1)

            update_hidden_weights_adam_(model, data, codes, lambda_w=0, n_iter=1)
        loss = test_set(model).item() 
        losses.append(loss)
    outputs = model(torch.tensor(np.array(X)).double())
    return losses, outputs

def run_cpp_altmin(neural_network, X, Y, epochs):
    losses= [] 
    for epoch in range(epochs):
        for index in range(0,99):
            data = torch.tensor(np.array(X[index])).double()
            targets = torch.tensor(np.array(Y[index])).double()
            targets = targets.reshape(1,1)
            data = data.reshape(1,1)
            _ = torch.from_numpy(neural_network.get_codes(data, True))
            neural_network.update_codes(targets)
            neural_network.update_weights(data,targets)
        loss = test_nn(neural_network).item() 
        losses.append(loss)
    outputs = neural_network.get_codes(torch.tensor(np.array(X)).double(),False)
    return losses, outputs

def show_res(X, Y, outputs_altmin, outputs_cpp, loss_altmin, loss_cpp):
    plt.figure(figsize = (10,6))
    plt.plot(X, Y, linewidth=2, marker='.', label="log")
    #plt.plot(X, outputs_sgd.detach(), linewidth=2, marker='.', label="sgd")
    plt.plot(X, outputs_altmin.detach(), linewidth=2, marker='.', label="altmin")
    plt.plot(X, outputs_cpp, linewidth=2, marker='.', label="altmin cpp")
    plt.legend()
    plt.show()

    plt.figure(figsize = (10,6))
    #plt.plot(loss_sgd, label="sgd")
    plt.plot(loss_altmin, label="altmin")
    plt.plot(loss_cpp, label="cpp")
    plt.legend()
    plt.show()


criterion = nn.MSELoss()
X = np.arange(0.0,10.0,0.1)
#Avoid divide by 0
X = X[1:]
X = X.reshape(99,1)
Y = np.log(X)
epochs = 150

# model = LogApproximator()
# model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.001})
# model[-1].optimizer.param_groups[0]['lr'] = 0.001
# model = model[1:]
# start = time.time()
# loss_sgd, outputs_sgd = sgd(model, X, Y, epochs)
# end = time.time()
# time_sgd = end-start
# print("Time for sgd :" + str(time_sgd))

model = LogApproximator()
model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.001})
model[-1].optimizer.param_groups[0]['lr'] = 0.001
model = model[1:]
start = time.time()
loss_altmin, outputs_altmin = run_altmin(model, X, Y, epochs)
end = time.time()
time_altmin = end-start 
print("Time for altmin :" + str(time_altmin))

model = LogApproximator()
model = get_mods(model)
model = model[1:]
neural_network = conv_model_to_class(model, nn.MSELoss(),  1, 1, 0.001)
start = time.time()
loss_cpp, outputs_cpp = run_cpp_altmin(neural_network, X, Y, epochs)
end = time.time()
time_cpp = end-start
print("Time for cpp :" + str(time_cpp))

show_res(X, Y, outputs_altmin, outputs_cpp, loss_altmin, loss_cpp)
