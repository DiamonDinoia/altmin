import torch.nn as nn
import torch as torch
import unittest
import derivatives
import sys
from altmin import simpleNN, get_mods, update_last_layer_, update_hidden_weights_adam_, update_codes, FFNet, get_codes, load_dataset, store_momentums
import pickle 
import fast_altmin
from fast_altmin import conv_model_to_class

from log_approximator import LogApproximator

def check_equal(first_imp, second_imp, eps):
    for x in range(len(first_imp)):
        for y in range(len(first_imp[x])):
            assert(abs(first_imp[x][y] - second_imp[x]
                        [y]) <= sys.float_info.epsilon*eps)
            
def check_equal_bias(first_imp, second_imp,eps):
    for x in range(len(first_imp)):
        assert(abs(first_imp[x] - second_imp[x]) <= sys.float_info.epsilon*eps)
            
def check_equal_weights_and_bias(model_python, weights, biases, eps = 10e6):
    y = 0
    for x,m in enumerate(model_python):
        if isinstance(m, nn.Linear):
            check_equal(model_python[x].weight.data, weights[y], eps)
            check_equal_bias(model_python[x].bias, biases[y], eps)
            y+=1
            
        if isinstance(m, nn.Sequential):
            check_equal(model_python[x][1].weight.data, weights[y], eps)
            check_equal_bias(model_python[x][1].bias, biases[y], eps)

def initialise_weights_and_biases(model_python, neural_network):
    weights = []
    biases = []
    for x,m in enumerate(model_python):
        if isinstance(m, nn.Linear):
            weights.append(model_python[x].weight.data)
            biases.append(model_python[x].bias.data)
    
        if isinstance(m, nn.Sequential):
            weights.append(model_python[x][1].weight.data)
            biases.append(model_python[x][1].bias.data)

    neural_network.set_weights_and_biases(weights,biases)


class TestLayers(unittest.TestCase):
    def test_lin(self):
        lin = nn.Linear(2, 4).double()
        in_tensor = torch.rand(1, 2, dtype=torch.double)
        weight = lin.weight.data
        bias = lin.bias.data
        #bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))
        cpp_imp = torch.from_numpy(fast_altmin.lin(in_tensor,
                                      weight, bias))
        python_imp = lin(in_tensor)
        check_equal(cpp_imp, python_imp, 10e8)

#     def test_lin_batch(self):
#         lin = nn.Linear(4, 6).double()
#         in_tensor = torch.rand(5, 4, dtype=torch.double)
#         weight = lin.weight.data
#         #bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))
#         bias = lin.bias.data
#         start = time.time()
#         cpp_imp = torch.from_numpy(fast_altmin.lin(in_tensor,
#                                       weight, bias))
#         end=time.time()
#         print("Cpp lin: " +str(end-start))
#         start = time.time()
#         python_imp = lin(in_tensor)
#         end=time.time()
#         print("py lin: " +str(end-start))
#         check_equal(cpp_imp, python_imp, 10e8)

#     # Functions for relu and sigmoid are in place so they don't need to return a value

#     def test_ReLU(self):
#         relu = nn.ReLU()
#         in_tensor = torch.rand(5, 10, dtype=torch.double)
#         start = time.time()
#         python_imp = relu(in_tensor)
#         # In tensor data updated in place
#         end=time.time()
#         print("py reliu: " +str(end-start))
#         start = time.time()
#         fast_altmin.ReLU(in_tensor)
#         end=time.time()
#         print("cpp relu: " +str(end-start))

#         check_equal(in_tensor, python_imp, 10e8)

#     def test_sigmoid(self):
#         sigmoid = nn.Sigmoid()
#         in_tensor = torch.rand(5, 10, dtype=torch.double)
#         start = time.time()
#         python_imp = sigmoid(in_tensor)
#         # In tensor data changed in place?
#         end=time.time()
#         print("py sig: " +str(end-start))
#         start = time.time()
#         fast_altmin.sigmoid(in_tensor)
#         end=time.time()
#         print("cpp sig: " +str(end-start))

#         check_equal(in_tensor, python_imp, 10e8)

# # Testing the cpp implementation of BCEloss and MSEloss
# # Also test the fuctions ability to calculate loss on batch input
# # Again take reference to tensor as input so no data copied between
# # Can't use epsilon as these diverge slightly around 8 decimal places but no a priority to fix rn but I'll come back to this
# # Again I don't use these functions I just use the torch imp atm but I'll keep them as they might be useful if I can move away from autograd. 
# class TestCriterion(unittest.TestCase):
#     def test_BCELoss(self):
#         targets = torch.round(torch.rand(1, 10, dtype=torch.double))
#         predictions = torch.rand(1, 10, dtype=torch.double)
#         cpp_loss = fast_altmin.BCELoss(predictions, targets)
#         python_loss = nn.BCELoss()(predictions, targets)
#         self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

#     def test_batch_BCELoss(self):
#         targets = torch.round(torch.rand(5000, 5, dtype=torch.double))
#         predictions = torch.rand(5000, 5, dtype=torch.double)
#         start = time.time()
#         cpp_loss = fast_altmin.BCELoss(predictions, targets)
#         end = time.time() 
#         print("BCE Loss: " + str(end-start))
#         start = time.time()
#         python_loss = nn.BCELoss()(predictions, targets)
#         end = time.time() 
#         print("BCE Loss python: " + str(end-start))
#         self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

#     def test_MSELoss(self):
#         targets = torch.rand(1, 10, dtype=torch.double)
#         predictions = torch.rand(1, 10, dtype=torch.double)
#         cpp_loss = fast_altmin.MSELoss(predictions, targets)
#         python_loss = nn.MSELoss()(predictions, targets)
#         self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

#     def test_batch_MSELoss(self):
#         targets = torch.rand(5000, 100, dtype=torch.double)
#         predictions = torch.rand(5000, 100, dtype=torch.double)
#         start = time.time()
#         cpp_loss = fast_altmin.MSELoss(predictions, targets)
#         end = time.time()
#         print("MSE Loss: " + str(end-start))
#         start = time.time()
#         python_loss = nn.functional.mse_loss(
#             predictions, targets)
#         end = time.time()
#         print("MSE Loss pyton: " + str(end-start))
#         self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

#     def test_log_softmax(self):
#         inputs = torch.rand(1,5, dtype=torch.double)
#         python_imp = nn.LogSoftmax(dim=1)(inputs)
#         fast_altmin.log_softmax(inputs)
#         check_equal(inputs, python_imp, 10e8)

#     def test_batch_log_softmax(self):
#         inputs = torch.rand(3,5, dtype=torch.double)
#         python_imp = nn.LogSoftmax(dim=1)(inputs)
#         fast_altmin.log_softmax(inputs)
#         check_equal(inputs, python_imp, 10e8)

#     def test_negative_log_likelihood(self):
#         inputs = torch.rand(3,5, dtype=torch.double)
#         targets = torch.tensor([2,1,4])
#         inputs = nn.LogSoftmax(dim=1)(inputs)
#         python_loss = nn.NLLLoss()(inputs, targets)
#         cpp_loss = fast_altmin.negative_log_likelihood(inputs,targets)
#         self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

#     def test_cross_entropy_loss(self):
#         inputs = torch.rand(5000,5, dtype=torch.double)
#         targets = torch.randint(0, 5, (5000,))
#         start = time.time()
#         python_loss = nn.CrossEntropyLoss()(inputs,targets)
#         end = time.time()
#         print("py cross ent Loss: " + str(end-start))
#         targets = targets.double()
#         start = time.time()
#         cpp_loss = fast_altmin.cross_entropy_loss(inputs,targets)
#         end = time.time()
#         print("cpp cross ent Loss: " + str(end-start))
#         self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)



# class TestDerivatives(unittest.TestCase):
#     def test_relu_derivative(self):
#         input_python = torch.rand(2,5)
#         input_python -=0.5
#         input_python.requires_grad_(True) 
#         input_cpp = input_python.detach().clone()

#         grad_python = derivatives.derivative_relu(input_python) 
        
#         #https://nanobind.readthedocs.io/en/latest/eigen.html
#         #https://pytorch.org/docs/stable/generated/torch.from_numpy.html
#         #I think this means there is no copy.
#         grad_cpp = torch.from_numpy(fast_altmin.differentiate_ReLU(input_cpp))
#         #print(type(grad_cpp))
#         #Correct to 8 d.p to need to improve but works for now 
#         check_equal(grad_python,grad_cpp, 10e9) 

#     def test_sigmoid_derivative(self):
#         input_python = torch.rand(2,5).requires_grad_(True) 
#         input_cpp = input_python.detach().clone()

#         grad_python = derivatives.derivative_sigmoid(input_python) 
#         grad_cpp = torch.from_numpy(fast_altmin.differentiate_sigmoid(input_cpp))
#         check_equal(grad_python,grad_cpp, 10e9) 

#     def test_lin_derivative(self):
#         input_python = torch.rand(2,5).requires_grad_(True) 
#         input_cpp = input_python.detach().clone()

#         w_grad_python, b_grad_python = derivatives.derivative_linear(input_python) 
#         #print(w_grad_python)
#         #print(b_grad_python)
#         #grads_cpp = torch.from_numpy(fast_altmin.differentiate_linear_layer(input_cpp))
#         #print(grads_cpp)
#         #check_equal(grad_python,grad_cpp, 10e9) 

#     def test_MSELoss_derivative(self):
#         output_python = torch.rand(2,5).requires_grad_(True) 
#         target = torch.rand(2,5)
#         output_cpp = output_python.detach().clone()

#         grad_python = derivatives.derivative_MSELoss(output_python,target)

#         grad_cpp = torch.from_numpy(fast_altmin.differentiate_MSELoss(output_cpp,target))
#         #print(grad_python)
#         #print(grad_cpp)
#         check_equal(grad_python,grad_cpp, 10e8) 


#     def test_BCELoss_derivative(self):
#         output_python = torch.rand(2,5).requires_grad_(True) 
#         target = torch.round(torch.rand(2,5))
#         output_cpp = output_python.detach().clone()

#         grad_python = derivatives.derivative_BCELoss(output_python,target)
#         grad_cpp = torch.from_numpy(fast_altmin.differentiate_BCELoss(output_cpp,target))
#         check_equal(grad_python,grad_cpp, 10e9) 

#     def test_CrossEntropyLoss_derivative(self):
#         output_python = torch.rand(4,5).requires_grad_(True) 
#         target = torch.tensor([2,1,4,0])
#         num_classes = 5
#         output_cpp = output_python.detach().clone()
#         start = time.time()
#         loss = nn.CrossEntropyLoss()(output_python,target)
#         loss.backward()
#         end = time.time()
#         print("Py Diff cross ent "+str(end-start))
#         grad_python = output_python.grad
#         start = time.time()
#         grad_cpp = torch.from_numpy(fast_altmin.differentiate_CrossEntropyLoss(output_cpp,target, num_classes))
#         end = time.time()
#         print("cpp Diff cross ent "+str(end-start))
#         #grad_cpp = derivatives.derivative_CrossEntropyLoss(output_cpp,target)
#         check_equal(grad_python,grad_cpp, 10e9) 



# class TestUpdateFunctions(unittest.TestCase):

    

#     def test_forward(self):
       
#         in_tensor = torch.rand(4,2,dtype = torch.double)
#         model = simpleNN(2, [4,3],1)
#         model = get_mods(model)
#         model = model[1:]

#         neural_network = conv_model_to_class(model, nn.BCELoss(), in_tensor.size(0))
#         output_python, codes_python = get_codes(model, in_tensor)
#         output_cpp = neural_network.get_codes(in_tensor, True)
#         codes_cpp = neural_network.return_codes() 

#         #Check output 
#         check_equal(output_python, output_cpp, 10e6)
#         #check codes
#         for x in range(len(codes_python)):
#             check_equal(codes_python[x], codes_cpp[x], 10e6)


#     def test_update_codes_BCELoss(self):
#         print("Codes:  ")
#         # Setup 
#         targets = torch.round(torch.rand(6, 1, dtype=torch.double))
#         code_one = torch.rand(6,4, dtype=torch.double) - 0.5
#         code_two = torch.rand(6,3, dtype=torch.double) - 0.5
#         codes_cpp = [code_one, code_two]
#         codes_python = [code_one.detach().clone(), code_two.detach().clone()]
#         n_iter = 1
#         lr = 0.3
#         mu = 0.003
#         criterion = nn.BCELoss()

#         # Model setup
#         model_python = simpleNN(2, [4,3],1)
#         model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
#                      scheduler=lambda epoch: 1/2**(epoch//8))
#         model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
#         model_python = model_python[1:]

#         print(targets.shape[0])
#         neural_network = conv_model_to_class(model_python, criterion, targets.shape[0])
#         neural_network.set_codes(codes_cpp)

#         # Python
#         for it in range(1):
#             update_codes(codes_python, model_python, targets, criterion, mu, 0, n_iter, lr)

#         #cpp
#         for it in range(1):     
#             neural_network.update_codes(targets)
#         codes_cpp = neural_network.return_codes() 

#         #Assert codes are equal
#         for x in range(len(codes_cpp)):
#             check_equal(codes_cpp[x], codes_python[x], 10e6)

#     def test_update_codes_CrossEntropyLoss(self):
#          # Setup 
#         targets = torch.randint(0, 4, (4,))
#         code_one = torch.rand(4,100, dtype=torch.double) - 0.5
#         code_two = torch.rand(4,100, dtype=torch.double) - 0.5
#         codes_cpp = [code_one, code_two]
#         codes_python = [code_one.detach().clone(), code_two.detach().clone()]
#         n_iter = 1
#         lr = 0.3
#         mu = 0.003


#         # Model setup
#         model_python = FFNet(10, n_hiddens=100, n_hidden_layers=2, batchnorm=False, bias=True).double()
#         model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
#                      scheduler=lambda epoch: 1/2**(epoch//8))
#         model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
#         model_python = model_python[1:]
        
#         neural_network = conv_model_to_class(model_python, nn.CrossEntropyLoss(), targets.shape[0])
#         neural_network.set_codes(codes_cpp)
        
#         #python
#         for it in range(1):
#             update_codes(codes_python, model_python, targets, nn.CrossEntropyLoss(), mu, 0, n_iter, lr)

#         #cpp
#         targets = targets.double()
#         targets = targets.reshape(1,len(targets))
#         for it in range(1):     
#             neural_network.update_codes(targets)
#         codes_cpp = neural_network.return_codes() 
        
#         # Assert codes are same
#         for x in range(len(codes_cpp)):
#             check_equal(codes_python[x], codes_cpp[x], 10e6)



#     def test_update_codes_MSELoss(self):
#         # Setup 
#         targets = torch.rand(4,1,dtype = torch.double)
#         code_one = torch.rand(4,5, dtype=torch.double) - 0.5
#         code_two = torch.rand(4,5, dtype=torch.double) - 0.5
#         code_three = torch.rand(4,5, dtype=torch.double) - 0.5
#         codes_cpp = [code_one, code_two, code_three]
#         codes_python = [code_one.detach().clone(), code_two.detach().clone(), code_three.detach().clone()]
#         n_iter = 1
#         lr = 0.3
#         mu = 0.003
#         criterion = nn.MSELoss()

#         # Model setup
#         model_python = LogApproximator(5)
#         model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
#                         scheduler=lambda epoch: 1/2**(epoch//8))
#         model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
#         model_python = model_python[1:]
#         neural_network = conv_model_to_class(model_python, criterion, targets.shape[0])
#         neural_network.set_codes(codes_cpp)


#         #python
#         for it in range(5):
#             update_codes(codes_python, model_python, targets, criterion, mu, 0, n_iter, lr)

#         #cpp
#         for it in range(5):     
#             neural_network.update_codes(targets)
#         codes_cpp = neural_network.return_codes() 
        
#         # print(codes_cpp[-1].shape)
#         # Assert codes are same
#         for x in range(len(codes_cpp)):
#             check_equal(codes_python[x], codes_cpp[x], 10e6)



#     def test_update_weights_BCELoss(self):
#         print("\n\nWeights: ")
#         # Setup 
#         inputs = torch.rand(5000,2,dtype = torch.double)
#         targets = torch.round(torch.rand(5000, 1, dtype=torch.double))
#         codes = [torch.rand(5000,4, dtype=torch.double).detach(), torch.rand(5000,3, dtype=torch.double).detach()]
#         n_iter = 1

#         # Model setup
#         model_python = simpleNN(2, [4,3],1)
#         model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
#                         scheduler=lambda epoch: 1/2**(epoch//8))
#         model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
#         model_python = model_python[1:]
#         # cpp model setup
#         neural_network = conv_model_to_class(model_python, nn.BCELoss(), targets.shape[0], 1, n_iter)
#         neural_network.set_codes(codes)
#         initialise_weights_and_biases(model_python, neural_network)

#         # python
#         for it in range(2):
#             update_hidden_weights_adam_(model_python, inputs, codes, lambda_w=0, n_iter=n_iter)
#             update_last_layer_(model_python[-1], codes[-1], targets, nn.BCELoss(), n_iter)

#         # cpp

#         for it in range(2):
#             neural_network.update_weights(inputs, targets)
        
#         weights = neural_network.return_weights()
#         biases = neural_network.return_bias()
        
       
#         # Assert weights and biases updated the same
#         check_equal_weights_and_bias(model_python, weights, biases, 10e9)


#     def test_update_weights_MSELoss(self):
#         # Setup 
#         inputs = torch.rand(4,1,dtype = torch.double)
#         targets = torch.round(torch.rand(4, 1, dtype=torch.double))
#         codes = [torch.rand(4,5, dtype=torch.double).detach(), torch.rand(4,5, dtype=torch.double).detach(), torch.rand(4,5, dtype=torch.double).detach()]
#         n_iter = 1

#         # Model setup
#         model_python = LogApproximator(5)
#         model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.001},
#                         scheduler=lambda epoch: 1/2**(epoch//8))
#         model_python[-1].optimizer.param_groups[0]['lr'] = 0.001
#         model_python = model_python[1:]
        
#         # cpp model setup
#         neural_network = conv_model_to_class(model_python, nn.MSELoss(), targets.shape[0], 1, n_iter, 0.3, 0.003, 0.9, 0.001)
#         neural_network.set_codes(codes)
#         initialise_weights_and_biases(model_python, neural_network)

#         # python
#         for it in range(1):
#             update_hidden_weights_adam_(model_python, inputs, codes, lambda_w=0, n_iter=n_iter)
#             update_last_layer_(model_python[-1], codes[-1], targets, nn.MSELoss(), n_iter)

#         # cpp
#         for it in range(1):
#             neural_network.update_weights(inputs, targets)
        
#         weights = neural_network.return_weights()
#         biases = neural_network.return_bias()

#         # Assert weights and biases updated the same
#         check_equal_weights_and_bias(model_python, weights, biases, 10e6)


#     def test_update_weights_CrossEntropyLoss(self):
#         # Setup 
#         inputs = torch.rand(4, 784, dtype=torch.double)
#         targets = torch.randint(0, 9, (4,))
#         codes = [torch.rand(4,100, dtype=torch.double).detach(), torch.rand(4,100, dtype=torch.double).detach()]
#         n_iter = 2
#         criterion = nn.CrossEntropyLoss()

#         # Model setup
#         model_python = FFNet(784, n_hiddens=100, n_hidden_layers=2, batchnorm=False, bias=True).double()
#         model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
#                         scheduler=lambda epoch: 1/2**(epoch//8))
#         model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
#         model_python = model_python[1:]
        
#         # cpp model setup
#         neural_network = conv_model_to_class(model_python, criterion, targets.shape[0],1, n_iter)
#         neural_network.set_codes(codes)
#         initialise_weights_and_biases(model_python, neural_network)

#         # python
#         for it in range(2):
#             update_hidden_weights_adam_(model_python, inputs, codes, lambda_w=0, n_iter=n_iter)
#             update_last_layer_(model_python[-1], codes[-1], targets, criterion, n_iter)

#         # cpp
#         targets = targets.double()
#         targets = targets.reshape(1,len(targets))
#         for it in range(2):
#             neural_network.update_weights(inputs, targets)
        
#         weights = neural_network.return_weights()
#         biases = neural_network.return_bias()

#         # Assert weights and biases updated the same
#         check_equal_weights_and_bias(model_python, weights, biases, 10e6)


import time 
import numpy as np
from log_approximator import *

# class TestConvergence(unittest.TestCase):

#     def test_log_approx(self):
#         criterion = nn.MSELoss()
#         X = np.arange(0.0,10.0,0.1)
#         #Avoid divide by 0
#         X = X[1:]
#         X = X.reshape(99,1)
#         Y = np.log(X)
#         epochs = 100

#         model = LogApproximator(100)
#         model = get_mods(model)
#         model = model[1:]
#         start = time.time()
#         loss_sgd, outputs_sgd = sgd(model, X, Y, epochs)
#         end = time.time()
#         time_sgd = end-start
#         print("Time for sgd :" + str(time_sgd))

#         model = LogApproximator(100)
#         model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.001})
#         model[-1].optimizer.param_groups[0]['lr'] = 0.001
#         model = model[1:]
#         start = time.time()
#         loss_altmin, outputs_altmin = run_altmin(model, X, Y, epochs)
#         end = time.time()
#         time_altmin = end-start 
#         print("Time for altmin :" + str(time_altmin))

#         model = LogApproximator(100)
#         model = get_mods(model)
#         model = model[1:]
#         neural_network = conv_model_to_class(model, nn.MSELoss(),  1, 1, 1, 0.001, 0.003, 0.9, 0.001)
#         start = time.time()
#         loss_cpp, outputs_cpp = run_cpp_altmin(neural_network, X, Y, epochs)
#         end = time.time()
#         time_cpp = end-start
#         print("Time for cpp :" + str(time_cpp))

#         #Optional but prints res and how loss changes
#         show_res(X, Y, outputs_sgd, outputs_altmin, outputs_cpp, loss_sgd, loss_altmin, loss_cpp)

# def save_state_to_dict(file_path):
#     model = simpleNN(2, [4,3],1)
#     model = get_mods(model)
#     model = model[1:]
#     params = model.state_dict()
#     neural_network = conv_model_to_class(model, nn.BCELoss(), 4)
#     weights = neural_network.return_weights()
#     biases = neural_network.return_bias()

#     print(params)
#     x = 0
#     y = 0
#     for key in params:
#         if 'weight' in key:
#             params[key]  = weights[x]
#             x+=1
#         else:
#             params[key] = biases[y]
#             y+=1

#     torch.save(model.state_dict(), "models.test.pt")
    

# def initialise_state_from_dict():
#     model = simpleNN(2, [4,3],1)
#     model = get_mods(model)
#     model = model[1:]
#     neural_network = conv_model_to_class(model, nn.BCELoss(), 4)
#     params = torch.load("models/test.pt")
    
#     weights = [] 
#     biases = []
#     for key in params:
#         if 'weight' in key:
#             weights.append(params[key])
#         else:
#             biases.append(params[key])

#     neural_network.set_weights_and_biases(weights,biases)
        
                
class BenchmarkForward(unittest.TestCase):

    # def test_transpose_vs_no_transpose(self):
    #     inputs = []
    #     for x in range(5000):
    #         inputs.append(np.random.rand(5000,5))

    #     model = nn.Linear(5,25).double()
    #     weight = model.weight.data.numpy()
    #     weight_transpose = weight.transpose(1,0)
    #     bias = model.bias.data.numpy()
    #     time_no_transpose = 0.0 
    #     time_transpose = 0.0
    #     for input in inputs:
    #         start = time.time()
    #         a = fast_altmin.lin(input, weight, bias)
    #         end = time.time()
    #         time_transpose += (end-start)
    #         start=time.time()
    #         b = fast_altmin.lin_no_transpose(input, weight_transpose, bias)
    #         end = time.time()
    #         time_no_transpose+= (end-start)
    #     print("Time lin with transpose: "+str(time_transpose))
    #     print("Time lin with no transpose: "+str(time_no_transpose))

    def test_linear(self):
        inputs_one_cpp = []
        inputs_two_cpp = []
        inputs_three_cpp = []

        inputs_one_py = []
        inputs_two_py = []
        inputs_three_py = []

        batch_size = 100

        for x in range(1500):
            input = np.random.rand(batch_size,5)
            inputs_one_cpp.append(input)
            inputs_one_py.append(torch.from_numpy(input))

        for x in range(1500):
            input = np.random.rand(batch_size,25)
            inputs_two_cpp.append(input)
            inputs_two_py.append(torch.from_numpy(input))

        for x in range(1500):
            input = np.random.rand(batch_size,30)
            inputs_three_cpp.append(input)
            inputs_three_py.append(torch.from_numpy(input))

        time_cpp = 0.0 
        time_py = 0.0

        model = nn.Linear(5,25).double()
        for input in inputs_one_py:
            start = time.time()
            model(input)
            end = time.time()
            time_py += (end-start)

        model = nn.Linear(25,30).double()
        for input in inputs_two_py:
            start = time.time()
            model(input)
            end = time.time()
            time_py += (end-start)

        model = nn.Linear(30,1).double()
        for input in inputs_three_py:
            start = time.time()
            model(input)
            end = time.time()
            time_py += (end-start)

        print("Time for py: " + str(time_py))

        model = nn.Linear(5,25).double()
        weight_transpose = model.weight.data.transpose(1,0).numpy()
        bias = model.bias.data.numpy()
        for input in inputs_one_cpp:
            start = time.time()
            fast_altmin.lin_no_transpose(input, weight_transpose, bias)
            end = time.time()
            time_cpp += (end-start)

        model = nn.Linear(25,30).double()
        weight_transpose = model.weight.data.transpose(1,0).numpy()
        bias = model.bias.data.numpy()
        for input in inputs_two_cpp:
            start = time.time()
            fast_altmin.lin_no_transpose(input, weight_transpose, bias)
            end = time.time()
            time_cpp += (end-start)

        model = nn.Linear(30,1).double()
        weight_transpose = model.weight.data.transpose(1,0).numpy()
        bias = model.bias.data.numpy()
        for input in inputs_three_cpp:
            start = time.time()
            fast_altmin.lin_no_transpose(input, weight_transpose, bias)
            end = time.time()
            time_cpp += (end-start)

        print("Time for cpp: " + str(time_cpp))

        

    # def test_forward_time(self):
    #     model = simpleNN(5, [25,30],1)
    #     model = get_mods(model)
    #     model = model[1:]
    #     neural_network = conv_model_to_class(model, nn.BCELoss(), 5000)
    #     start = time.time()
    #     for it in range(5000):
    #         in_tensor = torch.rand(5000,5,dtype = torch.double)
    #         output_cpp = neural_network.get_codes(in_tensor, True)
    #     end=time.time()
    #     print("Cpp forward: "+(str(end-start)))

    #     start = time.time()
    #     for it in range(5000):
    #         in_tensor = torch.rand(5000,5,dtype = torch.double)
    #         output_python, codes_python = get_codes(model, in_tensor)
    #     end=time.time()
    #     print("Python forward: "+(str(end-start)))

    def test_lin_time(self):
        
        n = 5000 
        m = 5
        a = 5
        b = 25
        start =time.time()
        fast_altmin.matrix_mul_two(m,n,b,a)
        end = time.time()
        print("cpp amt mul "+str(end-start))

        start=time.time()
        for i in range(5000):
            input = torch.rand(n,m)
            weight = torch.rand(a,b)
            res = torch.matmul(input,weight)

        end = time.time()
        print("py mat mul"+str(end-start))
        
        # model = nn.Linear(5,25).double()
        # weight = model.weight.data.transpose(1,0)        
        # bias = model.bias.data

        # count = 0.0
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,5,dtype = torch.double)
        #     start = time.time()
        #     fast_altmin.matrix_mul(in_tensor, weight) 
        #     end=time.time()
        #     count += (end-start)
        # print("cpp mul time " + str(count)) 

        # count = 0.0
        
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,5,dtype = torch.double)
        #     start=time.time()
        #     c = torch.matmul(in_tensor,weight)
        #     end=time.time()
        #     count+=(end-start)
        # print("py mul time " + str(count)) 

        # count=0.0
        # model = nn.Linear(5,25).double()
        # weight = model.weight.data
        # weight_transpose = weight.transpose(1,0)
        # print(weight)
        # print(weight_transpose)
        # bias = model.bias.data
        # for it in range(1):
        #     in_tensor = torch.rand(5000,5,dtype = torch.double)
        #     start = time.time()
        #     a = fast_altmin.lin(in_tensor, weight, bias)
        #     b = fast_altmin.lin_two(in_tensor, weight_transpose, bias)
        #     print(a)
        #     print(b)
        #     end=time.time() 
        #     count+=(end-start)

        # model = nn.Linear(25,30).double()
        # weight = model.weight.data
        # bias = model.bias.data
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,25,dtype = torch.double)
        #     start = time.time()
        #     fast_altmin.lin(in_tensor, weight, bias) 
        #     end = time.time()
        #     count+=(end-start)

        # model = nn.Linear(30,1).double()
        # weight = model.weight.data
        # bias = model.bias.data
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,30,dtype = torch.double)
        #     start = time.time()
        #     fast_altmin.lin(in_tensor, weight, bias) 
        #     end=time.time()
        #     count+=(end-start)
        # print("Cpp lin 1: "+str(count))

        # count = 0.0
        # model = nn.Linear(5,25).double()
        # weight = model.weight.data.transpose(1,0)
        # bias = model.bias.data
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,5,dtype = torch.double)
        #     start = time.time()
        #     fast_altmin.lin(in_tensor, weight, bias)
        #     end=time.time() 
        #     count+=(end-start)

        # model = nn.Linear(25,30).double()
        # weight = model.weight.data.transpose(1,0)
        # bias = model.bias.data
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,25,dtype = torch.double)
        #     start = time.time()
        #     fast_altmin.lin(in_tensor, weight, bias) 
        #     end = time.time()
        #     count+=(end-start)

        # model = nn.Linear(30,1).double()
        # weight = model.weight.data.transpose(1,0)
        # bias = model.bias.data
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,30,dtype = torch.double)
        #     start = time.time()
        #     fast_altmin.lin(in_tensor, weight, bias) 
        #     end=time.time()
        #     count+=(end-start)
        # print("Cpp lin 2: "+str(count))
       

        # count = 0.0
        # model = nn.Linear(5,25).double()
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,5,dtype = torch.double)
        #     start = time.time()
        #     x = model(in_tensor)
        #     end=time.time() 
        #     count+=(end-start)

        # model = nn.Linear(25,30).double()
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,25,dtype = torch.double)
        #     start = time.time()
        #     x = model(in_tensor)
        #     end = time.time()
        #     count+=(end-start)

        # model = nn.Linear(30,1).double()
        # for it in range(5000):
        #     in_tensor = torch.rand(5000,30,dtype = torch.double)
        #     start = time.time()
        #     x = model(in_tensor)
        #     end=time.time()
        #     count+=(end-start)

        # print("Py lin "+str(count))
       


    # Functions for relu and sigmoid are in place so they don't need to return a value

    # def test_ReLU_time(self):
    #     start = time.time()
    #     for it in range(5000):
    #         in_tensor = torch.rand(5000,30,dtype = torch.double)
    #         fast_altmin.ReLU(in_tensor)
    #     end=time.time()
    #     print("Cpp relu: "+(str(end-start)))

    #     model = nn.ReLU()
    #     start = time.time()
    #     for it in range(5000):
    #         in_tensor = torch.rand(5000,30,dtype = torch.double)
    #         model(in_tensor)
    #     end=time.time()
    #     print("Python relu: "+(str(end-start)))

    # def test_Sigmoid_time(self):
    #     start = time.time()
    #     for it in range(5000):
    #         in_tensor = torch.rand(5000,30,dtype = torch.double)
    #         fast_altmin.sigmoid(in_tensor)
    #     end=time.time()
    #     print("Cpp Sigmoid: "+(str(end-start)))

    #     model = nn.Sigmoid()
    #     start = time.time()
    #     for it in range(5000):
    #         in_tensor = torch.rand(5000,30,dtype = torch.double)
    #         model(in_tensor)
    #     end=time.time()
    #     print("Python Sigmoid: "+(str(end-start)))


  
if __name__ == '__main__':
    unittest.main()
