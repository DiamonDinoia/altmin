import fast_altmin
import torch.nn as nn
import torch 
import unittest
import torch.nn.functional as F
import torch.optim as optim
from altmin import simpleNN, get_mods, get_codes, update_codes, compute_codes_loss, update_hidden_weights_adam_, update_last_layer_, FFNet, load_dataset, LeNet
import derivatives
import sys 
import time 
import log_approximator

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


class TestLayers(unittest.TestCase):
    def test_lin(self):
        lin = nn.Linear(2, 4).double()
        in_tensor = torch.rand(1, 2, dtype=torch.double)
        weight = lin.weight.data
        bias = lin.bias.data
        #bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))
        print("Jdfg")
        print(bias.shape)
        cpp_imp = torch.from_numpy(fast_altmin.lin(in_tensor,
                                      weight, bias))
        python_imp = lin(in_tensor)
        check_equal(cpp_imp, python_imp, 10e8)

    def test_lin_batch(self):
        lin = nn.Linear(4, 6).double()
        in_tensor = torch.rand(5, 4, dtype=torch.double)
        weight = lin.weight.data
        #bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))
        bias = lin.bias.data
        start = time.time()
        cpp_imp = torch.from_numpy(fast_altmin.lin(in_tensor,
                                      weight, bias))
        end=time.time()
        print("Cpp lin: " +str(end-start))
        start = time.time()
        python_imp = lin(in_tensor)
        end=time.time()
        print("py lin: " +str(end-start))
        check_equal(cpp_imp, python_imp, 10e8)

    # Functions for relu and sigmoid are in place so they don't need to return a value

    def test_ReLU(self):
        relu = nn.ReLU()
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        start = time.time()
        python_imp = relu(in_tensor)
        # In tensor data updated in place
        end=time.time()
        print("py reliu: " +str(end-start))
        start = time.time()
        cpp_imp = fast_altmin.ReLU(in_tensor)
        end=time.time()
        print("cpp relu: " +str(end-start))

        check_equal(cpp_imp, python_imp, 10e8)

    def test_sigmoid(self):
        sigmoid = nn.Sigmoid()
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        start = time.time()
        python_imp = sigmoid(in_tensor)
        # In tensor data changed in place?
        end=time.time()
        print("py sig: " +str(end-start))
        start = time.time()
        cpp_imp = fast_altmin.sigmoid(in_tensor)
        end=time.time()
        print("cpp sig: " +str(end-start))

        check_equal(cpp_imp, python_imp, 10e8)

# Testing the cpp implementation of BCEloss and MSEloss
# Also test the fuctions ability to calculate loss on batch input
# Again take reference to tensor as input so no data copied between
# Can't use epsilon as these diverge slightly around 8 decimal places but no a priority to fix rn but I'll come back to this
# Again I don't use these functions I just use the torch imp atm but I'll keep them as they might be useful if I can move away from autograd. 
class TestCriterion(unittest.TestCase):
    def test_BCELoss(self):
        targets = torch.round(torch.rand(1, 10, dtype=torch.double))
        predictions = torch.rand(1, 10, dtype=torch.double)
        cpp_loss = fast_altmin.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_batch_BCELoss(self):
        targets = torch.round(torch.rand(5000, 5, dtype=torch.double))
        predictions = torch.rand(5000, 5, dtype=torch.double)
        start = time.time()
        cpp_loss = fast_altmin.BCELoss(predictions, targets)
        end = time.time() 
        print("BCE Loss: " + str(end-start))
        start = time.time()
        python_loss = nn.BCELoss()(predictions, targets)
        end = time.time() 
        print("BCE Loss python: " + str(end-start))
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_MSELoss(self):
        targets = torch.rand(1, 10, dtype=torch.double)
        predictions = torch.rand(1, 10, dtype=torch.double)
        cpp_loss = fast_altmin.MSELoss(predictions, targets)
        python_loss = nn.MSELoss()(predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_batch_MSELoss(self):
        targets = torch.rand(5000, 100, dtype=torch.double)
        predictions = torch.rand(5000, 100, dtype=torch.double)
        start = time.time()
        cpp_loss = fast_altmin.MSELoss(predictions, targets)
        end = time.time()
        print("MSE Loss: " + str(end-start))
        start = time.time()
        python_loss = nn.functional.mse_loss(
            predictions, targets)
        end = time.time()
        print("MSE Loss pyton: " + str(end-start))
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_log_softmax(self):
        inputs = torch.rand(1,5, dtype=torch.double)
        python_imp = nn.LogSoftmax(dim=1)(inputs)
        fast_altmin.log_softmax(inputs)
        check_equal(inputs, python_imp, 10e8)

    def test_batch_log_softmax(self):
        inputs = torch.rand(3,5, dtype=torch.double)
        python_imp = nn.LogSoftmax(dim=1)(inputs)
        fast_altmin.log_softmax(inputs)
        check_equal(inputs, python_imp, 10e8)

    def test_negative_log_likelihood(self):
        inputs = torch.rand(3,5, dtype=torch.double)
        targets = torch.tensor([2,1,4])
        inputs = nn.LogSoftmax(dim=1)(inputs)
        python_loss = nn.NLLLoss()(inputs, targets)
        cpp_loss = fast_altmin.negative_log_likelihood(inputs,targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_cross_entropy_loss(self):
        inputs = torch.rand(5000,5, dtype=torch.double)
        targets = torch.randint(0, 5, (5000,))
        start = time.time()
        python_loss = nn.CrossEntropyLoss()(inputs,targets)
        end = time.time()
        print("py cross ent Loss: " + str(end-start))
        targets = targets.double()
        start = time.time()
        cpp_loss = fast_altmin.cross_entropy_loss(inputs,targets)
        end = time.time()
        print("cpp cross ent Loss: " + str(end-start))
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)



class TestDerivatives(unittest.TestCase):
    def test_relu_derivative(self):
        input_python = torch.rand(2,5)
        input_python -=0.5
        input_python.requires_grad_(True) 
        input_cpp = input_python.detach().clone()

        grad_python = derivatives.derivative_relu(input_python) 
        
        #https://nanobind.readthedocs.io/en/latest/eigen.html
        #https://pytorch.org/docs/stable/generated/torch.from_numpy.html
        #I think this means there is no copy.
        grad_cpp = torch.from_numpy(fast_altmin.differentiate_ReLU(input_cpp))
        #print(type(grad_cpp))
        #Correct to 8 d.p to need to improve but works for now 
        check_equal(grad_python,grad_cpp, 10e9) 

    # def test_sigmoid_derivative(self):
    #     input_python = torch.rand(2,5).requires_grad_(True) 
    #     input_cpp = input_python.detach().clone()

    #     grad_python = derivatives.derivative_sigmoid(input_python) 
    #     grad_cpp = torch.from_numpy(fast_altmin.differentiate_sigmoid(input_cpp))
    #     check_equal(grad_python,grad_cpp, 10e9) 

    def test_lin_derivative(self):
        input_python = torch.rand(2,5).requires_grad_(True) 
        input_cpp = input_python.detach().clone()

        w_grad_python, b_grad_python = derivatives.derivative_linear(input_python) 
        #print(w_grad_python)
        #print(b_grad_python)
        #grads_cpp = torch.from_numpy(fast_altmin.differentiate_linear_layer(input_cpp))
        #print(grads_cpp)
        #check_equal(grad_python,grad_cpp, 10e9) 

    def test_MSELoss_derivative(self):
        output_python = torch.rand(2,5).requires_grad_(True) 
        target = torch.rand(2,5)
        output_cpp = output_python.detach().clone()

        grad_python = derivatives.derivative_MSELoss(output_python,target)

        grad_cpp = torch.from_numpy(fast_altmin.differentiate_MSELoss(output_cpp,target))
        #print(grad_python)
        #print(grad_cpp)
        check_equal(grad_python,grad_cpp, 10e8) 


    def test_BCELoss_derivative(self):
        output_python = torch.rand(2,5).requires_grad_(True) 
        target = torch.round(torch.rand(2,5))
        output_cpp = output_python.detach().clone()

        grad_python = derivatives.derivative_BCELoss(output_python,target)
        grad_cpp = torch.from_numpy(fast_altmin.differentiate_BCELoss(output_cpp,target))
        check_equal(grad_python,grad_cpp, 10e9) 

    def test_CrossEntropyLoss_derivative(self):
        output_python = torch.rand(4,5).requires_grad_(True) 
        target = torch.tensor([2,1,4,0])
        num_classes = 5
        output_cpp = output_python.detach().clone()
        start = time.time()
        loss = nn.CrossEntropyLoss()(output_python,target)
        loss.backward()
        end = time.time()
        print("Py Diff cross ent "+str(end-start))
        grad_python = output_python.grad
        start = time.time()
        grad_cpp = fast_altmin.differentiate_CrossEntropyLoss(output_cpp,target, num_classes)
        end = time.time()
        print("cpp Diff cross ent "+str(end-start))
        #grad_cpp = derivatives.derivative_CrossEntropyLoss(output_cpp,target)
        check_equal(grad_python,grad_cpp, 10e9) 




class TestUpdateFunctions(unittest.TestCase):
    def test_forward(self):
       
        in_tensor = torch.rand(7,2,dtype = torch.double)
        model = simpleNN(2, [4,3],1)
        
        model = get_mods(model)
        model = model[1:]
        neural_network = fast_altmin.NeuralNetworkBCE()
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()
        output_python, codes_python = get_codes(model, in_tensor)
        output_cpp = neural_network.get_codes(in_tensor, True)
        codes_cpp = neural_network.return_codes() 

        #Check output 
        check_equal(output_python, output_cpp, 10e6)
        #check codes
        for x in range(len(codes_python)):
            check_equal(codes_python[x], codes_cpp[x], 10e6)

            
    def test_update_codes_BCELoss(self):
        in_tensor = torch.rand(7,2,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.BCELoss()

        model = simpleNN(2, [4,3],1)
        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]

        for it in range(4):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, in_tensor)

            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)


        neural_network = fast_altmin.NeuralNetworkBCE()
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()
        
        for it in range(4):
            output_cpp = neural_network.get_codes(in_tensor, True)
 
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        
        #check codes
        for x in range(len(codes_python)):
            
            check_equal(codes_python[x], codes_cpp[x], 10e6)

    def test_update_codes_MSELoss(self):
        in_tensor = torch.rand(7,1,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.MSELoss()

        model= log_approximator.LogApproximator(5)
        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        with torch.no_grad():
            output_python, codes_python = get_codes(model, in_tensor)
        for it in range(1):
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)


        print("a")
        neural_network = fast_altmin.NeuralNetworkMSE()
        print("b")
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()

        output_cpp = neural_network.get_codes(in_tensor, True)

        print("JI")
        for it in range(1):     
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        #check codes
        for x in range(len(codes_python)):
            
            check_equal(codes_python[x], codes_cpp[x], 10e6)

    def test_update_codes_CrossEntropyLoss(self):
        in_tensor = torch.rand(4,10,dtype = torch.double)
        targets = torch.randint(0, 10, (4,))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.CrossEntropyLoss()

         # Model setup
        model = FFNet(10, n_hiddens=100, n_hidden_layers=2, batchnorm=False, bias=True).double()        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        with torch.no_grad():
            output_python, codes_python = get_codes(model, in_tensor)
        for it in range(1):
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)


        print("a")
        neural_network = fast_altmin.NeuralNetworkCrossEntropy()
        print("b")
        fast_altmin.create_model_class(model, neural_network, 4, 0)
        neural_network.construct_pairs()

        output_cpp = neural_network.get_codes(in_tensor, True)

        print("JI")
        targets = targets.double()
        targets = targets.reshape(1,len(targets))
        for it in range(1):   
              
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        #check codes
        for x in range(len(codes_python)):
            
            check_equal(codes_python[x], codes_cpp[x], 10e6)


    def test_update_weights_BCELoss(self):
        in_tensor = torch.rand(7,2,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.BCELoss()

        model = simpleNN(2, [4,3],1)
        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        neural_network = fast_altmin.NeuralNetworkBCE()
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()


        for it in range(1):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, in_tensor)

            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)
            update_hidden_weights_adam_(model, in_tensor, codes_python, lambda_w=0, n_iter=n_iter)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.BCELoss(), n_iter)

        
        
        for it in range(1):
            output_cpp = neural_network.get_codes(in_tensor, True)  
            neural_network.update_codes(targets)
            neural_network.update_weights_parallel(in_tensor, targets)
        
        
        weights = neural_network.return_weights()
        biases = neural_network.return_biases()


        check_equal_weights_and_bias(model, weights, biases, 10e9)

    def test_update_weights_MSELoss(self):
        in_tensor = torch.rand(7,1,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.MSELoss()

        model= log_approximator.LogApproximator(5)
        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        with torch.no_grad():
            output_python, codes_python = get_codes(model, in_tensor)
        for it in range(1):
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)


        print("a")
        neural_network = fast_altmin.NeuralNetworkMSE()
        print("b")
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()

        output_cpp = neural_network.get_codes(in_tensor, True)

        print("JI")
        for it in range(1):     
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        #Update weights
        for it in range(1):
            update_hidden_weights_adam_(model, in_tensor, codes_python, lambda_w=0, n_iter=n_iter)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.MSELoss(), n_iter)

        for it in range(1):
            #neural_network.update_weights_not_parallel(in_tensor, targets)
            neural_network.update_weights_parallel(in_tensor, targets)
        
        
        weights = neural_network.return_weights()
        biases = neural_network.return_biases()
        check_equal_weights_and_bias(model, weights, biases, 10e9)

    def test_update_weights_CrossEntropyLoss(self):
        in_tensor = torch.rand(4,10,dtype = torch.double)
        targets = torch.randint(0, 10, (4,))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.CrossEntropyLoss()

         # Model setup
        model = FFNet(10, n_hiddens=100, n_hidden_layers=2, batchnorm=False, bias=True).double()        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        with torch.no_grad():
            output_python, codes_python = get_codes(model, in_tensor)
        for it in range(1):
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)


        print("a")
        neural_network = fast_altmin.NeuralNetworkCrossEntropy()
        print("b")
        fast_altmin.create_model_class(model, neural_network, 4, 0)
        neural_network.construct_pairs()

        output_cpp = neural_network.get_codes(in_tensor, True)

        print("JI")
        targets_cpp = targets.double()
        targets_cpp = targets.reshape(1,len(targets))
        for it in range(1):   
              
            neural_network.update_codes(targets_cpp)
        codes_cpp = neural_network.return_codes() 

        #Update weights
        for it in range(1):
            update_hidden_weights_adam_(model, in_tensor, codes_python, lambda_w=0, n_iter=n_iter)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.CrossEntropyLoss(), n_iter)

        for it in range(1):
            #neural_network.update_weights_not_parallel(in_tensor, targets)
            neural_network.update_weights_parallel(in_tensor, targets_cpp)
        
        
        weights = neural_network.return_weights()
        biases = neural_network.return_biases()
        check_equal_weights_and_bias(model, weights, biases, 10e9)

    def test_a(self):
        
        fast_altmin.test_variant()

    def test_forward(self):
       
        in_tensor = torch.rand(7,2,dtype = torch.double)
        model = simpleNN(2, [4,3],1)
        
        model = get_mods(model)
        model = model[1:]
        neural_network = fast_altmin.VariantNeuralNetworkBCE()
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        #neural_network.construct_pairs()
        output_python, codes_python = get_codes(model, in_tensor)
        output_cpp = neural_network.get_codes(in_tensor, True)
        codes_cpp = neural_network.return_codes() 

        #Check output 
        check_equal(output_python, output_cpp, 10e6)
        #check codes
        for x in range(len(codes_python)):
            check_equal(codes_python[x], codes_cpp[x], 10e6)


    def test_update_codes_BCELoss(self):
        in_tensor = torch.rand(7,2,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.BCELoss()

        model = simpleNN(2, [4,3],1)
        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]

        for it in range(4):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, in_tensor)

            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)


        neural_network = fast_altmin.VariantNeuralNetworkBCE()
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        
        for it in range(4):
            output_cpp = neural_network.get_codes(in_tensor, True)
 
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        
        #check codes
        for x in range(len(codes_python)):
            
            check_equal(codes_python[x], codes_cpp[x], 10e6)

    def test_update_codes_MSELoss(self):
        in_tensor = torch.rand(7,1,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.MSELoss()

        model= log_approximator.LogApproximator(5)
        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        with torch.no_grad():
            output_python, codes_python = get_codes(model, in_tensor)
        for it in range(1):
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)



        neural_network = fast_altmin.VariantNeuralNetworkMSE()

        fast_altmin.create_model_class(model, neural_network, 7, 0)
        #neural_network.construct_pairs()

        output_cpp = neural_network.get_codes(in_tensor, True)


        for it in range(1):     
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        #check codes
        for x in range(len(codes_python)):
            
            check_equal(codes_python[x], codes_cpp[x], 10e6)

    def test_update_codes_CrossEntropyLoss(self):
        in_tensor = torch.rand(4,10,dtype = torch.double)
        targets = torch.randint(0, 10, (4,))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.CrossEntropyLoss()

         # Model setup
        model = FFNet(10, n_hiddens=100, n_hidden_layers=2, batchnorm=False, bias=True).double()        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        with torch.no_grad():
            output_python, codes_python = get_codes(model, in_tensor)
        for it in range(1):
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)


 
        neural_network = fast_altmin.VariantNeuralNetworkCrossEntropy()

        fast_altmin.create_model_class(model, neural_network, 4, 0)
        #neural_network.construct_pairs()

        output_cpp = neural_network.get_codes(in_tensor, True)


        targets = targets.double()
        targets = targets.reshape(1,len(targets))
        for it in range(1):   
              
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        #check codes
        for x in range(len(codes_python)):
            
            check_equal(codes_python[x], codes_cpp[x], 10e6)

    def test_update_weights_BCELoss(self):
        in_tensor = torch.rand(7,2,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.BCELoss()

        model = simpleNN(2, [4,3],1)
        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        neural_network = fast_altmin.VariantNeuralNetworkBCE()
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()


        for it in range(1):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, in_tensor)

            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)
            update_hidden_weights_adam_(model, in_tensor, codes_python, lambda_w=0, n_iter=n_iter)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.BCELoss(), n_iter)

        
        
        for it in range(1):
            output_cpp = neural_network.get_codes(in_tensor, True)  
            neural_network.update_codes(targets)
            neural_network.update_weights_parallel(in_tensor, targets)
        
        
        weights = neural_network.return_weights()
        biases = neural_network.return_biases()

        check_equal_weights_and_bias(model, weights, biases, 10e9)

    def test_update_weights_MSELoss(self):
        in_tensor = torch.rand(7,1,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.MSELoss()

        model= log_approximator.LogApproximator(5)
        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        with torch.no_grad():
            output_python, codes_python = get_codes(model, in_tensor)
        for it in range(1):
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)



        neural_network = fast_altmin.VariantNeuralNetworkMSE()

        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()

        output_cpp = neural_network.get_codes(in_tensor, True)


        for it in range(1):     
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        #Update weights
        for it in range(1):
            update_hidden_weights_adam_(model, in_tensor, codes_python, lambda_w=0, n_iter=n_iter)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.MSELoss(), n_iter)

        for it in range(1):
            #neural_network.update_weights_not_parallel(in_tensor, targets)
            neural_network.update_weights_parallel(in_tensor, targets)
        
        
        weights = neural_network.return_weights()
        biases = neural_network.return_biases()
        check_equal_weights_and_bias(model, weights, biases, 10e9)

    def test_update_weights_CrossEntropyLoss(self):
        in_tensor = torch.rand(4,10,dtype = torch.double)
        targets = torch.randint(0, 10, (4,))
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.CrossEntropyLoss()

         # Model setup
        model = FFNet(10, n_hiddens=100, n_hidden_layers=2, batchnorm=False, bias=True).double()        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        with torch.no_grad():
            output_python, codes_python = get_codes(model, in_tensor)
        for it in range(1):
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)



        neural_network = fast_altmin.VariantNeuralNetworkCrossEntropy()

        fast_altmin.create_model_class(model, neural_network, 4, 0)
        neural_network.construct_pairs()

        output_cpp = neural_network.get_codes(in_tensor, True)

        targets_cpp = targets.double()
        targets_cpp = targets.reshape(1,len(targets))
        for it in range(1):   
              
            neural_network.update_codes(targets_cpp)
        codes_cpp = neural_network.return_codes() 

        #Update weights
        for it in range(1):
            update_hidden_weights_adam_(model, in_tensor, codes_python, lambda_w=0, n_iter=n_iter)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.CrossEntropyLoss(), n_iter)

        for it in range(1):
            #neural_network.update_weights_not_parallel(in_tensor, targets)
            neural_network.update_weights_parallel(in_tensor, targets_cpp)
        
        
        weights = neural_network.return_weights()
        biases = neural_network.return_biases()
        check_equal_weights_and_bias(model, weights, biases, 10e9)


    

# class TestCNN(unittest.TestCase):
#     def test_conv2d(self):
#         train_loader, test_loader, n_inputs = load_dataset('mnist', 1,  conv_net=True,
#                                                        data_augmentation=True)
#         window_size = train_loader.dataset.data[0].shape[0]
        
#         if len(train_loader.dataset.data[0].shape) == 3:
#             num_input_channels = train_loader.dataset.data[0].shape[2]
#         else:
#             num_input_channels = 1
        

#         model = LeNet(num_input_channels=num_input_channels, window_size=window_size, bias=True)
#         print(model)
#         for batch_idx, (data, targets) in enumerate(train_loader):
#             print(data.shape)
#             #print(targets.shape)
#             #in channels, out channels, kernel size, bias
#             mod = nn.Conv2d(num_input_channels, 6, 5, bias=True)
#             out = mod(data)
#             print(out.shape)
#             break


if __name__ == '__main__':
    unittest.main()