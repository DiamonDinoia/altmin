import torch.nn as nn
import numpy as np
import torch


import unittest
import math


from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim


#from manual_altmin import update_last_layer_manual, store_momentums, update_hidden_weights_adam_manual, update_codes_manual
import pickle
import os, sys 
sys.path.insert(0, os.path.abspath("../artifacts"))
from control_flow import cf_get_codes, cf_update_codes, cf_update_hidden_weights, cf_update_last_layer
from altmin import get_mods, get_codes, update_codes, update_last_layer_, update_hidden_weights_adam_, store_momentums, simpleNN
import nanobind

# Assert all values are the same to tolerance of ulp
def check_equal(first_imp, second_imp, eps):
    for x in range(len(first_imp)):
        for y in range(len(first_imp[x])):
            assert(abs(first_imp[x][y] - second_imp[x]
                        [y]) <= sys.float_info.epsilon*eps)
            
def check_equal_bias(first_imp, second_imp,eps):
    for x in range(len(first_imp)):
        assert(abs(first_imp[x] - second_imp[x]) <= sys.float_info.epsilon*eps)
                
class TestHelloWorldOut(unittest.TestCase):
    # This test basically just checks nanobind is still working in it's simplest form
    def test_output_from_cpp(self):
        res = nanobind.hello_world_out()
        assert(res == "Hi python from c++")

class TestMatrixMultiplication(unittest.TestCase):
    # This test checks eigen and nanobind are working together
    def test_matrix_multiplication_in_cpp(self):
        a = np.asarray([[3, 2], [-1, 4]])
        b = np.asarray([[1, 0], [1, 1]])
        c = nanobind.matrix_multiplication(a, b)
        assert((c == np.asarray([[5, 2], [3, 4]])).all())

# These tests check the layers that have been implemented in c++.
# They all take tensors as input
# For non-linear layers the function performs the operation in place as a reference to the data is passed to avoid copying the data unecessarily
# The linear functions return a ndarray in row-major order so this needs to be converted to a tensor
# The relu and sigmoid functions aren't used but I'll keep them for now as they might be used if I can move away from autograd

class TestLayers(unittest.TestCase):
    # Assert all values are the same to tolerance of ulp
    def check_equal(self, cpp_imp, python_imp):
        for x in range(len(cpp_imp)):
            for y in range(len(cpp_imp[x])):
                assert(abs(cpp_imp[x][y] - python_imp[x]
                           [y]) <= sys.float_info.epsilon)

    def test_lin(self):
        lin = nn.Linear(2, 4).double()
        in_tensor = torch.rand(1, 2, dtype=torch.double)
        weight = lin.weight.data
        bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))
        cpp_imp = nanobind.lin(in_tensor,
                                      weight, bias)
        python_imp = lin(in_tensor).detach().numpy()
        self.check_equal(cpp_imp, python_imp)

    def test_lin_batch(self):
        lin = nn.Linear(4, 6).double()
        in_tensor = torch.rand(5, 4, dtype=torch.double)
        weight = lin.weight.data
        bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))
        cpp_imp = nanobind.lin(in_tensor,
                                      weight, bias)
        python_imp = lin(in_tensor).detach().numpy()
        self.check_equal(cpp_imp, python_imp)

    # Functions for relu and sigmoid are in place so they don't need to return a value

    def test_ReLU(self):
        relu = nn.ReLU()
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        python_imp = relu(in_tensor)
        # In tensor data updated in place
        cpp_imp = nanobind.ReLU(
            in_tensor)

        self.check_equal(cpp_imp, python_imp)

    def test_sigmoid(self):
        sigmoid = nn.Sigmoid()
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        python_imp = sigmoid(in_tensor)
        # In tensor data changed in place?
        cpp_imp = nanobind.sigmoid(
            in_tensor)

        self.check_equal(cpp_imp, python_imp)


# Testing the cpp implementation of BCEloss and MSEloss
# Also test the fuctions ability to calculate loss on batch input
# Again take reference to tensor as input so no data copied between
# Can't use epsilon as these diverge slightly around 8 decimal places but no a priority to fix rn but I'll come back to this
# Again I don't use these functions I just use the torch imp atm but I'll keep them as they might be useful if I can move away from autograd. 
class TestCriterion(unittest.TestCase):
    def test_BCELoss(self):
        targets = torch.round(torch.rand(1, 10, dtype=torch.double))
        predictions = torch.rand(1, 10, dtype=torch.double)
        cpp_loss = nanobind.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_batch_BCELoss(self):
        targets = torch.round(torch.rand(5, 10, dtype=torch.double))
        predictions = torch.rand(5, 10, dtype=torch.double)
        cpp_loss = nanobind.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_MSELoss(self):
        targets = torch.rand(1, 10, dtype=torch.double)
        predictions = torch.rand(1, 10, dtype=torch.double)
        cpp_loss = nanobind.MSELoss(predictions, targets)
        python_loss = nn.functional.mse_loss(
            predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_batch_MSELoss(self):
        targets = torch.rand(5, 10, dtype=torch.double)
        predictions = torch.rand(5, 10, dtype=torch.double)
        cpp_loss = nanobind.MSELoss(predictions, targets)
        python_loss = nn.functional.mse_loss(
            predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

#Again these are only relevant if I move away from autograd 
class TestDerivatives(unittest.TestCase):

    def test_ReLU_derivative(self):
        in_tensor = torch.rand(5, 3)
        in_tensor -= 0.5
        out = nanobind.differentiate_ReLU(in_tensor)
        for x in range(len(out)):
            for y in range(len(out[x])):
                if in_tensor[x][y] >= 0.0:
                    assert(out[x][y] == 1.0)
                else:
                    assert(out[x][y] == 0.0)

    def test_lin_derivative(self):
        lin = nn.Linear(2, 4).double()
        x = torch.rand(1, 2, dtype=torch.double)
        targets = torch.rand(1, 4, dtype=torch.double)
        for y in range(10):
            loss = torch.nn.functional.mse_loss(lin(x), targets)
           
            w = lin.weight.clone().detach()
            b = lin.bias.clone().detach()

            grad_w = 0.3*np.matmul(x.T, ((np.matmul(x, w.T)+b)-targets))
            grad_b = 0.3*np.matmul(x, w.T)+b-targets
            
            lin.weight = nn.Parameter((w.T-grad_w).T)
            lin.bias = nn.Parameter(b - grad_b)


# Test that the forward pass using cpp gives same res as forward pass using altmin
class TestGetCodes(unittest.TestCase):
    def check_equal(self, cpp_imp, python_imp):
        for x in range(len(cpp_imp)):
            for y in range(len(cpp_imp[x])):
                assert(abs(cpp_imp[x][y] - python_imp[x]
                           [y]) <= sys.float_info.epsilon)
                
    def test_get_codes(self):
        model = simpleNN(2, [4, 3], 1)
        model = get_mods(model)
        in_tensor = torch.rand(5, 2, dtype=torch.double)

        # Ignore Flatten for now
        model = model[1:]

        python_out, python_codes = get_codes(model, in_tensor)

        cpp_out, cpp_codes = cf_get_codes(model, in_tensor)

        self.check_equal(python_out.detach().numpy(), cpp_out.numpy())

        for x in range(len(cpp_codes)):
            self.check_equal(
                python_codes[x].detach().numpy(), cpp_codes[x].numpy())

#Test python and cpp update the codes the same
class TestUpdateCodes(unittest.TestCase):
    def test_update_codes(self):
        # Setup 
        targets = torch.round(torch.rand(5, 1, dtype=torch.double))
        code_one = torch.rand(5,4, dtype=torch.double)
        code_two = torch.rand(5,3, dtype=torch.double)
        codes_cpp = [code_one, code_two]
        codes_python = [code_one.detach().clone(), code_two.detach().clone()]
        n_iter = 1
        lr = 0.3
        mu = 0.003

        # Model setup
        model_cpp = simpleNN(2, [4,3],1)
        model_python = pickle.loads(pickle.dumps(model_cpp))

        # cpp
        model_cpp = get_mods(model_cpp)
        model_cpp = model_cpp[1:]
        cf_update_codes(codes_cpp, model_cpp, targets.detach(), nn.BCELoss(), mu=mu, lambda_c=0.0, n_iter=n_iter, lr=lr )
        cf_update_codes(codes_cpp, model_cpp, targets.detach(), nn.BCELoss(), mu=mu, lambda_c=0.0, n_iter=n_iter, lr=lr )

        # python
        model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler= lambda epoch: 1/2**(epoch//8))
        model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_python = model_python[1:]
        update_codes(codes_python, model_python, targets, nn.BCELoss(), mu, 0, n_iter, lr)
        update_codes(codes_python, model_python, targets, nn.BCELoss(), mu, 0, n_iter, lr)
        
        # Assert python and cpp give the same codes
        for x in range(len(codes_cpp)):
            check_equal(codes_python[x], codes_cpp[x], 10e12)

                
            
#Test cpp and python update the hidden weights the same
class TestUpdateHiddenWeights(unittest.TestCase):
    def test_update_hidden_weights(self):
        # Setup
        in_tensor = torch.rand(5, 2, dtype=torch.double)
        codes = [torch.rand(5,4, dtype=torch.double).detach(), torch.rand(5,3, dtype=torch.double).detach()]
        n_iter = 1
        lr = 0.008 

        # Model setup
        model_cpp = simpleNN(2, [4,3],1)
        model_python = pickle.loads(pickle.dumps(model_cpp))

        # cpp
        model_cpp = get_mods(model_cpp)
        model_cpp = model_cpp[1:]
        momentum_dict_cpp = store_momentums(model_cpp, init_vals = True)
        # Run twice to test parameters returned correctly after first execution
        cf_update_hidden_weights(model_cpp, in_tensor.detach(), codes, 0,n_iter, lr, momentum_dict_cpp, True)
        cf_update_hidden_weights(model_cpp, in_tensor.detach(), codes, 0,n_iter, lr, momentum_dict_cpp, False)

        # Python
        model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_python = model_python[1:]
        #Run twice to test parameters returned correctly after first execution
        update_hidden_weights_adam_(model_python, in_tensor, codes, lambda_w=0, n_iter=n_iter)
        update_hidden_weights_adam_(model_python, in_tensor, codes, lambda_w=0, n_iter=n_iter)
                 
        for x,m in enumerate(model_cpp):
            if isinstance(m, nn.Linear):
                #Assert model params are updated the same
                check_equal(model_python[x].weight.data, model_cpp[x].weight.data, 10e8)
                check_equal_bias(model_python[x].bias.data, model_cpp[x].bias.data, 10e8)
    



#Test cpp and python update the last layer the same
class TestUpdateLastLayer(unittest.TestCase):
    #Check updating last layer with adam gives same res in python and cpp
    def test_update_last_layer(self):
        # Setup
        in_tensor = torch.rand(10,5, dtype = torch.double)
        targets = torch.round(torch.rand(10, 1, dtype=torch.double))
        n_iter = 5
        lr = 0.008

        # Model setup
        model_cpp = simpleNN(2, [4,5],1)
        model_python = pickle.loads(pickle.dumps(model_cpp))

        # cpp
        model_cpp = get_mods(model_cpp)
        model_cpp = model_cpp[1:]
        momentum_dict_cpp = store_momentums(model_cpp, init_vals = True)
        criterion_cpp = 0
        # Run twice to test parameters returned correctly after first execution
        cf_update_last_layer(model_cpp, in_tensor.detach(), targets.detach(), criterion_cpp, n_iter, lr, momentum_dict_cpp, True )
        cf_update_last_layer(model_cpp, in_tensor.detach(), targets.detach(), criterion_cpp, n_iter, lr, momentum_dict_cpp, False )

        # Python
        model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_python = model_python[1:]
        # Run twice to test parameters returned correctly after first execution
        update_last_layer_(model_python[-1], in_tensor, targets, nn.BCELoss(), n_iter)
        update_last_layer_(model_python[-1], in_tensor, targets, nn.BCELoss(), n_iter)
        

        #Assert model params are updated the same
        check_equal(model_python[-1][1].weight.data, model_cpp[-1][1].weight.data, 10e8)
        check_equal_bias(model_python[-1][1].bias.data, model_cpp[-1][1].bias.data, 10e8)
        

        
# #Need to add proper tests to this 
class TestAdam(unittest.TestCase):
    
    def test_not_init(self):
        a = torch.rand(5,3)
        b = torch.rand(5,3)
        c = torch.rand(5,3)
        d = torch.rand(5,3)
        #print(a)
        #print(b)
        #print(c)
        nanobind.test_adam(a,b,c,d, False)
        #print(a)
        #print(b)
        #print(c)
        print("\n\n")

    def test_init(self):
        m_t = torch.zeros([5,3])
        v_t = torch.zeros([5,3])
        val = torch.rand(5,3)
        grad = torch.rand(5,3)
        # print("\n\n\n")
        # print(m_t)
        # print(val)
        nanobind.test_adam(m_t,v_t,val,grad, True)
        # print(m_t)
        # print(val)


#Test features of torch like autograd
class TestTorch(unittest.TestCase):
    # Assert all values are the same to tolerance of ulp
    def check_equal(self, cpp_imp, python_imp):
        for x in range(len(cpp_imp)):
            for y in range(len(cpp_imp[x])):
                assert(abs(cpp_imp[x][y] - python_imp[x][y]) <= sys.float_info.epsilon * 10)

    def test_autograd_example(self):
        input = torch.rand(2,2)
        x = input.detach().clone()
        x.requires_grad=True
        y = x + 2
        out = y.mean()
        out.backward()
        with torch.no_grad():
            x -= x.grad
        nanobind.autograd_example(input)
        self.check_equal(x,input)        

    def test_torch_lin(self):
        lin = nn.Linear(5,3)
        weight = lin.weight.data
        bias = lin.bias.data
        in_tensor = torch.rand(10,5)
        python_out = lin(in_tensor)
        cpp_out = torch.rand(10, 3)
        nanobind.test_tensor_lin(in_tensor, weight, bias, cpp_out)
        self.check_equal(python_out, cpp_out)

    def test_autograd(self):
        lin = nn.Linear(5,3)
        weight = lin.weight.data
        bias = lin.bias.data
        in_tensor = torch.rand(2,5)
        targets = torch.round(torch.rand(2,3)) 
        python_out = lin(in_tensor)
        cpp_out = torch.rand(10, 3)
        nanobind.test_autograd(in_tensor, weight, bias, targets)

        

if __name__ == '__main__':
    unittest.main()
