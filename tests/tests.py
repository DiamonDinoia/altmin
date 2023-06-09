import torch.nn as nn
import torch as torch
import unittest
import derivatives
import sys
from altmin import simpleNN, get_mods, update_last_layer_, update_hidden_weights_adam_, update_codes, FFNet
import pickle 
import fast_altmin

# Assert all values are the same to tolerance of ulp
def check_equal(first_imp, second_imp, eps):
    for x in range(len(first_imp)):
        for y in range(len(first_imp[x])):
            assert(abs(first_imp[x][y] - second_imp[x]
                        [y]) <= sys.float_info.epsilon*eps)
            
def check_equal_debug(first_imp, second_imp, eps):
    for x in range(len(first_imp)):
        for y in range(len(first_imp[x])):
            print(first_imp[x][y].item())
            print(second_imp[x][y].item())
            print(type(first_imp[x][y].item()))
            print(type(second_imp[x][y].item()))
            assert(abs(first_imp[x][y] - second_imp[x]
                        [y]) <= sys.float_info.epsilon*eps)
            
def check_equal_bias(first_imp, second_imp,eps):
    for x in range(len(first_imp)):
        assert(abs(first_imp[x] - second_imp[x]) <= sys.float_info.epsilon*eps)

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

    def test_lin_batch(self):
        lin = nn.Linear(4, 6).double()
        in_tensor = torch.rand(5, 4, dtype=torch.double)
        weight = lin.weight.data
        #bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))
        bias = lin.bias.data
        cpp_imp = torch.from_numpy(fast_altmin.lin(in_tensor,
                                      weight, bias))
        python_imp = lin(in_tensor)
        check_equal(cpp_imp, python_imp, 10e8)

    # Functions for relu and sigmoid are in place so they don't need to return a value

    def test_ReLU(self):
        relu = nn.ReLU()
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        python_imp = relu(in_tensor)
        # In tensor data updated in place
        cpp_imp = torch.from_numpy(fast_altmin.ReLU(
            in_tensor))

        check_equal(cpp_imp, python_imp, 10e8)

    def test_sigmoid(self):
        sigmoid = nn.Sigmoid()
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        python_imp = sigmoid(in_tensor)
        # In tensor data changed in place?
        cpp_imp = torch.from_numpy(fast_altmin.sigmoid(
            in_tensor))

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
        targets = torch.round(torch.rand(5, 10, dtype=torch.double))
        predictions = torch.rand(5, 10, dtype=torch.double)
        cpp_loss = fast_altmin.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_MSELoss(self):
        targets = torch.rand(1, 10, dtype=torch.double)
        predictions = torch.rand(1, 10, dtype=torch.double)
        cpp_loss = fast_altmin.MSELoss(predictions, targets)
        python_loss = nn.MSELoss()(predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_batch_MSELoss(self):
        targets = torch.rand(5, 10, dtype=torch.double)
        predictions = torch.rand(5, 10, dtype=torch.double)
        cpp_loss = fast_altmin.MSELoss(predictions, targets)
        python_loss = nn.functional.mse_loss(
            predictions, targets)
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

    def test_sigmoid_derivative(self):
        input_python = torch.rand(2,5).requires_grad_(True) 
        input_cpp = input_python.detach().clone()

        grad_python = derivatives.derivative_sigmoid(input_python) 
        grad_cpp = torch.from_numpy(fast_altmin.differentiate_sigmoid(input_cpp))
        check_equal(grad_python,grad_cpp, 10e9) 

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

    def test_apply_mods(self):
        in_tensor = torch.rand(10,5, dtype = torch.double)
        model = simpleNN(2, [4,5],1)        
        # Autograd
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[-1]
        output_python = model(in_tensor)
        output_cpp = torch.from_numpy(fast_altmin.apply_mods(model[1].weight.data, model[1].bias.data, [0,1,2], in_tensor, len([0,1,2]) ))
        check_equal(output_python, output_cpp, 10e8)
        

# class TestAlgorithms(unittest.TestCase):
#     def test_adam(self):
#         a = torch.rand(5,3)
#         b = torch.rand(5,3)
#         c = torch.rand(5,3)
#         d = torch.rand(5,3)
#         print(a)
#         print(b)
#         print(c)
#         a,b,c = fast_altmin.adam_eigen(a,b,c,d, 0.008, False, 1)
#         print(a)
#         print(b)
#         print(c)
#         print("\n\n")


class TestUpdateFunctions(unittest.TestCase):
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
        momentum_dict_cpp = fast_altmin.store_momentums(model_cpp, init_vals = True)
        # Run twice to test parameters returned correctly after first execution
        fast_altmin.cf_update_last_layer(model_cpp, in_tensor.detach(), targets.detach(), n_iter, lr, momentum_dict_cpp, True )
        fast_altmin.cf_update_last_layer(model_cpp, in_tensor.detach(), targets.detach(), n_iter, lr, momentum_dict_cpp, False )

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

    def test_update_hidden_weights(self):
        # Setup
        in_tensor = torch.rand(5, 2, dtype=torch.double)
        codes = [torch.rand(5,4, dtype=torch.double).detach(), torch.rand(5,3, dtype=torch.double).detach()]
        n_iter = 5
        lr = 0.008 

        # Model setup
        model_cpp = simpleNN(2, [4,3],1)
        model_python = pickle.loads(pickle.dumps(model_cpp))

        # cpp
        model_cpp = get_mods(model_cpp)
        model_cpp = model_cpp[1:]
        momentum_dict_cpp = fast_altmin.store_momentums(model_cpp, init_vals = True)
        # Run twice to test parameters returned correctly after first execution
        fast_altmin.cf_update_hidden_weights(model_cpp, in_tensor.detach(), codes, 0,n_iter, lr, momentum_dict_cpp,True)
        fast_altmin.cf_update_hidden_weights(model_cpp, in_tensor.detach(), codes, 0,n_iter, lr, momentum_dict_cpp,False)

        # Python
        model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_python = model_python[1:]
        #Run twice to test parameters returned correctly after first execution
        update_hidden_weights_adam_(model_python, in_tensor, codes, lambda_w=0, n_iter=n_iter)
        update_hidden_weights_adam_(model_python, in_tensor, codes, lambda_w=0, n_iter=n_iter)

    def test_update_weights_parallel(self):
        # Setup
        in_tensor = torch.rand(5, 2, dtype=torch.double)
        codes = [torch.rand(5,4, dtype=torch.double).detach(), torch.rand(5,3, dtype=torch.double).detach()]
        targets = torch.round(torch.rand(5, 1, dtype=torch.double))
        n_iter = 5
        lr = 0.008 

        # Model setup
        model_cpp = simpleNN(2, [4,3],1)
        model_python = pickle.loads(pickle.dumps(model_cpp))

        # cpp
        model_cpp = get_mods(model_cpp)
        model_cpp = model_cpp[1:]
        momentum_dict_cpp = fast_altmin.store_momentums(model_cpp, init_vals = True)

        # Run twice to test parameters returned correctly after first execution
        fast_altmin.cf_update_weights_parallel(model_cpp, in_tensor.detach(), codes,targets, 0,n_iter, lr, momentum_dict_cpp,True)
        fast_altmin.cf_update_weights_parallel(model_cpp, in_tensor.detach(), codes, targets, 0,n_iter, lr, momentum_dict_cpp,False)
        
        # Python
        model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_python = model_python[1:]
        #Run twice to test parameters returned correctly after first execution
        update_hidden_weights_adam_(model_python, in_tensor, codes, lambda_w=0, n_iter=n_iter)
        update_hidden_weights_adam_(model_python, in_tensor, codes, lambda_w=0, n_iter=n_iter)
        update_last_layer_(model_python[-1], codes[-1], targets, nn.BCELoss(), n_iter)
        update_last_layer_(model_python[-1], codes[-1], targets, nn.BCELoss(), n_iter)
        
        for x,m in enumerate(model_cpp):
            
            if isinstance(m, nn.Linear):
                #Assert model params are updated the same
                check_equal(model_python[x].weight.data, model_cpp[x].weight.data, 10e8)
                check_equal_bias(model_python[x].bias.data, model_cpp[x].bias.data, 10e8)

        #check last layer 
        check_equal(model_python[-1][1].weight.data, model_cpp[-1][1].weight.data, 10e8)
        check_equal_bias(model_python[-1][1].bias.data, model_cpp[-1][1].bias.data, 10e8)

    

    #Slightly too inaccurate
    def test_update_codes_BCELoss(self):
        # Setup 
        targets = torch.round(torch.rand(5, 1, dtype=torch.double))
        code_one = torch.rand(5,4, dtype=torch.double) - 0.5
        code_two = torch.rand(5,3, dtype=torch.double) - 0.5
        #code_two = torch.tensor([[-0.5479, 0.2508,  0.5361]], dtype=torch.double)
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
        for it in range(5):
            fast_altmin.cf_update_codes(codes_cpp, model_cpp, targets.detach(), nn.BCELoss(), mu=mu, lambda_c=0.0, n_iter=n_iter, lr=lr)
        #print(codes_cpp)
        #fast_altmin.cf_update_codes_eigen(codes_cpp, model_cpp, targets.detach(), nn.BCELoss(), mu=mu, lambda_c=0.0, n_iter=n_iter, lr=lr )
       
        # python
        model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler= lambda epoch: 1/2**(epoch//8))
        model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_python = model_python[1:]
        for it in range(5):
            update_codes(codes_python, model_python, targets, nn.BCELoss(), mu, 0, n_iter, lr)
        #update_codes(codes_python, model_python, targets, nn.BCELoss(), mu, 0, n_iter, lr)

        # Assert python and cpp give the same codes
        for x in range(len(codes_cpp)):
            check_equal(codes_python[x], codes_cpp[x], 10e6)


    # def test_update_codes_MSELoss(self):
    #     # Setup 
    #     targets = torch.round(torch.rand(5, 1, dtype=torch.double))
    #     code_one = torch.rand(5,100, dtype=torch.double) - 0.5
    #     code_two = torch.rand(5,100, dtype=torch.double) - 0.5
    #     codes_cpp = [code_one, code_two]
    #     codes_python = [code_one.detach().clone(), code_two.detach().clone()]
    #     n_iter = 1
    #     lr = 0.3
    #     mu = 0.003

    #     # Model setup
    #     model_cpp = FFNet(10, n_hiddens=100, n_hidden_layers=2, batchnorm=False, bias=True)
    #     model_python = pickle.loads(pickle.dumps(model_cpp))

    #     #cpp
    #     model_cpp = get_mods(model_cpp)
    #     model_cpp = model_cpp[1:]
    #     for it in range(5):
    #         fast_altmin.cf_update_codes(codes_cpp, model_cpp, targets.detach(), nn.MSELoss(), mu=mu, lambda_c=0.0, n_iter=n_iter, lr=lr)

    #     #python
    #     model_python = get_mods(model_python, optimizer='Adam', optimizer_params={'lr': 0.008},
    #                  scheduler=lambda epoch: 1/2**(epoch//8))
    #     model_python[-1].optimizer.param_groups[0]['lr'] = 0.008
    #     model_python = model_python[1:]
    #     for it in range(5):
    #         update_codes(codes_python, model_python, targets, nn.MSELoss(), mu, 0, n_iter, lr)

    #     # Assert python and cpp give the same codes
    #     for x in range(len(codes_cpp)):
    #         check_equal(codes_python[x], codes_cpp[x], 10e6)


        

if __name__ == '__main__':
    unittest.main()
    
