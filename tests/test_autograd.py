import torch.nn as nn
import torch as torch
import unittest
import derivatives
import sys
from altmin import simpleNN, get_mods, update_last_layer_, update_hidden_weights_adam_, update_codes
import pickle 

# Assert all values are the same to tolerance of ulp
def check_equal(first_imp, second_imp, eps):
    for x in range(len(first_imp)):
        for y in range(len(first_imp[x])):
            assert(abs(first_imp[x][y] - second_imp[x]
                        [y]) <= sys.float_info.epsilon*eps)
            
def check_equal_bias(first_imp, second_imp,eps):
    for x in range(len(first_imp)):
        assert(abs(first_imp[x] - second_imp[x]) <= sys.float_info.epsilon*eps)
            
class TestDerivatives(unittest.TestCase):
    def test_BCELoss_derivative(self):
        output = torch.rand(2,5).requires_grad_(True) 
        target = torch.round(torch.rand(2,5))
        loss = nn.BCELoss()(output, target)
        loss.backward()
        grad_autograd = output.grad

        grad_no_autograd = derivatives.derivative_BCELoss(output,target)
        #Correct to 8 d.p to need to improve but works for now 
        check_equal(grad_autograd,grad_no_autograd, 10e08)

    def test_MSELoss_derivative(self):
        output = torch.rand(2,5).requires_grad_(True) 
        target = torch.rand(2,5)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        grad_autograd = output.grad
        #print(grad_autograd)
        grad_no_autograd = derivatives.derivative_MSELoss(output,target)
        #print(grad_no_autograd)
        #Correct to 8 d.p to need to improve but works for now 
        check_equal(grad_autograd,grad_no_autograd, 10e08)

    def test_functional_MSELoss_derivative(self):
        output = torch.rand(2,5).requires_grad_(True) 
        target = torch.rand(2,5)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        grad_autograd = output.grad
        #print(grad_autograd)
        grad_no_autograd = derivatives.derivative_MSELoss(output,target)
        #print(grad_no_autograd)
        #Correct to 8 d.p to need to improve but works for now 
        check_equal(grad_autograd,grad_no_autograd, 10e08)

    def test_sigmoid_derivative(self):
        input = torch.rand(2,5).requires_grad_(True) 
        output = nn.Sigmoid()(input)
        target = torch.round(torch.rand(2,5))
        loss = nn.BCELoss()(output, target)
        loss.backward()
        grad_autograd = input.grad

        dL_doutput = derivatives.derivative_BCELoss(output,target)
        doutput_dinput = derivatives.derivative_sigmoid(input) 
        grad_no_autograd = dL_doutput * doutput_dinput
        #Correct to 8 d.p to need to improve but works for now 
        check_equal(grad_autograd,grad_no_autograd, 10e08)

    def test_relu_derivative(self):
        input = torch.rand(2,5).requires_grad_(True) 
        output = nn.ReLU()(input)
        target = torch.round(torch.rand(2,5))
        loss = nn.BCELoss()(output, target)
        loss.backward()
        grad_autograd = input.grad

        dL_doutput = derivatives.derivative_BCELoss(output,target)
        doutput_dinput = derivatives.derivative_relu(input) 
        grad_no_autograd = dL_doutput * doutput_dinput
        #Correct to 8 d.p to need to improve but works for now 
        check_equal(grad_autograd,grad_no_autograd, 10e9)

    def test_linear_derivatives(self):
        #Output needs to be between 0 and 1 so need sigmoid layer as well
        model = nn.Sequential(nn.Linear(5,3), nn.Sigmoid())
        input = torch.rand(2,5).requires_grad_(True)
        output = model(input) 
        target = torch.round(torch.rand(2,3))
        loss = nn.BCELoss()(output, target)
        loss.backward()
        W_grad_autograd = model[0].weight.grad
        b_grad_autograd = model[0].bias.grad

        dL_doutput = derivatives.derivative_BCELoss(output,target)
        doutput_dsigmoid = derivatives.derivative_sigmoid(model[0](input)) 
        dsigmoid_dW, dsigmoid_db = derivatives.derivative_linear(input)

        dL_dW = torch.matmul( torch.transpose(dL_doutput * doutput_dsigmoid, 1, 0), dsigmoid_dW)
        dL_db = torch.sum(dL_doutput * doutput_dsigmoid * dsigmoid_db,0)

        W_grad_no_autograd = dL_dW
        b_grad_no_autograd = dL_db
        
        check_equal(W_grad_autograd, W_grad_no_autograd, 10e08)
        check_equal_bias(b_grad_autograd, b_grad_no_autograd, 10e08)


class TestMethods(unittest.TestCase):
    def test_update_last_layer(self):
        # Setup
        in_tensor = torch.rand(10,5, dtype = torch.double)
        targets = torch.round(torch.rand(10, 1, dtype=torch.double))
        n_iter = 1
        lr = 0.008

        # Model setup
        model_autograd = simpleNN(2, [4,5],1)
        model_no_autograd = pickle.loads(pickle.dumps(model_autograd))

        
        # Autograd
        model_autograd = get_mods(model_autograd, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_autograd[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_autograd = model_autograd[1:]
        # Run twice to test parameters returned correctly after first execution
        update_last_layer_(model_autograd[-1], in_tensor, targets, nn.BCELoss(), n_iter)
        
        # No autograd 
        model_no_autograd = get_mods(model_no_autograd, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_no_autograd[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_no_autograd = model_no_autograd[1:]
        W,b = derivatives.update_last_layer_no_autograd(model_no_autograd[-1], in_tensor, targets, nn.BCELoss(), n_iter)
        
        check_equal(model_autograd[-1][1].weight.data, W, 10e08)
        check_equal_bias(model_autograd[-1][1].bias.data, b, 10e08)
    
    def test_update_hidden_weights(self):
        # Setup
        in_tensor = torch.rand(5, 2, dtype=torch.double)
        codes = [torch.rand(5,4, dtype=torch.double).detach(), torch.rand(5,3, dtype=torch.double).detach()]
        n_iter = 1
        lr = 0.008 

        # Model setup
        model_autograd = simpleNN(2, [4,3],1)
        model_no_autograd = pickle.loads(pickle.dumps(model_autograd))

        # Autograd
        model_autograd = get_mods(model_autograd, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_autograd[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_autograd = model_autograd[1:]
        # Run twice to test parameters returned correctly after first execution
        update_hidden_weights_adam_(model_autograd, in_tensor, codes, lambda_w=0, n_iter=n_iter)
        
        # No autograd 
        model_no_autograd = get_mods(model_no_autograd, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_no_autograd[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_no_autograd = model_no_autograd[1:]
        derivatives.update_hidden_weights_no_autograd(model_no_autograd, in_tensor, codes, lambda_w=0, n_iter=n_iter)

        for x,m in enumerate(model_no_autograd):
            if isinstance(m, nn.Linear):
                #Assert model params are updated the same
                check_equal(model_autograd[x].weight.data, model_no_autograd[x].weight.data, 10e8)
                check_equal_bias(model_autograd[x].bias.data, model_no_autograd[x].bias.data, 10e8)

    def test_update_codes(self):
        # Setup 
        targets = torch.round(torch.rand(5, 1, dtype=torch.double))
        code_one = torch.rand(5,4, dtype=torch.double)
        code_two = torch.rand(5,3, dtype=torch.double)
        codes_autograd = [code_one, code_two]
        codes_no_autograd = [code_one.detach().clone(), code_two.detach().clone()]
        n_iter = 1
        lr = 0.3
        mu = 0.003

        # Model setup
        model_autograd = simpleNN(2, [4,3],1)
        model_no_autograd = pickle.loads(pickle.dumps(model_autograd))

        # Autograd
        model_autograd = get_mods(model_autograd, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_autograd[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_autograd = model_autograd[1:]
        update_codes(codes_autograd, model_autograd, targets, nn.BCELoss(), mu, 0, n_iter, lr)

        # No autograd 
        model_no_autograd = get_mods(model_no_autograd, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model_no_autograd[-1].optimizer.param_groups[0]['lr'] = 0.008
        model_no_autograd = model_no_autograd[1:]
        derivatives.update_codes_no_autograd(codes_no_autograd, model_no_autograd, targets, nn.BCELoss(), mu, 0, n_iter, lr)
      
        
        # Assert python and cpp give the same codes
        for x in range(len(codes_autograd)):
            check_equal(codes_autograd[x], codes_no_autograd[x], 10e8)
    

        

if __name__ == '__main__':
    unittest.main()
    
