import torch.nn as nn
import torch as torch
import unittest
import derivatives
import sys
from altmin import simpleNN, get_mods, update_last_layer_, update_hidden_weights_adam_, update_codes, FFNet, get_codes, load_dataset
import pickle 
import fast_altmin

def check_equal(first_imp, second_imp, eps):
    for x in range(len(first_imp)):
        for y in range(len(first_imp[x])):
            assert(abs(first_imp[x][y] - second_imp[x]
                        [y]) <= sys.float_info.epsilon*eps)
            

def conv_model_to_class(model):
    neural_network = fast_altmin.NeuralNetwork()
    create_model_class(model, neural_network)
    return neural_network
    
def create_model_class(model, neural_network, has_codes = True):
    #print(model)
    for mod in model:
        #print(mod)
        if isinstance(mod, nn.ReLU):
            layer = fast_altmin.Layer(fast_altmin.Layer.relu)
            neural_network.push_back_layer(layer)
        elif isinstance(mod, nn.Linear):
            layer = fast_altmin.Layer(mod.weight.size(0), mod.weight.size(1), has_codes)
            layer.initialise_matrices(mod.weight.data, mod.bias.data)
            neural_network.push_back_layer(layer)
        elif isinstance(mod, nn.Sigmoid):
            layer = fast_altmin.Layer(fast_altmin.Layer.sigmoid)
            neural_network.push_back_layer(layer)
        elif isinstance(mod, nn.Sequential):
            layer = create_model_class(mod, neural_network, False)
        else:
            print("layer not imp yet")

def get_codes(model_mods, inputs):
    '''Runs the architecture forward using `inputs` as inputs, and returns outputs and intermediate codes
    '''
    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    # As codes only return outputs of linear layers
    codes = []
    for m in model_mods:
        x = m(x)
   
        if hasattr(m, 'has_codes') and getattr(m, 'has_codes'):
            #print(m)
            codes.append(x.clone())
    # Do not include output of very last linear layer (not counted among codes)
    return x, codes

class TestUpdateFunctions(unittest.TestCase):
    def test_forward(self):
        in_tensor = torch.rand(4,2,dtype = torch.double)
        model = simpleNN(2, [4,3],1)
        model = get_mods(model)
        model = model[1:]
        output_python, codes_python = get_codes(model, in_tensor)

        neural_network = conv_model_to_class(model)
        #neural_network.print_info()
        
        output_cpp = neural_network.get_codes(in_tensor)
        codes_cpp = neural_network.return_codes() 


        #Check output 
        check_equal(output_python, output_cpp, 10e6)
        print(codes_python)
        print(codes_cpp)
        for x in range(len(codes_python)):
            check_equal(codes_python[x], codes_cpp[x], 10e6)

    def test_update_codes_BCELoss(self):
        # Setup 
        inputs = torch.rand(4,2,dtype = torch.double)
        targets = torch.round(torch.rand(5, 1, dtype=torch.double))
  
        n_iter = 1
        lr = 0.3
        mu = 0.003

        # Model setup
        model_cpp = simpleNN(2, [4,3],1)
        model_cpp = get_mods(model_cpp)
        model_cpp = model_cpp[1:]
        

        output_cpp, codes_cpp = get_codes(model_cpp, inputs)

        neural_network = conv_model_to_class(model_cpp)

        output_nn = neural_network.get_codes(inputs)
        codes_nn = neural_network.return_codes() 

        #neural_network.print_info()
        
        check_equal(output_cpp, output_nn, 10e6)
        for x in range(len(codes_cpp)):
            check_equal(codes_nn[x], codes_cpp[x], 10e6)

        for it in range(1):
            fast_altmin.cf_update_codes(codes_cpp, model_cpp, targets.detach(), nn.BCELoss(), mu=mu, lambda_c=0.0, n_iter=n_iter, lr=lr)
        
        print("--------------------------------------------------------------------\n")

        #neural_network.print_info()

        neural_network.update_codes(targets)
        codes_nn = neural_network.return_codes() 
        
        print(codes_nn)
        print(codes_cpp)
        for x in range(len(codes_cpp)):
            check_equal(codes_nn[x], codes_cpp[x], 10e6)

    def test_update_weights_BCELoss(self):
        # Setup 
        inputs = torch.rand(4,2,dtype = torch.double)
        targets = torch.round(torch.rand(5, 1, dtype=torch.double))

        n_iter = 1
        lr = 0.3
        mu = 0.003

        # Model setup
        model_cpp = simpleNN(2, [4,3],1)
        model_cpp = get_mods(model_cpp)
        model_cpp = model_cpp[1:]
        

        output_cpp, codes_cpp = get_codes(model_cpp, inputs)

        neural_network = conv_model_to_class(model_cpp)

        output_nn = neural_network.get_codes(inputs)
        codes_nn = neural_network.return_codes() 

        #neural_network.print_info()
        
        check_equal(output_cpp, output_nn, 10e6)
        for x in range(len(codes_cpp)):
            check_equal(codes_nn[x], codes_cpp[x], 10e6)

        for it in range(1):
            fast_altmin.cf_update_codes(codes_cpp, model_cpp, targets.detach(), nn.BCELoss(), mu=mu, lambda_c=0.0, n_iter=n_iter, lr=lr)
        

        #neural_network.print_info()

        neural_network.update_codes(targets)
        codes_nn = neural_network.return_codes() 

   
        lr = 0.008
        momentum_dict_cpp = fast_altmin.store_momentums(model_cpp, init_vals = True)
        for it in range(1):
            fast_altmin.cf_update_weights_parallel(model_cpp, inputs.detach(), codes_cpp,targets, 0,n_iter, lr, momentum_dict_cpp,True,nn.BCELoss())

        print("a")
        print("---------------------------------------------------------------------------")
        neural_network.update_weights(inputs, targets)
        print("b")

        for x in range(len(codes_cpp)):
            check_equal(codes_nn[x], codes_cpp[x], 10e6)
  
if __name__ == '__main__':
    unittest.main()
    
