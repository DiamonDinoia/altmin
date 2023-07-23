from test_helper_methods import check_equal, check_equal_weights_and_bias, check_equal_weights_and_bias4d, check_equal4d
import torch.nn as nn 
import torch 
import fast_altmin 
import unittest 
import sys
from altmin import simpleNN, get_mods, get_codes, update_codes, FFNet, update_hidden_weights_adam_, update_last_layer_, load_dataset, LeNet
from log_approximator import LogApproximator

class TestAltminFunctionFeedForward(unittest.TestCase):

    def test_forward(self):
        # data 
        data = torch.rand(7,2,dtype = torch.double)
        
        # model init
        model = simpleNN(2, [4,3],1)
        model = get_mods(model)
        model = model[1:]
        neural_network = fast_altmin.VariantNeuralNetworkBCE()
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()

        # python and cpp implementation
        output_python, codes_python = get_codes(model, data)
        output_cpp = neural_network.get_codes(data, True)
        codes_cpp = neural_network.return_codes() 

        #Check output 
        check_equal(output_python, output_cpp, 10e6)
        #check codes
        for x in range(len(codes_python)):
            check_equal(codes_python[x], codes_cpp[x], 10e6)

            
    def test_update_codes_BCELoss(self):
        # data
        data = torch.rand(7,2,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        
        # model intialisation
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

        # python implementation
        for it in range(4):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, data)

            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)

        # cpp implementation
        for it in range(4):
            output_cpp = neural_network.get_codes(data, True)
 
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        
        #check codes
        for x in range(len(codes_python)):
            check_equal(codes_python[x], codes_cpp[x], 10e6)


    def test_update_codes_MSELoss(self):
        # data
        data = torch.rand(7,1,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        
        # model init
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.MSELoss()
        model= LogApproximator(5)
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        neural_network = fast_altmin.VariantNeuralNetworkMSE()
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()

        # python implementation
        for it in range(1):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, data)
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)

        # cpp implementation 
        for it in range(1):
            output_cpp = neural_network.get_codes(data, True)     
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        #check codes
        for x in range(len(codes_python)):        
           check_equal(codes_python[x], codes_cpp[x], 10e6)


    def test_update_codes_CrossEntropyLoss(self):
        # data
        data = torch.rand(4,10,dtype = torch.double)
        targets = torch.randint(0, 10, (4,))
        
        # Model init
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.CrossEntropyLoss()
        model = FFNet(10, n_hiddens=100, n_hidden_layers=2, batchnorm=False, bias=True).double()        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]

        neural_network = fast_altmin.VariantNeuralNetworkCrossEntropy()
        fast_altmin.create_model_class(model, neural_network, 4, 0)
        neural_network.construct_pairs()

        # Python implementation
        for it in range(1):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, data)
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)



        # cpp implementation
        targets = targets.double()
        targets = targets.reshape(1,len(targets))
        for it in range(1):   
            output_cpp = neural_network.get_codes(data, True)
            neural_network.update_codes(targets)
        codes_cpp = neural_network.return_codes() 

        #check codes
        for x in range(len(codes_python)):
            check_equal(codes_python[x], codes_cpp[x], 10e6)


    def test_update_weights_BCELoss(self):
        # data
        data = torch.rand(7,2,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        
        # model init
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
        
        # Python imp
        for it in range(1):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, data)

            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)
            update_hidden_weights_adam_(model, data, codes_python, lambda_w=0, n_iter=n_iter)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.BCELoss(), n_iter)

        
        # cpp imp
        for it in range(1):
            output_cpp = neural_network.get_codes(data, True)  
            neural_network.update_codes(targets)
            neural_network.update_weights_parallel(data, targets)
        
        
        weights = neural_network.return_weights()
        biases = neural_network.return_biases()
        check_equal_weights_and_bias(model, weights, biases, 10e9)


    def test_update_weights_MSELoss(self):
        # data
        data = torch.rand(7,1,dtype = torch.double)
        targets = torch.round(torch.rand(7, 1, dtype=torch.double))
        
        # model init
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.MSELoss()
        model= LogApproximator(5)
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        neural_network = fast_altmin.VariantNeuralNetworkMSE()
        fast_altmin.create_model_class(model, neural_network, 7, 0)
        neural_network.construct_pairs()


        # python imp
        for it in range(1):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, data)
            
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)
            update_hidden_weights_adam_(model, data, codes_python, lambda_w=0, n_iter=n_iter)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.MSELoss(), n_iter)

        # cpp implementation 
        for it in range(1):
            output_cpp = neural_network.get_codes(data, True)
            neural_network.update_codes(targets)
            neural_network.update_weights_parallel(data, targets)
        
        
        weights = neural_network.return_weights()
        biases = neural_network.return_biases()
        check_equal_weights_and_bias(model, weights, biases, 10e9)


    def test_update_weights_CrossEntropyLoss(self):
        # data
        data = torch.rand(4,10,dtype = torch.double)
        targets = torch.randint(0, 10, (4,))
        
        # model init
        n_iter = 1
        lr = 0.3
        mu = 0.003
        criterion = nn.CrossEntropyLoss()
        model = FFNet(10, n_hiddens=100, n_hidden_layers=2, batchnorm=False, bias=True).double()        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008
        model = model[1:]
        neural_network = fast_altmin.VariantNeuralNetworkCrossEntropy()
        fast_altmin.create_model_class(model, neural_network, 4, 0)
        neural_network.construct_pairs()

        # python imp
        for it in range(1):
            with torch.no_grad():
                output_python, codes_python = get_codes(model, data)
            codes_python = update_codes(codes_python, model, targets, criterion, mu, 0, n_iter, lr)
            update_hidden_weights_adam_(model, data, codes_python, lambda_w=0, n_iter=n_iter)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.CrossEntropyLoss(), n_iter)

        # cpp imp
        targets_cpp = targets.double()
        targets_cpp = targets.reshape(1,len(targets))
        for it in range(1):   
            output_cpp = neural_network.get_codes(data, True)              
            neural_network.update_codes(targets_cpp)
            neural_network.update_weights_parallel(data, targets_cpp)
        

        weights = neural_network.return_weights()
        biases = neural_network.return_biases()
        check_equal_weights_and_bias(model, weights, biases, 10e10)




class TestCNN(unittest.TestCase):

    def test_cnn_forward(self):
        #Load data
        batch_size = 5
        train_loader, test_loader, n_inputs = load_dataset('mnist', batch_size,  conv_net=True,
                                                       data_augmentation=True)
        window_size = train_loader.dataset.data[0].shape[0]
        num_input_channels = 1    
        data, _ = next(iter(train_loader))
        data = data.double()

        #Create model
        model = LeNet(num_input_channels=num_input_channels, window_size=window_size, bias=True).double()
        model = get_mods(model)

        model_cnn = model[0:4]
        model_nn = model[4:]

        convolutional_neural_network = fast_altmin.VariantCNNCrossEntropy()
        C_out, H_out, W_out  = fast_altmin.create_CNN(model_cnn, convolutional_neural_network, batch_size, num_input_channels, data.shape[2],data.shape[3])
        neural_network = fast_altmin.VariantNeuralNetworkCrossEntropy()
        fast_altmin.create_model_class(model_nn, neural_network, batch_size, 0)
        neural_network.construct_pairs()
        lenet = fast_altmin.LeNetCrossEntropy(batch_size, 16, 4)
        lenet.AddCNN(convolutional_neural_network)
        lenet.AddFeedForwardNN(neural_network)

        #Generate input to cpp
        input_list = []
        for n in range(data.shape[0]):
            tmp = []
            for c in range(data.shape[1]):
                tmp.append(data[n,c].numpy())
            input_list.append(tmp)


        cpp_imp = lenet.GetCodesLeNet(input_list, True)
        cpp_cnn_codes = lenet.ReturnCodesFromCNNLeNet()
        cpp_nn_codes = lenet.ReturnCodesFromFFLeNet()
        
        py_imp, codes_python = get_codes(model, data)
        
        check_equal(py_imp, cpp_imp, 10e9)
        y = 0
        for x in range(len(cpp_cnn_codes)):
            check_equal4d(codes_python[x], cpp_cnn_codes[x], 10e9)
            y = x
        for x in range(len(cpp_nn_codes)):
            check_equal(codes_python[x+y+1], cpp_nn_codes[x], 10e9)
        

    def test_cnn_update_codes(self):
        #Load data
        batch_size = 5
        train_loader, test_loader, n_inputs = load_dataset('mnist', batch_size,  conv_net=True,
                                                       data_augmentation=True)
        window_size = train_loader.dataset.data[0].shape[0]
        num_input_channels = 1    
        data, targets = next(iter(train_loader))
        data = data.double()
        

        #Create model
        model = LeNet(num_input_channels=num_input_channels, window_size=window_size, bias=True).double()
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008

        #print(model)
        model_cnn = model[0:4]
        model_nn = model[4:]

        convolutional_neural_network = fast_altmin.VariantCNNCrossEntropy()
        C_out, H_out, W_out  = fast_altmin.create_CNN(model_cnn, convolutional_neural_network, batch_size, num_input_channels, data.shape[2],data.shape[3])
        neural_network = fast_altmin.VariantNeuralNetworkCrossEntropy()
        fast_altmin.create_model_class(model_nn, neural_network, batch_size, 0)
        neural_network.construct_pairs()
        lenet = fast_altmin.LeNetCrossEntropy(batch_size, 16, 4)
        lenet.AddCNN(convolutional_neural_network)
        lenet.AddFeedForwardNN(neural_network)

        
        #Generate input to cpp
        input_list = []
        for n in range(data.shape[0]):
            tmp = []
            for c in range(data.shape[1]):
                tmp.append(data[n,c].numpy())
            input_list.append(tmp)
        
       
        cpp_imp = lenet.GetCodesLeNet(input_list, True)

        with torch.no_grad():
            py_imp, codes_python = get_codes(model, data)
        
        codes_python = update_codes(codes_python, model, targets, nn.CrossEntropyLoss(), 0.003, 0, 1, 0.3)
        
        targets = targets.double()
        targets = targets.reshape(1,len(targets))
        
        lenet.UpdateCodesLeNet(targets)
        cpp_cnn_codes = lenet.ReturnCodesFromCNNLeNet()
        cpp_nn_codes = lenet.ReturnCodesFromFFLeNet()

        for x in range(2):
            check_equal4d(codes_python[x], cpp_cnn_codes[x], 10e9)

        for x in range(2):
            check_equal(codes_python[x+2], cpp_nn_codes[x], 10e9)



    def test_cnn_update_weights(self):
        #Load data
        batch_size = 200
        train_loader, test_loader, n_inputs = load_dataset('mnist', batch_size,  conv_net=True,
                                                       data_augmentation=True)
        window_size = train_loader.dataset.data[0].shape[0]
        num_input_channels = 1    
        data, targets = next(iter(train_loader))
        data = data.double()
        

        #Create model
        model = LeNet(num_input_channels=num_input_channels, window_size=window_size, bias=True).double()
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
        model[-1].optimizer.param_groups[0]['lr'] = 0.008

        #print(model)
        model_cnn = model[0:4]
        model_nn = model[4:]


        convolutional_neural_network = fast_altmin.VariantCNNCrossEntropy()
  
        C_out, H_out, W_out  = fast_altmin.create_CNN(model_cnn, convolutional_neural_network, batch_size, num_input_channels, data.shape[2],data.shape[3])
        convolutional_neural_network.construct_pairs()
        neural_network = fast_altmin.VariantNeuralNetworkCrossEntropy()
        fast_altmin.create_model_class(model_nn, neural_network, batch_size, 0)
        neural_network.construct_pairs()
        lenet = fast_altmin.LeNetCrossEntropy(batch_size, 16, 4)
        lenet.AddCNN(convolutional_neural_network)
        lenet.AddFeedForwardNN(neural_network)

        
        #Generate input to cpp
        input_list = []
        for n in range(data.shape[0]):
            tmp = []
            for c in range(data.shape[1]):
                tmp.append(data[n,c].numpy())
            input_list.append(tmp)
        
       
        cpp_imp = lenet.GetCodesLeNet(input_list, True)

        with torch.no_grad():
            py_imp, codes_python = get_codes(model, data)
        
        codes_python = update_codes(codes_python, model, targets, nn.CrossEntropyLoss(), 0.003, 0, 1, 0.3)
        for it in range(2):
            update_hidden_weights_adam_(model, data, codes_python, lambda_w=0, n_iter=1)
            update_last_layer_(model[-1], codes_python[-1], targets, nn.CrossEntropyLoss(), n_iter=1)
        
        targets = targets.double()
        targets = targets.reshape(1,len(targets))

        lenet.UpdateCodesLeNet(targets)

        for it in range(2):
        #data not yet used so dummy value
            lenet.UpdateWeightsLeNet(input_list,targets)
        
        weights_cnn = lenet.ReturnWeightsFromCNNLeNet()
        biases_cnn = lenet.ReturnBiasesFromCNNLeNet()
        weights_ff = lenet.ReturnWeightsFromFFLeNet()
        biases_ff = lenet.ReturnBiasesFromFFLeNet()
        
        check_equal_weights_and_bias4d(model[0:4], weights_cnn, biases_cnn, 10e12)

        check_equal_weights_and_bias(model[4:], weights_ff, biases_ff, 10e13)


if __name__ == '__main__':
    unittest.main()