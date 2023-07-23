from test_helper_methods import check_equal, check_equal_bias
import torch.nn as nn 
import torch 
import fast_altmin 
import unittest 
from altmin import load_dataset, get_mods, LeNet

#Test the implementation of the derivatives

# Linear not tested as linear derivative is just equal to the weights or the inputs so that doesn't need a function

class TestDerivatives(unittest.TestCase):

    def test_relu_derivative(self):
        # data
        data = torch.rand(2,5)
        data -=0.5
        data.requires_grad_(True) 

        # python and cpp implementation
        out = nn.ReLU()(data)
        out.backward(torch.ones(out.shape))
        grad_python = data.grad
        
        grad_cpp = torch.from_numpy(fast_altmin.differentiate_ReLU(data))

        check_equal(grad_python,grad_cpp, 10e9) 

    
    def test_sigmoid_derivative(self):
        # data
        data = torch.rand(2,5).requires_grad_(True) 
        
        # python and cpp implementation
        out = nn.Sigmoid()(data)
        out.backward(torch.ones(out.shape))
        grad_python = data.grad

        grad_cpp = torch.from_numpy(fast_altmin.differentiate_sigmoid(data))

        check_equal(grad_python,grad_cpp, 10e9) 

    def test_MSELoss_derivative(self):
        # data
        output = torch.rand(2,5).requires_grad_(True) 
        target = torch.rand(2,5)
        
        # python and cpp implementation
        out = nn.MSELoss()(output, target)
        out.backward(torch.ones(out.shape))
        grad_python = output.grad
        
        grad_cpp = torch.from_numpy(fast_altmin.differentiate_MSELoss(output,target))
        
        check_equal(grad_python,grad_cpp, 10e8) 


    def test_BCELoss_derivative(self):
        # data
        output = torch.rand(2,5).requires_grad_(True) 
        target = torch.round(torch.rand(2,5))

        # python and cpp implementation
        out = nn.BCELoss()(output, target)
        out.backward(torch.ones(out.shape))
        grad_python = output.grad

        grad_cpp = torch.from_numpy(fast_altmin.differentiate_BCELoss(output,target))
        
        check_equal(grad_python,grad_cpp, 10e9) 


    def test_CrossEntropyLoss_derivative(self):
        # data
        output = torch.rand(4,5).requires_grad_(True) 
        target = torch.tensor([2,1,4,0])
        num_classes = 5
        
        # python and cpp implementation
        out = nn.CrossEntropyLoss()(output, target)
        out.backward(torch.ones(out.shape))
        grad_python = output.grad
        
        grad_cpp = fast_altmin.differentiate_CrossEntropyLoss(output,target, num_classes)
        
        check_equal(grad_python,grad_cpp, 10e9) 


    def test_relu_cnn_derivative(self):
        # data
        batch_size = 5
        train_loader, test_loader, n_inputs = load_dataset('mnist', batch_size,  conv_net=True,
                                                       data_augmentation=True)
        window_size = train_loader.dataset.data[0].shape[0]
        num_input_channels = 1    
        data, _ = next(iter(train_loader))
        data = data.double()

        # cpp imp
        N = data.shape[0]
        C_out = data.shape[1]
        H_out = data.shape[2]
        W_out = data.shape[3]

        cpp_grad = torch.zeros(N,C_out,H_out,W_out)
        for n in range(N):
            for c in range(C_out):
                cpp_out = torch.from_numpy(fast_altmin.differentiate_ReLU(data[n,c]))
                cpp_grad[n][c] = cpp_out

        # py imp
        data.requires_grad_(True)
        out = nn.ReLU()(data)
        out.backward(torch.ones(data.shape))
        py_grad = data.grad

        check_equal(py_grad, cpp_grad, 10e9)


    def test_maxpool2d_derivative(self):
        # data
        batch_size = 5
        train_loader, test_loader, n_inputs = load_dataset('mnist', batch_size,  conv_net=True,
                                                       data_augmentation=True, shuffle = False)
        window_size = train_loader.dataset.data[0].shape[0]
        num_input_channels = 1    
        data, _ = next(iter(train_loader))
        data = data.double()

        # maxpool 2d
        mod = nn.MaxPool2d(kernel_size = 2)
        kernel_size = mod.kernel_size
        stride = mod.stride


        # cpp imp
        H =(int) (1 + ((data.shape[2]+2*mod.padding - mod.dilation *(mod.kernel_size-1)-1)/mod.stride))
        W = (int) (1 + ((data.shape[3]+2*mod.padding - mod.dilation *(mod.kernel_size-1)-1)/mod.stride))


        N = data.shape[0]
        C_out = data.shape[1]
        H_out = data.shape[2]
        W_out = data.shape[3]

        cpp_grad = torch.zeros(N,C_out,H_out,W_out)
        for n in range(N):
            for c in range(C_out):
                cpp_out = torch.from_numpy(fast_altmin.differentiate_maxpool2d(data[n,c], kernel_size, stride, H, W))
                cpp_grad[n][c] = cpp_out


        # py imp
        data.requires_grad_(True)
        out = mod(data)
        out.backward(torch.ones(out.shape))

        py_grad = data.grad

        check_equal(py_grad, cpp_grad, 10e9)


    def test_conv2d_multi_channel_derivative(self):
        # data
        batch_size = 5
        train_loader, test_loader, n_inputs = load_dataset('mnist', batch_size,  conv_net=True,
                                                       data_augmentation=True, shuffle = False)
        window_size = train_loader.dataset.data[0].shape[0]
        num_input_channels = 1    
        data, _ = next(iter(train_loader))
        data = data.double()
        
        # model
        model = LeNet(num_input_channels=num_input_channels, window_size=window_size, bias=True).double()
        model = get_mods(model)
        model_cnn = model[0:1]
        mod = model[2]
        data = model_cnn(data).detach()
        data.requires_grad_(True)

        # Check partial derivative with respect to input
        tmp = mod(data)
        tmp.retain_grad()
        out = nn.ReLU()(tmp)
        out.backward(torch.ones(out.shape))
      

        dL_dout = tmp.grad

        cpp_grad = torch.zeros(batch_size,6,24,24)
        for n in range(batch_size):
            for c in range(6):
                sum = fast_altmin.differentiate_conv2d(dL_dout[n][0],mod.weight.data[0][c],  data.shape[2],data.shape[3], False)
                for i in range(1,16):
                    sum += fast_altmin.differentiate_conv2d(dL_dout[n][i],mod.weight.data[i][c],  data.shape[2],data.shape[3], False)
                cpp_grad[n][c] = torch.from_numpy(sum)
     

        # sum = sum.reshape(1,1,data.shape[2], data.shape[3])

        check_equal(cpp_grad, data.grad, 10e9)

        # Check partial derivative with respect to weights
        import numpy as np
        dW = np.zeros((16,6,5,5))

        for n in range(batch_size):
            for c_out in range(16):
                sum = 0
                for c_in in range(6):
                    dW[c_out, c_in] += fast_altmin.differentiate_conv2d(data[n, c_in], dL_dout[n][c_out], mod.weight.data.shape[2], mod.weight.data.shape[3], True)
        


        check_equal(torch.from_numpy(dW), mod.weight.grad, 10e9)

        # Check partial derivative with respect to bias
        db = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for n in range(batch_size):
            for c in range(16):
                db[c] += dL_dout[n][c].sum().item()

        check_equal_bias(db, mod.bias.grad, 10e9)


if __name__ == '__main__':
    unittest.main()