from test_helper_methods import check_equal
import torch.nn as nn 
import torch 
import fast_altmin 
import unittest 

# Test the implementation of the layers

class TestLayers(unittest.TestCase):

    # Test the linear layer with a batch size of 1
    def test_lin(self):
        # data
        in_tensor = torch.rand(1, 2, dtype=torch.double)

        # linear layer
        lin = nn.Linear(2, 4).double()
        weight = lin.weight.data
        bias = lin.bias.data

        # cpp and python implementation
        cpp_imp = torch.from_numpy(fast_altmin.lin(in_tensor,
                                      weight, bias))
        python_imp = lin(in_tensor)

        check_equal(cpp_imp, python_imp, 10e8)


    # Test the linear layer with a batch size greater than 1
    def test_lin_batch(self):
        # data
        in_tensor = torch.rand(5, 4, dtype=torch.double)

        # linear layer
        lin = nn.Linear(4, 6).double()
        weight = lin.weight.data
        bias = lin.bias.data

        # cpp and python implementation
        cpp_imp = torch.from_numpy(fast_altmin.lin(in_tensor,
                                      weight, bias))
        python_imp = lin(in_tensor)

        check_equal(cpp_imp, python_imp, 10e8)


    def test_ReLU(self):
        # data
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        
        # relu layer
        relu = nn.ReLU()
        
        # cpp and python implementation
        python_imp = relu(in_tensor)
        cpp_imp = fast_altmin.ReLU(in_tensor)
        check_equal(cpp_imp, python_imp, 10e8)


    def test_sigmoid(self):
        # data
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        
        # sigmoid layer 
        sigmoid = nn.Sigmoid()

        # cpp and python implementation
        python_imp = sigmoid(in_tensor)
        cpp_imp = fast_altmin.sigmoid(in_tensor)

        check_equal(cpp_imp, python_imp, 10e8)


    def test_conv2d(self):
        
        # data
        data = torch.rand(5,6,28,28, dtype = torch.double)
        
        # conv2d layer
        mod = nn.Conv2d(6, 16, 5, bias=True).double()
        
        # python imp
        py_imp = mod(data)
       
        # cpp implementation -- All the extra parts are normally handled by the neural network object but needs to be written out to unit test
        kernels = mod.weight.data 
        bias = mod.bias.data
        N = data.shape[0] 
        C_in = 6
        C_out = 16
        H_out =(int) (1 + ((data.shape[2]+2*mod.padding[0] - mod.dilation[0] *(mod.kernel_size[0]-1)-1)/mod.stride[0]))
        W_out = (int) (1 + ((data.shape[3]+2*mod.padding[1] - mod.dilation[1] *(mod.kernel_size[1]-1)-1)/mod.stride[1]))
        cpp_imp = torch.zeros(N,C_out,H_out,W_out)

        for n in range(N):
            for c_out in range(C_out):
                sum = torch.from_numpy(fast_altmin.conv2d(data[n,0], kernels[c_out, 0],bias[c_out], H_out, W_out))
                for c_in in range(1, C_in):
                    sum +=  fast_altmin.conv2d(data[n][c_in], kernels[c_out][c_in], 0.0, H_out, W_out)
            
                cpp_imp[n][c_out] = sum

        
        check_equal(py_imp, cpp_imp, 10e8)


    def test_maxpool2d(self):
        # data
        data = torch.rand(5,6,24,24, dtype = torch.double)
        
        # maxpool2d layer
        mod = nn.MaxPool2d(kernel_size = 2)
        
        # python implementation
        py_imp = mod(data)

        # cpp implementation -- All the extra parts are normally handled by the neural network object but needs to be written out to unit test
        kernel_size = mod.kernel_size
        stride = mod.stride
        N = data.shape[0] 
        C_out = data.shape[1]
        H_out =(int) (1 + ((data.shape[2]+2*mod.padding - mod.dilation *(mod.kernel_size-1)-1)/mod.stride))
        W_out = (int) (1 + ((data.shape[3]+2*mod.padding - mod.dilation *(mod.kernel_size-1)-1)/mod.stride))
        
        
        cpp_imp = torch.zeros(N,C_out,H_out,W_out)
        for n in range(N):
            for c in range(C_out):
                cpp_out = torch.from_numpy(fast_altmin.maxpool2d(data[n,c], kernel_size, stride, H_out, W_out))
                cpp_imp[n][c] = cpp_out
        

        check_equal(py_imp, cpp_imp, 10e8)

    
if __name__ == '__main__':
    unittest.main()