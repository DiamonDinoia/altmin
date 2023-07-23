from test_helper_methods import check_equal
import torch.nn as nn 
import torch 
import fast_altmin 
import unittest 
import sys

# Test the loss functions of the criterion 

class TestCriterion(unittest.TestCase):
    def test_BCELoss(self):
        # data
        targets = torch.round(torch.rand(1, 10, dtype=torch.double))
        predictions = torch.rand(1, 10, dtype=torch.double)
        
        # python and cpp implementation
        cpp_loss = fast_altmin.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)
        
        eps = 10e6
        assert(abs(cpp_loss - python_loss.item()) <= sys.float_info.epsilon*eps)


    def test_batch_BCELoss(self):
        # data
        targets = torch.round(torch.rand(5000, 5, dtype=torch.double))
        predictions = torch.rand(5000, 5, dtype=torch.double)
        
        # python and cpp implementation
        cpp_loss = fast_altmin.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)
        
        eps = 10e6
        assert(abs(cpp_loss - python_loss.item()) <= sys.float_info.epsilon*eps)


    def test_MSELoss(self):
        # data
        targets = torch.rand(1, 10, dtype=torch.double)
        predictions = torch.rand(1, 10, dtype=torch.double)
        
        # python and cpp implementation
        cpp_loss = fast_altmin.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)
        
        cpp_loss = fast_altmin.MSELoss(predictions, targets)
        python_loss = nn.MSELoss()(predictions, targets)

        eps = 10e6
        assert(abs(cpp_loss - python_loss.item()) <= sys.float_info.epsilon*eps)

    def test_batch_MSELoss(self):
        # data
        targets = torch.rand(5000, 100, dtype=torch.double)
        predictions = torch.rand(5000, 100, dtype=torch.double)
        
        # python and cpp implementation
        cpp_loss = fast_altmin.MSELoss(predictions, targets)
        python_loss = nn.functional.mse_loss(predictions, targets)
        
        eps = 10e6
        assert(abs(cpp_loss - python_loss.item()) <= sys.float_info.epsilon*eps)

    def test_log_softmax(self):
        # data
        inputs = torch.rand(1,5, dtype=torch.double)
        
        # python and cpp implementation
        python_imp = nn.LogSoftmax(dim=1)(inputs)
        fast_altmin.log_softmax(inputs)

        check_equal(inputs, python_imp, 10e8)

    def test_batch_log_softmax(self):
        # data
        inputs = torch.rand(3,5, dtype=torch.double)

        # python and cpp implementation
        python_imp = nn.LogSoftmax(dim=1)(inputs)
        fast_altmin.log_softmax(inputs)
        check_equal(inputs, python_imp, 10e8)

    def test_negative_log_likelihood(self):
        # data
        inputs = torch.rand(3,5, dtype=torch.double)
        targets = torch.tensor([2,1,4])
        inputs = nn.LogSoftmax(dim=1)(inputs)

        # python and cpp implementation
        python_loss = nn.NLLLoss()(inputs, targets)
        cpp_loss = fast_altmin.negative_log_likelihood(inputs,targets)

        eps = 10e6
        assert(abs(cpp_loss - python_loss.item()) <= sys.float_info.epsilon*eps)

    def test_cross_entropy_loss(self):
        # data
        inputs = torch.rand(5000,5, dtype=torch.double)
        targets = torch.randint(0, 5, (5000,))

        # python and cpp implementation
        python_loss = nn.CrossEntropyLoss()(inputs,targets)

        targets = targets.double()
        cpp_loss = fast_altmin.cross_entropy_loss(inputs,targets)

        eps = 10e6
        assert(abs(cpp_loss - python_loss.item()) <= sys.float_info.epsilon*eps)

    
if __name__ == '__main__':
    unittest.main()