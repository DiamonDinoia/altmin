import torch.nn as nn
import numpy as np
import torch

import nanobind_get_codes
import nanobind_matrix_funcs
import nanobind_hello_world
import nanobind_pass_dict
import nanobind_layers
import nanobind_criterion
import unittest

from altmin import get_mods, get_codes, update_codes

from control_flow import cf_get_codes, cf_update_codes, test_loss

from models import simpleNN

import sys


# This test basically just checks nanobind is still working in it's simplest form


class TestHelloWorldOut(unittest.TestCase):
    def test_output_from_cpp(self):
        res = nanobind_hello_world.hello_world_out()
        assert(res == "Hi python from c++")

# This test checks eigen is working


class TestMatrixMultiplication(unittest.TestCase):
    def test_matrix_multiplication_in_cpp(self):
        a = np.asarray([[3, 2], [-1, 4]])
        b = np.asarray([[1, 0], [1, 1]])
        c = nanobind_matrix_funcs.matrix_multiplication(a, b)
        assert((c == np.asarray([[5, 2], [3, 4]])).all())

# These tests check the layers that have been implemented in c++.
# They all take tensors as input
# For non-linear layers the function performs the operation in place as a reference to the data is passed to avoid copying the data unecessarily
# The linear functions return a ndarray in row-major order so this needs to be converted to a tensor


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
        cpp_imp = nanobind_layers.lin(in_tensor,
                                      weight, bias)
        python_imp = lin(in_tensor).detach().numpy()
        self.check_equal(cpp_imp, python_imp)

    def test_lin_batch(self):
        lin = nn.Linear(4, 6).double()
        in_tensor = torch.rand(5, 4, dtype=torch.double)
        weight = lin.weight.data
        bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))
        cpp_imp = nanobind_layers.lin(in_tensor,
                                      weight, bias)
        python_imp = lin(in_tensor).detach().numpy()
        self.check_equal(cpp_imp, python_imp)

    # Functions for relu and sigmoid are in place so they don't need to return a value

    def test_ReLU(self):
        relu = nn.ReLU()
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        python_imp = relu(in_tensor)
        # In tensor data updated in place
        nanobind_layers.ReLU(
            in_tensor)

        in_tensor = in_tensor.numpy()
        self.check_equal(in_tensor, python_imp)

    def test_sigmoid(self):
        sigmoid = nn.Sigmoid()
        in_tensor = torch.rand(5, 10, dtype=torch.double)
        python_imp = sigmoid(in_tensor)
        # In tensor data changed in place?
        nanobind_layers.sigmoid(
            in_tensor)

        in_tensor = in_tensor.numpy()
        self.check_equal(in_tensor, python_imp)


# Testing the cpp implementation of BCEloss and MSEloss
# Also test the fuctions ability to calculate loss on batch input
# Again take reference to tensor as input so no data copied between
# Can't use epsilon as these diverge slightly around 8 decimal places but no a priority to fix rn but I'll come back to this
class TestCriterion(unittest.TestCase):
    def test_BCELoss(self):
        targets = torch.round(torch.rand(1, 10, dtype=torch.double))
        predictions = torch.rand(1, 10, dtype=torch.double)
        # print(targets)
        # print(predictions)
        cpp_loss = nanobind_criterion.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_batch_BCELoss(self):
        targets = torch.round(torch.rand(5, 10, dtype=torch.double))
        predictions = torch.rand(5, 10, dtype=torch.double)
        cpp_loss = nanobind_criterion.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_MSELoss(self):
        targets = torch.rand(1, 10, dtype=torch.double)
        predictions = torch.rand(1, 10, dtype=torch.double)
        cpp_loss = nanobind_criterion.MSELoss(predictions, targets)
        python_loss = nn.functional.mse_loss(
            predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

    def test_batch_MSELoss(self):
        targets = torch.rand(5, 10, dtype=torch.double)
        predictions = torch.rand(5, 10, dtype=torch.double)
        cpp_loss = nanobind_criterion.MSELoss(predictions, targets)
        python_loss = nn.functional.mse_loss(
            predictions, targets)
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 6)

# Forward pass using cpp to calculate the layers


class TestGetCodes(unittest.TestCase):
    # Assert all values are the same to tolerance of ulp
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


class TestUpdateCodes(unittest.TestCase):
    def test_update_codes(self):
        model = simpleNN(2, [4, 3], 1)
        model = get_mods(model)
        in_tensor = torch.rand(5, 2, dtype=torch.double)
        targets = torch.round(torch.rand(5, 1, dtype=torch.double))

        # Ignore Flatten for now
        model = model[1:]

        cpp_out, cpp_codes = cf_get_codes(model, in_tensor)

        #cf_update_codes(cpp_codes, model, targets)
        test_loss()
        # update_codes(cpp_codes, model, targets, nn.BCELoss(),
        #  mu = 0.003, lambda_c = 0.0, n_iter = 5, lr = 0.3)


if __name__ == '__main__':
    unittest.main()
