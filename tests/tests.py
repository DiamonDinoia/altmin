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

from altmin import get_mods, get_codes

from control_flow import cf_get_codes

from models import simpleNN

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
    def test_lin(self):
        lin = nn.Linear(2, 4)
        in_tensor = torch.rand(1, 2)
        weight = lin.weight.data
        bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))

        cpp_imp = nanobind_layers.lin(in_tensor,
                                      weight, bias)

        python_imp = lin(in_tensor).detach().numpy().astype(np.float64)

        assert(np.allclose(python_imp, cpp_imp, rtol=1e-04, atol=1e-07))

    def test_lin_batch(self):
        lin = nn.Linear(10, 30)
        in_tensor = torch.rand(5, 10)
        weight = lin.weight.data
        bias = torch.reshape(lin.bias.data, (len(lin.bias.data), 1))

        cpp_imp = nanobind_layers.lin(in_tensor,
                                      weight, bias)

        python_imp = lin(in_tensor).detach().numpy().astype(np.float64)
        assert(np.allclose(python_imp, cpp_imp, rtol=1e-04, atol=1e-07))

    # Functions for relu and sigmoid are in place so they don't need to return a value
    def test_ReLU(self):
        relu = nn.ReLU()
        in_tensor = torch.rand(5, 10)
        python_imp = relu(in_tensor)
        # In tensor data updated in place
        nanobind_layers.ReLU(
            in_tensor)
        assert(np.allclose(python_imp, in_tensor, rtol=1e-04, atol=1e-07))

    def test_sigmoid(self):
        sigmoid = nn.Sigmoid()
        in_tensor = torch.rand(5, 10)
        python_imp = sigmoid(in_tensor)
        # In tensor data changed in place?
        nanobind_layers.sigmoid(
            in_tensor)

        assert(np.allclose(python_imp, in_tensor, rtol=1e-04, atol=1e-07))

# Testing the cpp implementation of BCEloss and MSEloss
# Also test the fuctions ability to calculate loss on batch input
# Again take reference to tensor as input so no data copied between


class TestCriterion(unittest.TestCase):
    def test_BCELoss(self):
        targets = torch.round(torch.rand(1, 10))
        predictions = torch.rand(1, 10)
        # print(targets)
        # print(predictions)
        cpp_loss = nanobind_criterion.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)

        self.assertAlmostEqual(cpp_loss, python_loss.item(), 4)

    def test_batch_BCELoss(self):
        targets = torch.rand(5, 10)
        predictions = torch.rand(5, 10)
        cpp_loss = nanobind_criterion.BCELoss(predictions, targets)
        python_loss = nn.BCELoss()(predictions, targets)

        self.assertAlmostEqual(cpp_loss, python_loss.item(), 4)

    def test_MSELoss(self):
        predictions = np.random.rand(10)
        targets = np.random.rand(10)
        cpp_loss = nanobind_criterion.MSELoss(predictions, targets)
        python_loss = nn.functional.mse_loss(
            torch.from_numpy(predictions), torch.from_numpy(targets))
        self.assertAlmostEqual(cpp_loss, python_loss.item(), 4)

# Forward pass using cpp to calculate the layers


class TestGetCodes(unittest.TestCase):
    def test_get_codes(self):
        model = simpleNN(2, [4, 3], 1)
        model = get_mods(model)
        in_tensor = torch.rand(5, 2)

        # Ignore Flatten for now
        model = model[1:]

        python_out, python_codes = get_codes(model, in_tensor)

        cpp_out, cpp_codes = cf_get_codes(model, in_tensor)

        assert(np.allclose(python_out.detach().numpy(),
               cpp_out, rtol=1e-04, atol=1e-07))

        for x in range(len(cpp_codes)):
            assert(np.allclose(python_codes[x].detach().numpy(),
                               cpp_codes[x], rtol=1e-04, atol=1e-07))


'''


class TestUpdateCodes(unittest.TestCase):
    def test_update_codes(self):
        model = simpleNN(2, [4, 3], 1)
        model = get_mods(model)
        in_tensor = torch.rand(5, 2)

        # Ignore Flatten for now
        model = model[1:]
        id_codes = [i for i, m in enumerate(model) if hasattr(
            m, 'has_codes') and getattr(m, 'has_codes')]

        layer_map = []
        for mod in model:
            if isinstance(mod, nn.Linear):
                layer_map.append(0)
            elif isinstance(mod, nn.ReLU):
                layer_map.append(1)
            elif isinstance(mod, nn.Sigmoid):
                layer_map.append(2)
            elif isinstance(mod, nn.Sequential):
                layer_map.append(3)
            else:
                print("layer not imp yet")

        for mod in model[-1]:
            if isinstance(mod, nn.Linear):
                layer_map.append(0)
            elif isinstance(mod, nn.ReLU):
                layer_map.append(1)
            elif isinstance(mod, nn.Sigmoid):
                layer_map.append(2)
            else:
                print("layer not imp yet")

        layer_map = np.asarray(layer_map)

        tmp = model.state_dict()
        for key in tmp.keys():
            if 'bias' in key:
                tmp[key] = torch.reshape(tmp[key], (1, len(tmp[key])))

        print(model)
        print(id_codes)
        print(tmp)
        print(layer_map)

        tmp = list(tmp.values())

        codes = [torch.rand(5, 4), torch.rand(5, 3)]
        print(model[3](codes[-1]))
        targets = np.random.choice([0.0, 1.0], 10)
        cpp_codes = nanobind_get_codes.update_codes(
            codes, layer_map, id_codes, targets, tmp)

'''

if __name__ == '__main__':
    unittest.main()
