import torch.nn as nn
import numpy as np
import torch

import nanobind_get_codes
import nanobind_relu
import nanobind_lin
import nanobind_sigmoid
import nanobind_matrix_in
import nanobind_return_matrix
import nanobind_matrix_multiplication
import nanobind_hello_world_out
import nanobind_hello_world_in
import nanobind_hello_world
import nanobind_pass_dict
import unittest

from altmin import get_mods, get_codes
from models import simpleNN


class TestHelloWorldOut(unittest.TestCase):
    def test_output_from_cpp(self):
        res = nanobind_hello_world_out.hello_world_out()
        assert(res == "Hi python from c++")


class TestMatrixMultiplication(unittest.TestCase):
    def test_matrix_multiplication_in_cpp(self):
        a = np.asarray([[3, 2], [-1, 4]])
        b = np.asarray([[1, 0], [1, 1]])
        c = nanobind_matrix_multiplication.matrix_multiplication(a, b)
        assert((c == np.asarray([[5, 2], [3, 4]])).all())


class TestLin(unittest.TestCase):

    '''
    def test_lin(self):
        lin = nn.Linear(10, 30)
        in_tensor = torch.rand(1, 10)
        weight = lin.weight.data.numpy().astype(np.float64)
        bias = lin.bias.data.numpy().astype(
            np.float64).reshape(1, lin.out_features)

        cpp_imp = nanobind_lin.lin(in_tensor.numpy().astype(np.float64),
                                   weight, bias)

        python_imp = lin(in_tensor).detach().numpy().astype(np.float64)

        assert(np.allclose(python_imp, cpp_imp, rtol=1e-04, atol=1e-07))

    def test_lin_batch(self):
        lin = nn.Linear(10, 30)
        in_tensor = torch.rand(5, 10)
        weight = lin.weight.data.numpy().astype(np.float64)
        bias = lin.bias.data.numpy().astype(
            np.float64).reshape(1, lin.out_features)

        cpp_imp = nanobind_lin.lin(in_tensor.numpy().astype(np.float64),
                                   weight, bias)

        python_imp = lin(in_tensor).detach().numpy().astype(np.float64)

        assert(np.allclose(python_imp, cpp_imp, rtol=1e-04, atol=1e-07))


    def test_lin_problem(self):
        model = simpleNN(10, [20, 4], 1)
        model = get_mods(model)
        in_tensor = torch.rand(5, 10)

        # Ignore Flatten for now
        model = model[1:]
        tmp = dict(model.state_dict())

        for keys in tmp:
            if 'bias' in keys:
                tmp[keys] = tmp[keys].numpy().reshape(1, len(tmp[keys]))
            else:
                tmp[keys] = tmp[keys].numpy()

        # print(len(str_array))
        # print(str_array)

        print(tmp["1.bias"])
        print("\n")
        print(tmp["1.weight"])
        print("\n\n")

        nanobind_lin.lin(in_tensor.numpy().astype(np.float64),
                         tmp["1.weight"], tmp["1.bias"])
    '''


class TestReLU(unittest.TestCase):

    def test_ReLU(self):
        relu = nn.ReLU()
        in_tensor = torch.rand(5, 10)

        cpp_imp = nanobind_relu.ReLU(in_tensor.numpy())
        python_imp = relu(in_tensor)

        assert(np.allclose(python_imp, cpp_imp, rtol=1e-04, atol=1e-07))


class TestSigmoid(unittest.TestCase):

    def test_sigmoid(self):
        sigmoid = nn.Sigmoid()
        in_tensor = torch.rand(5, 10)

        cpp_imp = nanobind_sigmoid.sigmoid(in_tensor.numpy())
        python_imp = sigmoid(in_tensor)

        assert(np.allclose(python_imp, cpp_imp, rtol=1e-04, atol=1e-07))


'''
class TestGetCodes(unittest.TestCase):
    def test_get_codes(self):
        model = simpleNN(2, [5, 2], 1)
        model = get_mods(model)
        in_tensor = torch.rand(2, 10)

        # Ignore Flatten for now
        model = model[1:]
        id_codes = np.array([i for i, m in enumerate(model) if hasattr(
            m, 'has_codes') and getattr(m, 'has_codes')])
        str_array = np.array([str(mod) for mod in model])

        # print(str_array)
        for mod in model[-1]:
            str_array = np.append(str_array, str(mod))

        tmp = model.state_dict()

        weights = []
        biases = []
        for keys in tmp:
            if 'bias' in keys:
                biases.append(tmp[keys].numpy().flatten())
            else:
                weights.append(tmp[keys].numpy())

        print(tmp)
        print(weights)
        print(biases)
        cpp_out, cpp_codes = nanobind_get_codes.get_codes(
            str_array, id_codes, in_tensor.numpy(), weights, biases)

        # print(nanobind_lin.lin(in_tensor.numpy().astype(np.float64),
        # tmp["1.weight"], tmp["1.bias"]))
        python_out, python_codes = get_codes(model, in_tensor)

        # print(python_out.detach().numpy())
        # print(cpp_out)
        assert(np.allclose(python_out.detach().numpy(),
               cpp_out, rtol=1e-04, atol=1e-07))


'''


class TestPassDict(unittest.TestCase):
    '''

   def test_pass_dict(self):
        a = torch.rand(5, 2).numpy()
        b = torch.rand(1, 5).numpy().flatten()
        c = torch.rand(2, 5).numpy()
        d = torch.rand(1, 5).numpy().flatten()

        weights = [a, c]
        biases = [b, d]

        # print(a)
        # print(c)

        # print(b)
        # print(d)
        nanobind_pass_dict.pass_dict(weights, biases)
    '''

    def test_model(self):
        model = simpleNN(2, [4, 3], 1)
        model = get_mods(model)
        in_tensor = torch.rand(2, 2)

        # Ignore Flatten for now
        model = model[1:]
        id_codes = np.array([i for i, m in enumerate(model) if hasattr(
            m, 'has_codes') and getattr(m, 'has_codes')])
        str_array = np.array([str(mod) for mod in model])

        # print(str_array)
        for mod in model[-1]:
            str_array = np.append(str_array, str(mod))

        tmp = model.state_dict()

        weights = []
        biases = []
        for keys in tmp:
            if 'bias' in keys:
                biases.append(tmp[keys].numpy().flatten())
            else:
                weights.append(tmp[keys].numpy())

        # print(tmp)
        print(weights)
        print("\n")
        print(biases)
        print("\n\n")

        cpp_out, cpp_codes = nanobind_get_codes.get_codes(
            str_array, id_codes, in_tensor.numpy(), weights[:-1], biases, weights[-1].flatten())

        python_out, python_codes = get_codes(model, in_tensor)

        # print(python_out.detach().numpy())
        # print(cpp_out)
        assert(np.allclose(python_out.detach().numpy(),
               cpp_out, rtol=1e-04, atol=1e-07))


if __name__ == '__main__':
    unittest.main()
