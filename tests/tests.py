import unittest
import nanobind_hello_world
import nanobind_hello_world_in
import nanobind_hello_world_out
import nanobind_matrix_multiplication
import nanobind_return_matrix
import nanobind_matrix_in

import nanobind_lin

import torch
import numpy as np
import torch.nn as nn


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


if __name__ == '__main__':
    unittest.main()
