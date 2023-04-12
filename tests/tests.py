import unittest
import nanobind_hello_world
import nanobind_hello_world_in
import nanobind_hello_world_out
import nanobind_matrix_multiplication
import nanobind_return_matrix
import nanobind_matrix_in
import torch
import numpy as np


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


if __name__ == '__main__':
    unittest.main()
