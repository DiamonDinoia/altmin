import nanobind_hello_world
import nanobind_hello_world_in
import nanobind_hello_world_out 
import nanobind_matrix_multiplication
import nanobind_return_matrix
import nanobind_matrix_in
import torch 
import numpy as np

print("Test basic function call")
nanobind_hello_world.hello_world()
print("Test pass value to c++")
nanobind_hello_world_in.hello_world_in("Hi from python")
print("Test pass value to python")
print(nanobind_hello_world_out.hello_world_out())

print(nanobind_return_matrix.return_matrix())
print(type(nanobind_return_matrix.return_matrix()))
a = np.asarray([[3,2],[-1,4]])
b = np.asarray([[1,1],[1,1]])

nanobind_matrix_in.matrix_in(a)
print(nanobind_matrix_multiplication.matrix_multiplication(a,b))