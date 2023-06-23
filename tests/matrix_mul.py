import torch 
import torch.nn as nn
import numpy as np
import time
import fast_altmin
import matplotlib.pyplot as plt
import pandas as pd

def test_lin_time():
        
        n = 5000 
        m = 5
        a = 5
        b = 25
        start =time.time()
        fast_altmin.matrix_mul_two(m,n,a,b)
        end = time.time()
        print("cpp mat mul "+str(end-start))

        start=time.time()
        for i in range(5000):
            input = torch.rand(n,m)
            weight = torch.rand(a,b)
            print(input.shape)
            print(weight.shape)
            break
            res = torch.matmul(input,weight)

        end = time.time()
        print("py mat mul"+str(end-start))

test_lin_time()