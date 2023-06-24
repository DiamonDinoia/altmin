import torch 
import torch.nn as nn
import numpy as np
import time
import fast_altmin
import matplotlib.pyplot as plt
import pandas as pd

def test_lin_time():
        
        for i in [1 << x for x in range(10, 14)]:
            n = i 
            m = i
            a = i
            b = i
            input = np.random.rand(n,m)
            weight = np.random.rand(a,b)
            print("cpp mat mul start")
            start =time.time()
            res = fast_altmin.matrix_mul(input, weight)
            end = time.time()
            print("cpp mat mul "+str(end-start))

            input = torch.rand(n,m)
            weight = torch.rand(a,b)
            print("py mat mul start")
            start=time.time()
            res = torch.matmul(input,weight)
            end = time.time()
            print("py mat mul "+str(end-start))

test_lin_time()