import sys
import torch.nn as nn

def check_equal(first_imp, second_imp, eps):
    first_imp = first_imp.flatten()
    second_imp = second_imp.flatten()
    for i in range(len(first_imp)):
        assert(abs(first_imp[i] - second_imp[i]) <= sys.float_info.epsilon*eps)
        
def check_equal_bias(first_imp, second_imp,eps):
    for x in range(len(first_imp)):
        assert(abs(first_imp[x] - second_imp[x]) <= sys.float_info.epsilon*eps)
            
def check_equal_weights_and_bias(model_python, weights, biases, eps = 10e6):
    y = 0
    for x,m in enumerate(model_python):

        if isinstance(m, nn.Linear):
            check_equal(model_python[x].weight.data, weights[y], eps)
            check_equal_bias(model_python[x].bias, biases[y], eps)
            y+=1
            
        if isinstance(m, nn.Sequential):
            check_equal(model_python[x][1].weight.data, weights[y], eps)
            check_equal_bias(model_python[x][1].bias, biases[y], eps)

def check_equal_weights_and_bias4d(model_python, weights, biases, eps = 10e6):
    y = 0
    for x,m in enumerate(model_python):

        if isinstance(m, nn.Conv2d):
            check_equal4d(model_python[x].weight.data, weights[y], eps)
            y+=1


def check_equal4d(first_imp, second_imp, eps):
    N = len(first_imp)
    C_out = len(first_imp[0])
    H_out = len(first_imp[0][0])
    W_out = len(first_imp[0][0][0])
    for n in range(N):
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    #print(str(n) + " " + str(c) + " "+str(i)+" "+str(j))
                    if (abs(first_imp[n][c][i][j] - second_imp[n][c][i][j]) > sys.float_info.epsilon*eps):
                        print(first_imp[n][c][i][j])
                        print(second_imp[n][c][i][j])
                    assert(abs(first_imp[n][c][i][j] - second_imp[n][c][i][j]) <= sys.float_info.epsilon*eps)
