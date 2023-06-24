import torch 
import torch.nn as nn
import numpy as np
import time
import fast_altmin
import matplotlib.pyplot as plt
import pandas as pd

def test_linear(batch_size, n_iter, epochs):
        inputs_one_cpp = []
        inputs_two_cpp = []
        inputs_three_cpp = []

        inputs_one_py = []
        inputs_two_py = []
        inputs_three_py = []
        
        for x in range(n_iter):
            input = np.random.rand(batch_size,5)
            inputs_one_cpp.append(input)
            inputs_one_py.append(torch.from_numpy(input))

        for x in range(n_iter):
            input = np.random.rand(batch_size,25)
            inputs_two_cpp.append(input)
            inputs_two_py.append(torch.from_numpy(input))

        for x in range(n_iter):
            input = np.random.rand(batch_size,30)
            inputs_three_cpp.append(input)
            inputs_three_py.append(torch.from_numpy(input))

        time_cpp = 0.0 
        time_py = 0.0

        model = nn.Linear(5,25).double()
        for epoch in range(epochs):
            for input in inputs_one_py:
                start = time.time()
                model(input)
                end = time.time()
                time_py += (end-start)

        model = nn.Linear(25,30).double()
        for epoch in range(epochs):
            for input in inputs_two_py:
                start = time.time()
                model(input)
                end = time.time()
                time_py += (end-start)

        model = nn.Linear(30,1).double()
        for epoch in range(epochs):
            for input in inputs_three_py:
                start = time.time()
                model(input)
                end = time.time()
                time_py += (end-start)

        

        model = nn.Linear(5,25).double()
        weight = model.weight.data
        weight_transpose = model.weight.data.transpose(1,0).numpy()
        bias = model.bias.data.numpy()
        for epoch in range(epochs):
            for input in inputs_one_cpp:
                start = time.time()
                fast_altmin.lin(input, weight, bias)
                end = time.time()
                time_cpp += (end-start)

        model = nn.Linear(25,30).double()
        weight = model.weight.data
        weight_transpose = model.weight.data.transpose(1,0).numpy()
        bias = model.bias.data.numpy()
        for epoch in range(epochs):
            for input in inputs_two_cpp:
                start = time.time()
                fast_altmin.lin(input, weight, bias)
                end = time.time()
                time_cpp += (end-start)

        model = nn.Linear(30,1).double()
        weight = model.weight.data
        weight_transpose = model.weight.data.transpose(1,0).numpy()
        bias = model.bias.data.numpy()
        for epoch in range(epochs):
            for input in inputs_three_cpp:
                start = time.time()
                fast_altmin.lin(input, weight, bias)
                end = time.time()
                time_cpp += (end-start)

        return time_py, time_cpp 


def plot_two_grouped_bar(data_one, data_two, group_labels, title, legend, xlabel, ylabel, file_path):
    groups = {legend[0] : data_one, legend[1] : data_two}
    df = pd.DataFrame(groups, index=group_labels)
    ax = df.plot.bar(rot=0,figsize = (10,6))
    plt.title(title)
    plt.legend(loc='lower right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_path)
    plt.show()

def main():
    batch_sizes = [1 << x for x in range(5, 15)]
    python_times = [0.0] * len(batch_sizes)
    cpp_times = [0.0] * len(batch_sizes)

    for x, batch_size in enumerate(batch_sizes):
        python_times[x], cpp_times[x] = test_linear(batch_size,1500, 3)


    for x in range(len(python_times)):
        print("\nTime for python with batch size "+str(batch_sizes[x]) +": " + str(python_times[x]))
        print("Time for cpp with batch size "+str(batch_sizes[x]) +": " + str(cpp_times[x]))

    plot_two_grouped_bar(python_times, cpp_times, batch_sizes , "Show performance of linear layer in cpp and python", ["python", "cpp"], "batch_size", "time", "problem.png")

main()