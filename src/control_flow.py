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
from models import simpleNN


def cf_get_codes(model_mods, inputs):
    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    # As codes only return outputs of linear layers
    codes = []
    i = 0
    tmp = list(model_mods.state_dict().values())

    for m in model_mods:
        if isinstance(m, nn.Linear):
            weight = tmp[i]
            bias = torch.reshape(tmp[i+1], (len(tmp[i+1]), 1))
            i += 2
            # I can probably tell cpp to return a tensor instead of a numpy
            # I'll come back to this

            print(type(x))
            print(type(weight))
            print(type(bias))
            x = torch.from_numpy(nanobind_layers.lin(x, weight, bias))
        elif isinstance(m, nn.ReLU):
            nanobind_layers.ReLU(x)
        elif isinstance(m, nn.Sigmoid):
            nanobind_layers.sigmoid(x)
        elif isinstance(m, nn.Sequential):
            for n in m:
                if isinstance(n, nn.Linear):
                    weight = tmp[i]
                    bias = torch.reshape(tmp[i+1], (len(tmp[i+1]), 1))
                    i += 2
                    x = torch.from_numpy(nanobind_layers.lin(x, weight, bias))
                elif isinstance(n, nn.ReLU):
                    nanobind_layers.ReLU(x)
                elif isinstance(n, nn.Sigmoid):
                    nanobind_layers.sigmoid(x)
        else:
            print("layer not imp yet")

        if hasattr(m, 'has_codes') and getattr(m, 'has_codes'):
            codes.append(x.clone())
    # Do not include output of very last linear layer (not counted among codes)
    return x, codes
