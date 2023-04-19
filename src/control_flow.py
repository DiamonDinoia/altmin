import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim

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


def cf_update_codes(codes, model_mods, targets, mu=0.003, lambda_c=0.0, n_iter=5, lr=0.3):

    id_codes = [i for i, m in enumerate(model_mods) if hasattr(
        m, 'has_codes') and getattr(m, 'has_codes')]

    tmp = list(model_mods.state_dict().values())

    print(model_mods)
    print(id_codes)
    print(tmp)

    for l in range(1, len(codes)+1):
        idx = id_codes[-l]
        codes[-l].requires_grad_(True)
        optimizer = optim.SGD([codes[-l]], lr=lr, momentum=0.9, nesterov=True)
        codes_initial = codes[-l].clone()
        i = len(tmp) - 1
        if idx+1 in id_codes:
            # weight =
            # bias =
            # codes[-l] = torch.from_numpy(nanobind_layers.lin(codes[-l], weight, bias))
            pass
        else:
            if idx+1 < len(model_mods):
                # nmod = model_modes[idx+1]
                nmod = model_mods[idx+1]

            if idx+2 < len(model_mods):
                # weight =
                # bias =
                # codes[-l] = torch.from_numpy(nanobind_layers.lin(codes[-l], weight, bias))
                print("two")

        for it in range(n_iter):
            # loss = compute_codes_loss(codes[-l], nmod, lin, loss_fn, codes_initial, mu, lambda_c)
            output = codes[-l].detach().clone()
            m = nmod
            if isinstance(m, nn.ReLU):
                nanobind_layers.ReLU(output)
            elif isinstance(m, nn.Sigmoid):
                nanobind_layers.sigmoid(output)
            elif isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n, nn.Linear):
                        weight = tmp[i-1]
                        bias = torch.reshape(tmp[i], (len(tmp[i]), 1))
                        i -= 2
                        output = torch.from_numpy(
                            nanobind_layers.lin(output, weight, bias))
                    elif isinstance(n, nn.ReLU):
                        nanobind_layers.ReLU(output)
                    elif isinstance(n, nn.Sigmoid):
                        nanobind_layers.sigmoid(output)

            print(output)
            optimizer.zero_grad()
            if l == 1:
                loss = (1.0/mu)*nanobind_criterion.BCELoss(
                    output, targets) + nanobind_criterion.MSELoss(codes[-l], codes_initial)
            else:
                loss = (1.0/mu)*(mu * nanobind_criterion.MSELoss(
                    output, codes[-l+1])) + nanobind_criterion.MSELoss(codes[-l], codes_initial)
            loss.backward()
            optimizer.step()

        break

    return 3
