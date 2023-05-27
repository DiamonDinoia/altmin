import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim

import nanobind
import unittest

from altmin import get_mods, get_codes
from models import simpleNN


def select_func(m, x, tmp, i):
    if isinstance(m, nn.Linear):
        weight = tmp[i]
        bias = torch.reshape(tmp[i+1], (len(tmp[i+1]), 1))
        i += 2
        # I can probably tell cpp to return a tensor instead of a numpy
        # I'll come back to this

        x = torch.from_numpy(nanobind.lin(x, weight, bias))
    elif isinstance(m, nn.ReLU):
        nanobind.ReLU_inplace(x)
    elif isinstance(m, nn.Sigmoid):
        nanobind.sigmoid_inplace(x)
    elif isinstance(m, nn.Sequential):
        for n in m:
            if isinstance(n, nn.Linear):
                weight = tmp[i]
                bias = torch.reshape(tmp[i+1], (len(tmp[i+1]), 1))
                i += 2
                x = torch.from_numpy(nanobind.lin(x, weight, bias))
            elif isinstance(n, nn.ReLU):
                nanobind.ReLU_inplace(x)
            elif isinstance(n, nn.Sigmoid):
                nanobind.sigmoid_inplace(x)
    else:
        pass
        #print(m)
        #print("layer not imp yet")
    return x, i


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
        x, i = select_func(m, x, tmp, i)

        if hasattr(m, 'has_codes') and getattr(m, 'has_codes'):
            codes.append(x.clone())
    # Do not include output of very last linear layer (not counted among codes)
    return x, codes

def conv_mods(nmod, lin):
    mods = []
    if isinstance(nmod, nn.ReLU):
        mods.append(0)
    elif isinstance(nmod, nn.Sigmoid):
        mods.append(2)
    elif isinstance(nmod, nn.Sequential):
        for mod in nmod:
            if isinstance(mod, nn.ReLU):
                mods.append(0)
            elif isinstance(mod, nn.Linear):
                mods.append(1)
            elif isinstance(mod, nn.Sigmoid):
                mods.append(2)
    else:
        pass
        #print("Not imp")
    if isinstance(lin, nn.Linear):
        mods.append(1)
    return mods

def cf_update_codes(codes, model_mods, targets, criterion, momentum_dict, init_vals, mu=0.003, lambda_c=0.0, n_iter=5, lr=0.3):

    id_codes = [i for i, m in enumerate(model_mods) if hasattr(
        m, 'has_codes') and getattr(m, 'has_codes')]


    for l in range(1, len(codes)+1):
        idx = id_codes[-l]
        codes[-l].requires_grad_(True)
        codes_initial = codes[-l].clone()

        if idx+1 in id_codes:
            nmod = (lambda x: x)
            lin = model_mods[idx+1]
        else:
            nmod, lin = (lambda x: x), (lambda x: x) 
            if idx+1 < len(model_mods):
                nmod = model_mods[idx+1]
            if idx+2 < len(model_mods):
                lin = model_mods[idx+2]


        #Use int to signify to cpp what loss func to use
        if l == 1:  # last layer
            #loss_fn = lambda x: criterion(x, targets)
            last_layer = 1
        else:       # intermediate layers
            #loss_fn = lambda x: mu*torch.nn.functional.mse_loss(x, codes[-l+1].detach())
            last_layer = 0
            targets = codes[-l+1]

        mods = conv_mods(nmod,lin)
        
        criterion = 0
        #I think targets and next codes can be merged into one var
        lr = 0.3
        if isinstance(lin, nn.Linear):
            nanobind.update_codes(lin.weight.data, lin.bias.data, mods, codes[-l], targets, momentum_dict[str(idx)+".code_m"], momentum_dict[str(idx)+".code_v"],
                                  criterion, n_iter, last_layer, lr, init_vals   )
        else:
            nanobind.update_codes(nmod[1].weight.data, nmod[1].bias.data, mods, codes[-l], targets, momentum_dict[str(idx)+".code_m"], momentum_dict[str(idx)+".code_v"],
                                   criterion, n_iter, last_layer, lr, init_vals  )

    return codes

def cf_update_hidden_weights(model_mods, inputs, codes, lambda_w, n_iter, lr, momentum_dict, init_vals):
    lr_weights = 0.008
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]

    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    for idx, c_in, c_out in zip(id_codes, [x]+codes[:-1], codes):
        lin = model_mods[idx]
        if idx >= 1 and not idx-1 in id_codes:
            nmod = model_mods[idx-1]
        else:
            nmod = lambda x: x

       
        mods = conv_mods(nmod, lin)
       
        nanobind.update_hidden_weights(lin.weight.data, lin.bias.data, mods, c_in, c_out, momentum_dict[str(idx)+".weight_m"], momentum_dict[str(idx)+".weight_v"],
                                        momentum_dict[str(idx)+".bias_m"], momentum_dict[str(idx)+".bias_v"], n_iter, lr, init_vals )

        
def cf_update_last_layer(model, inputs, targets, criterion, n_iter, lr, momentum_dict, init_vals ):
    #nanobind.update_last_layer(model_mods[1].weight.data, model_mods[1].bias.data, conv_mods(model_mods), inputs, targets, criterion, n_iter)#
    #conv_mods is set up stupidly
    nanobind.update_last_layer(model[-1][1].weight.data,model[-1][1].bias.data, conv_mods(model[-1],-1), inputs.detach(), targets, 
                                           momentum_dict["-1.weight_m"], momentum_dict["-1.weight_v"], momentum_dict["-1.bias_m"], 
                                           momentum_dict["-1.bias_v"], criterion, n_iter, lr, init_vals )



 