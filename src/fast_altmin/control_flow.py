import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim

import fast_altmin
import unittest

import os, sys 
sys.path.insert(0, os.path.abspath("../artifacts"))


def store_momentums(model, init_vals):
   
    momentum_dict = {}
    for i,m in enumerate(model):
        if hasattr(m, 'has_codes') and getattr(m, 'has_codes'):
            momentum_dict[str(i)+".weight_m"] = torch.zeros(m.weight.shape, dtype=torch.double)
            momentum_dict[str(i)+".weight_v"] = torch.zeros(m.weight.shape, dtype=torch.double)
            if m.bias!=None:
                momentum_dict[str(i)+".bias_m"] = torch.zeros(m.bias.shape, dtype=torch.double)
                momentum_dict[str(i)+".bias_v"] = torch.zeros(m.bias.shape, dtype=torch.double)
               
            momentum_dict[str(i)+".b_t"] = torch.zeros(m.weight.shape, dtype=torch.double)

            momentum_dict[str(i)+".layer_step"] = 0
            momentum_dict[str(i)+".code_step"] = 0

    #last layer 
    for m in model[-1]:
        if isinstance(m,nn.Linear):
            momentum_dict["-1.weight_m"] = torch.zeros(m.weight.shape, dtype=torch.double)
            momentum_dict["-1.weight_v"] = torch.zeros(m.weight.shape, dtype=torch.double)
            if m.bias!=None:
                momentum_dict["-1.bias_m"] = torch.zeros(m.bias.shape, dtype=torch.double)
                momentum_dict["-1.bias_v"] = torch.zeros(m.bias.shape, dtype=torch.double)

            momentum_dict["-1.layer_step"] = 0
            


    if not init_vals:
        for key in momentum_dict:
            momentum_dict[key] = 0
    return momentum_dict


def select_func(m, x, tmp, i):
    if isinstance(m, nn.Linear):
        weight = tmp[i]
        bias = tmp[i+1]
        i += 2
        # I can probably tell cpp to return a tensor instead of a numpy
        # I'll come back to this

        x = torch.from_numpy(fast_altmin.lin(x, weight, bias))
    elif isinstance(m, nn.ReLU):
        fast_altmin.ReLU_inplace(x)
    elif isinstance(m, nn.Sigmoid):
        fast_altmin.sigmoid_inplace(x)
    elif isinstance(m, nn.Sequential):
        for n in m:
            if isinstance(n, nn.Linear):
                weight = tmp[i]
                bias = tmp[i+1]
                i += 2
                x = torch.from_numpy(fast_altmin.lin(x, weight, bias))
            elif isinstance(n, nn.ReLU):
                fast_altmin.ReLU_inplace(x)
            elif isinstance(n, nn.Sigmoid):
                fast_altmin.sigmoid_inplace(x)
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


def cf_update_last_layer(model, inputs, targets, n_iter, lr, momentum_dict, init_vals ):
    #fast_altmin.update_last_layer(model_mods[1].weight.data, model_mods[1].bias.data, conv_mods(model_mods), inputs, targets, criterion, n_iter)#
    #conv_mods is set up stupidly
    is_last_layer = 1
    fast_altmin.update_weights(model[-1][1].weight.data,model[-1][1].bias.data, conv_mods(model[-1],-1), inputs.detach(), targets, 
                                            momentum_dict["-1.weight_m"], momentum_dict["-1.weight_v"], momentum_dict["-1.bias_m"], 
                                            momentum_dict["-1.bias_v"], is_last_layer, n_iter, lr, init_vals, momentum_dict["-1.layer_step"] )
    momentum_dict["-1.layer_step"] += n_iter
        
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
       
        is_last_layer = 0
        fast_altmin.update_weights(lin.weight.data, lin.bias.data, mods, c_in, c_out, momentum_dict[str(idx)+".weight_m"], momentum_dict[str(idx)+".weight_v"],
                                        momentum_dict[str(idx)+".bias_m"], momentum_dict[str(idx)+".bias_v"], is_last_layer, n_iter, lr, init_vals, 
                                        momentum_dict[str(idx)+".layer_step"])
        
        
        momentum_dict[str(idx)+".layer_step"] += n_iter

def cf_update_weights_parallel(model_mods, inputs, codes, targets, lambda_w, n_iter, lr, momentum_dict, init_vals):
    lr_weights = 0.008
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
    parallel = []
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
        parallel.append([lin.weight.data, lin.bias.data, mods, c_in, c_out, momentum_dict[str(idx)+".weight_m"], momentum_dict[str(idx)+".weight_v"],
                                        momentum_dict[str(idx)+".bias_m"], momentum_dict[str(idx)+".bias_v"], 0, n_iter, lr, init_vals, 
                                        momentum_dict[str(idx)+".layer_step"]])

    #Do in parallel 
    n_tasks = len(parallel) + 1
    #Depends on number of layers 

    #fast_altmin.update_weights(parallel[0][0], parallel[0][1], parallel[0][2], parallel[0][3], parallel[0][4], parallel[0][5], parallel[0][6], parallel[0][7], 
    # parallel[0][8], parallel[0][9],parallel[0][10],parallel[0][11],parallel[0][12],parallel[0][13])
    #fast_altmin.update_weights(parallel[1][0], parallel[1][1], parallel[1][2], parallel[1][3], parallel[1][4], parallel[1][5], parallel[1][6], parallel[1][7], 
    # parallel[1][8], parallel[1][9],parallel[1][10],parallel[1][11],parallel[1][12],parallel[1][13])
    #etc
    for args in parallel:
        fast_altmin.update_weights(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],args[10],args[11],args[12],args[13])

    #last_layer can also be done in parallel 
    last_layer = 1
    fast_altmin.update_weights(model_mods[-1][1].weight.data,model_mods[-1][1].bias.data, conv_mods(model_mods[-1],-1), codes[-1].detach(), targets, 
                                            momentum_dict["-1.weight_m"], momentum_dict["-1.weight_v"], momentum_dict["-1.bias_m"], 
                                            momentum_dict["-1.bias_v"], last_layer, n_iter, lr, init_vals, momentum_dict["-1.layer_step"] )
    
    #Update step counters 
    for idx in id_codes:
        momentum_dict[str(idx)+".layer_step"] += n_iter
    momentum_dict["-1.layer_step"] += n_iter
    


def cf_update_codes(codes, model_mods, targets, criterion, mu=0.003, lambda_c=0.0, n_iter=5, lr=0.3):

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
            targets = codes[-l+1].detach()

        mods = conv_mods(nmod,lin)
        
        criterion = 0
        #I think targets and next codes can be merged into one var
       
        if isinstance(lin, nn.Linear):
            fast_altmin.update_codes(lin.weight.data, lin.bias.data, mods, codes[-l], targets, last_layer, n_iter, lr ,mu )
        else:
            fast_altmin.update_codes(nmod[1].weight.data, nmod[1].bias.data, mods, codes[-l], targets, last_layer, n_iter,  lr,mu )