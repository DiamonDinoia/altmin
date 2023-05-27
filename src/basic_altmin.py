import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np 


def compute_codes_loss(codes, nmod, lin, loss_fn, codes_targets, mu, lambda_c):
    r"""Function that computes the code loss

    Inputs:
        - **codes** (batch, num_features): outputs of the linear modules
        - **nmod** (nn.Module): non-linear module downstream from linear module
        - **lin** (nn.Conv2d or nn.Lineae):  linear module
        - **loss_fn**: loss function computed on outputs
        - **codes_targets** (batch, num_features): target to which codes have to be close (in L2 distance)
        - **lambda_c** (scalar): Lagrance muliplier for code loss function
    Outputs:
        - **loss**: loss
    """
    output = lin(nmod(codes))
    loss = (1/mu)*loss_fn(output) + torch.nn.functional.mse_loss(codes, codes_targets)
    if lambda_c>0.0:
        loss += (lambda_c/mu)*codes.abs().mean()
    return loss





def update_codes_cpp(codes, model_mods, targets, criterion, mu, lambda_c, n_iter, lr):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
    for l in range(1, len(codes)+1):
        idx = id_codes[-l]

        
        #optimizer = optim.SGD([codes[-l]], lr=lr, momentum=0.9, nesterov=True)
        codes_initial = codes[-l].clone()

        if idx+1 in id_codes:
            nmod = lambda x: x
            lin = model_mods[idx+1]
        else:
            try:
                nmod = model_mods[idx+1]
            except IndexError:
                nmod = lambda x: x
            try:
                lin = model_mods[idx+2]
            except IndexError:
                lin = lambda x: x

        if l == 1:  # last layer
            loss_fn = lambda x: criterion(x, targets)
        else:       # intermediate layers
            loss_fn = lambda x: mu*torch.nn.functional.mse_loss(x, codes[-l+1].detach())
    
        
        for it in range(n_iter):
            #optimizer.zero_grad()
            codes[-l].requires_grad_(True)
            #codes[-l].grad.zero_()
            loss = compute_codes_loss(codes[-l], nmod, lin, loss_fn, codes_initial, mu, lambda_c)
            # output = lin(nmod(codes))
            # loss = (1/mu)*loss_fn(output) + F.mse_loss(codes, codes_targets)
            loss.backward()
            with torch.no_grad():
                codes[-l] = codes[-l] - lr*codes[-l].grad
            #optimizer.step()

    return codes

def update_hidden_weights_cpp(model_mods, inputs, codes, lambda_w, n_iter):
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

        for it in range(n_iter):
            
            #lin.optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(lin(nmod(c_in)), c_out.detach())
            if lambda_w > 0.0:
                loss += lambda_w*lin.weight.abs().mean()
            loss.backward()
            with torch.no_grad():
                lin.weight = nn.Parameter(lin.weight - lr_weights * lin.weight.grad)
                #These ones have no bias
                #Can rewrite this code to be more resiliant at some point
                #lin.bias = nn.Parameter(lin.bias - lr_weights * lin.bias.grad)
            #lin.optimizer.step()


def update_last_layer_cpp(mod_out, inputs, targets, criterion, n_iter):
    lr_weights = 0.008
    for it in range(n_iter):
        outputs = mod_out(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for mod in mod_out:
            if isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight = nn.Parameter(mod.weight - lr_weights * mod.weight.grad)
                    mod.bias = nn.Parameter(mod.bias - lr_weights * mod.bias.grad)
