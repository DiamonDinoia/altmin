import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np 


def adam(m_t_minus_1, v_t_minus_1,val,  grad, lr = 0.001, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0):
    m_t = betas[0]*m_t_minus_1 + (1-betas[0])*grad 
    v_t = betas[1]*v_t_minus_1 + (1-betas[1])*(grad**2)
    
    m_t_correct = m_t / (1-betas[0])
    v_t_correct = v_t / (1-betas[1])

    res = val - lr * (m_t_correct/(np.sqrt(v_t_correct)+eps))

    return res, m_t, v_t

def compute_codes_loss(codes, nmod, lin, loss_fn, codes_targets, mu, lambda_c):
    """Function that computes the code loss

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

def store_momentums(model, cpp):
   
    momentum_dict = {}
    for i,m in enumerate(model):
        if hasattr(m, 'has_codes') and getattr(m, 'has_codes'):
            momentum_dict[str(i)+".weight_m"] = torch.zeros(m.weight.shape, dtype=torch.double)
            momentum_dict[str(i)+".weight_v"] = torch.zeros(m.weight.shape, dtype=torch.double)
            if m.bias!=None:
                momentum_dict[str(i)+".bias_m"] = torch.zeros(m.bias.shape, dtype=torch.double)
                momentum_dict[str(i)+".bias_v"] = torch.zeros(m.bias.shape, dtype=torch.double)
               
            momentum_dict[str(i)+".code_m"] = torch.zeros(m.weight.shape, dtype=torch.double)
            momentum_dict[str(i)+".code_v"] = torch.zeros(m.weight.shape, dtype=torch.double)

    #last layer 
    for m in model[-1]:
        if isinstance(m,nn.Linear):
            momentum_dict["-1.weight_m"] = torch.zeros(m.weight.shape, dtype=torch.double)
            momentum_dict["-1.weight_v"] = torch.zeros(m.weight.shape, dtype=torch.double)
            if m.bias!=None:
                momentum_dict["-1.bias_m"] = torch.zeros(m.bias.shape, dtype=torch.double)
                momentum_dict["-1.bias_v"] = torch.zeros(m.bias.shape, dtype=torch.double)


    if not cpp:
        for key in momentum_dict:
            momentum_dict[key] = 0
    return momentum_dict



def update_codes_manual(codes, model_mods, targets, criterion, mu, lambda_c, n_iter, momentum_dict):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
    count = 0
    for l in range(1, len(codes)+1):
        idx = id_codes[-l]

        codes[-l].requires_grad_(True)
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
            loss_fn = lambda x: mu*F.mse_loss(x, codes[-l+1].detach())

        for it in range(n_iter):
            loss = compute_codes_loss(codes[-l], nmod, lin, loss_fn, codes_initial, mu, lambda_c)
            loss.backward()
            dc = codes[-l].grad 
            c, m_c_t, v_c_t = adam(momentum_dict[str(id_codes[-l])+".code_m"], momentum_dict[str(id_codes[-l])+".code_v"], codes[-l], dc, lr=0.3)
            momentum_dict[str(id_codes[-l])+".code_m"] = m_c_t
            momentum_dict[str(id_codes[-l])+".code_v"] = v_c_t
            with torch.no_grad():
                codes[-l] = c.detach().requires_grad_(True)
            #print(codes[-l].grad)
            #.zero()
        count+=1

    return codes

#Have to pass whole model not just last layer otherwise parameters are only updated in the scope of this function
def update_last_layer_manual(mod_out, inputs, targets, criterion, n_iter, momentum_dict):

    for it in range(n_iter):
        outputs = mod_out(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        dw = mod_out[1].weight.grad
        db = mod_out[1].bias.grad

        w, m_w_t, v_w_t = adam(momentum_dict['-1.weight_m'], momentum_dict['-1.weight_v'], mod_out[1].weight, dw, lr=0.008)
        b, m_b_t, v_b_t = adam(momentum_dict['-1.bias_m'], momentum_dict['-1.bias_v'], mod_out[1].bias, db, lr=0.008)

        momentum_dict['-1.weight_m'] = m_w_t 
        momentum_dict['-1.weight_v'] = v_w_t 

        momentum_dict['-1.bias_m'] = m_b_t 
        momentum_dict['-1.bias_v'] = v_b_t 

        with torch.no_grad():
            mod_out[1].weight = nn.Parameter(w)
            mod_out[1].bias = nn.Parameter(b)


def update_hidden_weights_adam_manual(model_mods, inputs, codes, lambda_w, n_iter, momentum_dict):
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
            loss = F.mse_loss(lin(nmod(c_in)), c_out.detach())
            if lambda_w > 0.0:
                loss += lambda_w*lin.weight.abs().mean()
            loss.backward()
            dw = lin.weight.grad
            db = lin.bias.grad
            w, m_w_t, v_w_t = adam(momentum_dict[str(idx)+".weight_m"], momentum_dict[str(idx)+".weight_v"], lin.weight, dw, lr=0.008)
            b, m_b_t, v_b_t = adam(momentum_dict[str(idx)+".bias_m"], momentum_dict[str(idx)+".bias_v"], lin.bias, db, lr=0.008)

            momentum_dict[str(idx)+".weight_m"] = m_w_t
            momentum_dict[str(idx)+".weight_v"] = v_w_t

            momentum_dict[str(idx)+".bias_m"] = m_b_t 
            momentum_dict[str(idx)+".bias_v"] = v_b_t 
            

            with torch.no_grad():
                lin.weight = nn.Parameter(w)
                lin.bias = nn.Parameter(b)