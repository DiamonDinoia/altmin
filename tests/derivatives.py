import torch.nn as nn
import torch as torch

def derivative_BCELoss(output, target):
    #https://github.com/torch/nn/blob/master/lib/THNN/generic/BCECriterion.c#L57
    #Same eps as torch uses 
    eps=1e-12
    norm = 1.0/output.shape[1]
    return (-norm * ((target - output) / ((1.0 - output + eps) * (output + eps))))/output.shape[0]

def derivative_MSELoss(output, target):
    #https://github.com/torch/nn/blob/master/lib/THNN/generic/MSECriterion.c
    norm = 2.0/output.shape[1]
    return (norm * (output - target))/output.shape[0]

def derivative_CrossEntropyLoss(output, target):
    x = output.shape[0]
    output = nn.Softmax(dim=1)(output)
    target = torch.nn.functional.one_hot(target, 5)
    return (output - target) / x

def derivative_sigmoid(input):
    return nn.Sigmoid()(input)*(1-nn.Sigmoid()(input))

def derivative_relu(input):
    #Tmp lambda func as needs to be done in cpp anyway
    tmp = input.clone().detach()
    return tmp.apply_(lambda x : 1 if x > 0 else 0)

def derivative_linear(input):
    dlin_dw = input 
    dlin_db = 1.0 
    return dlin_dw, dlin_db

def adam(m_t_minus_1, v_t_minus_1,val,  grad, lr = 0.008, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0, step = 1):
    m_t = betas[0]*m_t_minus_1 + (1-betas[0])*grad 
    v_t = betas[1]*v_t_minus_1 + (1-betas[1])*(grad**2)
    
    m_t_correct = m_t / (1-(betas[0]**step))
    v_t_correct = v_t / (1-(betas[1]**step))

    res = val - lr * (m_t_correct/(torch.sqrt(v_t_correct)+eps))

    return res, m_t, v_t

def nesterov(codes, grad, b_t, lr, momentum, step):    
    b_t = grad.clone().detach()
    grad = grad + (momentum * b_t)
    codes = codes - (lr*grad)
    return codes, b_t



# def update_last_layer_no_autograd(model, input, target, criterion, n_iter):
#     output = model(input)
#     dL_doutput = derivative_BCELoss(output,target)
#     doutput_dsigmoid = derivative_sigmoid(model[:-1](input)) 
#     dsigmoid_dW, dsigmoid_db = derivative_linear(model[0](input))
#     dL_dW = torch.matmul( torch.transpose(dL_doutput * doutput_dsigmoid, 1, 0), dsigmoid_dW)
#     dL_db = torch.sum(dL_doutput * doutput_dsigmoid * dsigmoid_db,0)
#     print("No autograd \n")
#     print(dL_dW)
#     print(model[1].weight.data)
#     W, _, _ = adam(0,0, model[1].weight.data, dL_dW)
#     b, _, _ = adam(0,0, model[1].bias.data, dL_db)
#     return W,b
    
def update_last_layer_no_autograd(model, input, target, criterion, n_iter):
    output = model(input)
    # 0 BCELoss
    # 1 Sigmoid 
    # 2 Linear 
    # 3 ReLU
    chain_rule = []
    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i], nn.Sigmoid):
            chain_rule.append(1)
        elif isinstance(model[i], nn.Linear):
            chain_rule.append(2)
            break
        elif isinstance(model[i], nn.ReLU):
            chain_rule.append(3)
    
    dL_doutput = derivative_BCELoss(output,target)
   
    for x,i in enumerate(chain_rule):
        if i == 1:
            #print("\nDerivative sigmoid: ")
            #print(model[:-(1+x)](input))
            #print(derivative_sigmoid(model[:-(1+x)](input)))
            dL_doutput *= derivative_sigmoid(model[:-(1+x)](input))
        elif i == 2:
            #print("Derivative linear: ")
            dsigmoid_dW, dsigmoid_db = derivative_linear(model[:-(1+x)](input))
            dL_dW = torch.matmul( torch.transpose(dL_doutput, 1, 0), dsigmoid_dW)
            dL_db = torch.sum(dL_doutput * dsigmoid_db,0)
            break
        elif i == 3:
            #print("Derivative relu: ")
            dL_doutput *= derivative_relu(model[:-(1+x)](input))
 
    W, _, _ = adam(0,0, model[1].weight.data, dL_dW)
    b, _, _ = adam(0,0, model[1].bias.data, dL_db)

    #print(W)
    return W,b


def update_hidden_weights_no_autograd(model_mods, inputs, codes, lambda_w, n_iter):
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

        # # for it in range(n_iter):

        dL_doutput = derivative_MSELoss(lin(nmod(c_in)),c_out.detach())
        if lambda_w > 0.0:
            pass 

        doutput_dW, doutput_db = derivative_linear(nmod(c_in))

        dL_dW = torch.matmul( torch.transpose(dL_doutput, 1, 0), doutput_dW)
        dL_db = torch.sum(dL_doutput * doutput_db,0)
        
        W, _, _ = adam(0,0, lin.weight.data, dL_dW)
        b, _, _ = adam(0,0, lin.bias.data, dL_db)

        with torch.no_grad():
            lin.weight = nn.Parameter(W)
            lin.bias = nn.Parameter(b)
    

def update_codes_no_autograd(codes, model_mods, targets, criterion, mu, lambda_c, n_iter, lr):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
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
            #loss_fn = lambda x: criterion(x, targets)
            dL_doutput = (1/mu) * derivative_BCELoss(lin(nmod(codes[-l])),targets) #+ derivative_MSELoss(codes, codes_initial)
            #nmod is sequential 
            for i in range(len(nmod)-1,-1,-1):
                m = nmod[i]
                if isinstance(m, nn.Sigmoid):
                    print(nmod[:(i)])
                    dL_doutput *= derivative_sigmoid(nmod[:i](codes[-l]))
                elif isinstance(m, nn.Linear):
                    dL_doutput = torch.matmul( dL_doutput, m.weight.data)
                elif isinstance(m, nn.ReLU):
                    print(nmod[:(i)])
                    dL_doutput *= derivative_relu(nmod[:i](codes[-l]))

            dL_dc = dL_doutput
            codes[-l], _ = nesterov(codes[-l], dL_dc, 0, 0.3, 0.9, 1)
                
            
            
        else:       # intermediate layers
            #loss_fn = lambda x: mu*F.mse_loss(x, codes[-l+1].detach())
            
            dL_doutput = derivative_MSELoss(lin(nmod(codes[-l])),codes[-l+1].detach()) #+ derivative_MSELoss(codes, codes_initial)
            doutput_dnmodcin = lin.weight.data
            if isinstance(nmod, nn.ReLU):
                dnmodcin_dcin = derivative_relu(codes[-l])
            elif isinstance(nmod, nn.Sigmoid):
                dnmodcin_dcin = derivative_sigmoid(codes[-l])
            else:
                print("Later not imp yet")
                return
                
            dL_dc = torch.matmul(dL_doutput, doutput_dnmodcin) * dnmodcin_dcin
            codes[-l], _ = nesterov(codes[-l], dL_dc, 0, 0.3, 0.9, 1)