import torch
import numpy as np

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import fast_altmin
from altmin import get_codes, get_mods, FFNet, update_codes, update_last_layer_, update_hidden_weights_adam_

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
    loss = (1/mu)*loss_fn(output) + F.mse_loss(codes, codes_targets)
    if lambda_c>0.0:
        loss += (lambda_c/mu)*codes.abs().mean()
    return loss

def update_codes(codes, model_mods, targets, criterion, mu, lambda_c, n_iter, lr):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
    for l in range(1, len(codes)+1):
        idx = id_codes[-l]

        codes[-l].requires_grad_(True)
        optimizer = optim.SGD([codes[-l]], lr=lr, momentum=0.9, nesterov=True)
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
            optimizer.zero_grad()
            loss = compute_codes_loss(codes[-l], nmod, lin, loss_fn, codes_initial, mu, lambda_c)
            loss.backward()
            optimizer.step()

    return codes


def update_last_layer_(mod_out, inputs, targets, criterion, n_iter):
    for it in range(n_iter):
        mod_out.optimizer.zero_grad()
        outputs = mod_out(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        mod_out.optimizer.step()


def update_hidden_weights_adam_(model_mods, inputs, codes, lambda_w, n_iter):
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
            lin.optimizer.zero_grad()
            loss = F.mse_loss(lin(nmod(c_in)), c_out.detach())
            if lambda_w > 0.0:
                loss += lambda_w*lin.weight.abs().mean()
            loss.backward()
            lin.optimizer.step()


T = 10000
X = np.array(range(T))
Y = np.sin(3.5 * np.pi * X / T) 

model = FFNet(1, n_hiddens=1024, n_hidden_layers=2, n_outputs = 1, batchnorm=False, bias=True).double()
#model = get_mods(model)
model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
                     scheduler=lambda epoch: 1/2**(epoch//8))
model[-1].optimizer.param_groups[0]['lr'] = 0.008

model = model[1:]
print(model)

criterion = nn.MSELoss()
init_vals = True
optimizer = optim.Adam(model.parameters(), lr=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
for epoch in range(3):
    print("Epoch: "+str(epoch))
    for index in range(0,9000):
        data = torch.tensor(np.array(X[index:index+100])).double()
        targets = torch.tensor(np.array(Y[index:index+100])).double()
        data=data.reshape(100,1)
        targets = targets.reshape(100,1)
        
        # print(data)
        # print(targets)
        # print(data.shape)
        # print(targets.shape)
        # model.train()
        # with torch.no_grad():
        #     outputs, codes = get_codes(model, data)
        
        # # (2) Update codes
        # fast_altmin.cf_update_codes(codes,model,targets, criterion, 0.003, 0.0, 1, 0.3 )  

        # if init_vals:
        #     momentum_dict = fast_altmin.store_momentums(model, True)
        # # (3) Update weights
        # fast_altmin.cf_update_weights_parallel(model, data.detach(), codes,targets, 0, 1, 0.008, momentum_dict, init_vals , nn.CrossEntropyLoss())
        # init_vals = False
        # model.train()
        # with torch.no_grad():
        #     outputs, codes = get_codes(model, data)

        # # (2) Update codes
        # codes = update_codes(codes, model, targets, criterion, 0.003, lambda_c=0.0, n_iter=5, lr=0.3 )

        # # (3) Update weights
        # update_last_layer_(model[-1], codes[-1], targets, criterion, n_iter=1)
        # update_hidden_weights_adam_(model, data, codes, lambda_w=0.0, n_iter=1)

        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()


#test model 
outputs = []
for x in range(10000):
    data = torch.tensor([X[index]]).double()
    res = model(data).detach().numpy()
    outputs.append(res[0])


plt.figure(figsize = (10,6))
plt.plot(Y, linewidth=2, marker='.', label="sin")
plt.plot(outputs, linewidth=2, marker='.', label="sin")
plt.show()
#plt.plot(data_two, linewidth=2, marker='.', label=labels[1])
# plt.ylabel('Test accuracy')
# plt.xlabel('Epoch')