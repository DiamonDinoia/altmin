
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from altmin import get_mods, get_codes
from altmin import scheduler_step
from models import simpleNN
from dataset import get_devices
from altmin import get_mods, get_codes


from manual_altmin import update_last_layer_manual, update_codes_manual, update_hidden_weights_adam_manual, store_momentums
from altmin import update_last_layer_, update_codes, update_hidden_weights_adam_
from control_flow import cf_get_codes, cf_update_codes, cf_update_hidden_weights, cf_update_last_layer

#Evaluation method to measure performance of the model during training and testing.
def eval_model(model, test_loader, criterion=nn.BCELoss(), label=""):
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            if len(target.size())<2 and isinstance(criterion, nn.BCELoss):
                target = target.reshape(len(target),1).double()
            test_loss += criterion(output, target)
            if isinstance(criterion, nn.BCELoss):
                output = output.round()
            elif isinstance(criterion, nn.CrossEntropyLoss):
                output = output.argmax(1)
            correct += float(torch.sum(torch.eq(output, target)))
            
    accuracy = correct/len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)  # loss function already averages over batch size
    if label:
        print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            label, test_loss, correct, len(test_loader.dataset), 100. * accuracy))
    return accuracy


def train_using_SGD(model, train_loader, test_loader, criterion, file_name, epochs = 30, mu = 0.003, lr_decay = 1.0, log_interval = 100, no_cuda = False, seed = 1):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)    

    # Check cuda
    device, num_gpus = get_devices("cuda:0" if not no_cuda and torch.cuda.is_available() else "cpu", seed=seed)

    train_acc = []
    test_acc = []

    for epoch in range(1, epochs+1):
        print('\nEpoch {} of {}'.format(epoch, epochs))
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, targets = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)

        # (1) Forward
            model.train()
            with torch.no_grad():
                outputs, codes = cf_get_codes(model, data)

            # (2) Update codes
            codes = update_codes(codes, model, targets, criterion, mu, lambda_c=lambda_c, n_iter=n_iter_codes, lr=lr_codes)
                        
            # (3) Update weights
            update_last_layer_(model[-1], codes[-1], targets, criterion, n_iter=n_iter_weights)
            update_hidden_weights_adam_(model, data, codes, lambda_w=lambda_w, n_iter=n_iter_weights)

            loss = criterion(output,target)
            loss.backward()
            optimizer.step()

            # Outputs to terminal
            if batch_idx % log_interval == 0:
                print(' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        scheduler.step()
        # Print performances
        train_acc += [eval_model(model, train_loader, criterion, label=" - Training")]

        if test_loader != "-1":
            test_acc += [eval_model(model, test_loader, criterion, label=" - Test")]
        
        if epoch % 5 == 0:
            if file_name != "-1":
                torch.save(model.state_dict(), file_name+str(epoch)+'_epochs.pt')
    
    return model,train_acc,test_acc
    
    
def train_using_altmin( model, train_loader, test_loader, criterion, file_name, epochs, manual, mu = 0.003, lambda_c = 0.0, n_iter_codes = 5, lr_codes = 0.3, n_iter_weights=1, lambda_w=0.0, log_first_epoch=False,  log_interval=100, save_interval=1000, d_mu=0.0, no_cuda = False, seed = 1):
    
    # Check cuda
    device, num_gpus = get_devices("cuda:0" if not no_cuda and torch.cuda.is_available() else "cpu", seed=seed)
    mu_max = 10*mu

    train_acc = []
    test_acc = []
    momentum_dict = store_momentums(model, False) 
    if manual == 1:
         momentum_dict = store_momentums(model, True) 

    init_vals = True
    for epoch in range(1, epochs+1):
        
        print('\nEpoch {} of {}. mu = {:.4f}, lr_out = {}'.format(epoch, epochs, mu, model[-1].scheduler.get_lr()))

        for batch_idx, (data, targets) in enumerate(train_loader):
            #Ignore last batch until I fix adam.
            if data.shape[0] != 5000:
                continue
            data, targets = data.to(device), targets.to(device)
            if len(targets.shape) == 1:
                targets = targets.reshape(len(targets),1).double()

            if manual == 0:
                # (1) Forward
                model.train()
                with torch.no_grad():
                    outputs, codes = get_codes(model, data)

                # (2) Update codes
                codes = update_codes(codes, model, targets, criterion, mu, lambda_c=lambda_c, n_iter=n_iter_codes, lr=lr_codes)
                            
                # (3) Update weights
                
                update_last_layer_(model[-1], codes[-1], targets, criterion, n_iter=n_iter_weights)
                
                update_hidden_weights_adam_(model, data, codes, lambda_w=lambda_w, n_iter=n_iter_weights)
                
            else:
                # (1) Forward
                model.train()
                with torch.no_grad():
                    outputs, codes = cf_get_codes(model, data)

                # # (2) Update codes
                # codes = update_codes_manual(codes, model, targets, criterion, mu, lambda_c, n_iter_codes, momentum_dict)
                            
                # # (3) Update weights
                # update_last_layer_manual(model[-1], codes[-1], targets, criterion, n_iter_weights, momentum_dict)
                
                # update_hidden_weights_adam_manual(model, data, codes, lambda_w, n_iter_weights, momentum_dict)
               

                if init_vals:
                    momentum_dict["0.code_m"] = torch.zeros(codes[0].shape, dtype=torch.double)
                    momentum_dict["0.code_v"] = torch.zeros(codes[0].shape, dtype=torch.double)
                    momentum_dict["2.code_m"] = torch.zeros(codes[1].shape, dtype=torch.double)
                    momentum_dict["2.code_v"] = torch.zeros(codes[1].shape, dtype=torch.double)

                cf_update_codes(codes, model, targets, nn.BCELoss(), momentum_dict, init_vals, mu=0.003, lambda_c=0.0, n_iter=n_iter_codes, lr=0.3 )

                cf_update_last_layer(model, codes[-1], targets, 0, n_iter_weights, 0.008, momentum_dict, init_vals )

                cf_update_hidden_weights(model, data, codes, 0,n_iter_weights, 0.008, momentum_dict, init_vals)

                init_vals = False

                


            # Outputs to terminal
            if batch_idx % log_interval == 0:
                if str(criterion) == 'BCELoss()':
                    targets = targets.reshape(len(targets),1).double()
                loss = criterion(outputs,targets)
                print(' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            # Increment mu
            if mu < mu_max:
                mu = mu + d_mu

        #scheduler_step(model)

        # Print performances
        train_acc += [eval_model(model, train_loader, criterion, label=" - Training")]

        if test_loader != "-1":
            test_acc += [eval_model(model, test_loader, criterion, label=" - Test")]
        
      
            
    return model, train_acc, test_acc


def test_eval_sets(model, sgd_file_path, altmin_file_path, eval_sets, lr_weights, lr_half_epochs, lr_out):
    accs_sgd = []
    for eval_set in eval_sets:
        model = simpleNN(5, [25,30],1)

        model.load_state_dict(torch.load(sgd_file_path))
        acc = eval_model(model, torch.utils.data.DataLoader(eval_set, batch_size=5000, shuffle=True,num_workers=1) )
        accs_sgd.append(acc)
        
    accs_altmin = []

    for eval_set in eval_sets:
        model = simpleNN(5, [25,30],1)

        # Expose model modules that has_codes
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': lr_weights},
                        scheduler=lambda epoch: 1/2**(epoch//lr_half_epochs))
        model[-1].optimizer.param_groups[0]['lr'] = lr_out
        model.load_state_dict(torch.load(altmin_file_path))
        acc = eval_model(model, torch.utils.data.DataLoader(eval_set, batch_size=5000, shuffle=True,num_workers=1) )
        accs_altmin.append(acc)

    return accs_sgd, accs_altmin

