import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from altmin import get_mods, get_codes, update_codes, update_last_layer_, update_hidden_weights_adam_
from altmin import scheduler_step, post_processing_step
from altmin import test
from altmin import get_devices, ddict, load_dataset


# Training settings
parser = argparse.ArgumentParser(
    description='Online Alternating-Minimization with SGD')
parser.add_argument('--model', default='feedforward', metavar='M',
                    help='name of model: `feedforward`, `binary` or `LeNet` (default: `feedforward`)')
parser.add_argument('--n-hidden-layers', type=int, default=2, metavar='L',
                    help='number of hidden layers (default: 2; ignored for LeNet)')
parser.add_argument('--n-hiddens', type=int, default=100, metavar='N',
                    help='number of hidden units (default: 100; ignored for LeNet)')
parser.add_argument('--dataset', default='mnist', metavar='D',
                    help='name of dataset')
parser.add_argument('--data-augmentation', action='store_true', default=False,
                    help='enables data augmentation')
parser.add_argument('--batch-size', type=int, default=200, metavar='B',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--n-iter-codes', type=int, default=5, metavar='N',
                    help='number of internal iterations for codes optimization')
parser.add_argument('--n-iter-weights', type=int, default=1, metavar='N',
                    help='number of internal iterations in learning weights')
parser.add_argument('--lr-codes', type=float, default=0.3, metavar='LR',
                    help='learning rate for codes updates')
parser.add_argument('--lr-out', type=float, default=0.008, metavar='LR',
                    help='learning rate for last layer weights updates')
parser.add_argument('--lr-weights', type=float, default=0.008, metavar='LR',
                    help='learning rate for hidden weights updates')
parser.add_argument('--lr-half-epochs', type=int, default=8, metavar='LH',
                    help='number of epochs after which learning rate if halfed')
parser.add_argument('--no-batchnorm', action='store_true', default=False,
                    help='disables batchnormalization')
parser.add_argument('--lambda_c', type=float, default=0.0, metavar='L',
                    help='codes sparsity')
parser.add_argument('--lambda_w', type=float, default=0.0, metavar='L',
                    help='weight sparsity')
parser.add_argument('--mu', type=float, default=0.003, metavar='M',
                    help='initial mu parameter')
parser.add_argument('--d-mu', type=float, default=0.0/300, metavar='M',
                    help='increase in mu after every mini-batch')
parser.add_argument('--postprocessing-steps', type=int, default=0, metavar='N',
                    help='number of Carreira-Perpinan post-processing steps after training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before saving test performance (if set to zero, it does not save)')
parser.add_argument('--log-first-epoch', action='store_true', default=False,
                    help='whether or not it should test and log after every mini-batch in first epoch')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()


# Check cuda
device, num_gpus = get_devices(
    "cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu", seed=args.seed)


# Load data and model
model_name = args.model.lower()
if model_name == 'feedforward' or model_name == 'binary':
    model_name += '_' + str(args.n_hidden_layers) + 'x' + str(args.n_hiddens)
file_name = 'output/save_' + os.path.basename(__file__).split('.')[0] + '_' + model_name +\
    '_' + args.dataset + '_' + str(args.seed) + '.pt'

print('\nOnline alternating-minimization with sgd')
print('* Loading dataset {}'.format(args.dataset))
print('* Loading model {}'.format(model_name))
print('     BatchNorm: {}'.format(not args.no_batchnorm))

if args.model.lower() == 'feedforward' or args.model.lower() == 'binary':
    from altmin import simpleNN

    train_loader, test_loader, n_inputs = load_dataset(
        args.dataset, batch_size=args.batch_size, conv_net=False)

    model = simpleNN(10, [30, 20], 1).to(device)


if __name__ == "__main__":

    model = get_mods(model, optimizer='Adam', optimizer_params={'lr': args.lr_weights},
                     scheduler=lambda epoch: 1/2**(epoch//args.lr_half_epochs))
    model[-1].optimizer.param_groups[0]['lr'] = args.lr_out

    print(model)
    in_tensor = torch.rand(1, 10).double()
    # Easy to imp in c++
    # Need to imp the basic layer functions and then can do the harder calcs
    # At least I know my plan for tomororow
    res = torch.matmul(in_tensor, torch.transpose(
        model[1].weight, 0, 1)) + model[1].bias
    res_two = model[1](in_tensor)

    print(res)
    print(res_two)
    model.a()
