import os, sys 
sys.path.insert(0, os.path.abspath("../.."))
import argparse
import torch
import torch.nn as nn

from altmin import get_mods
from models import simpleNN
from dataset import TrackDatasetDelphes, ddict
from helper_methods import eval_model, train_using_SGD, train_using_altmin

 
parser = argparse.ArgumentParser(description='Online Alternating-Minimization with SGD')
parser.add_argument('--strategy', default = 'baseline', help='training strategy')
parser.add_argument('--lr-out', type=float, default=0.008, metavar='LR',
                    help='learning rate for last layer weights updates')
parser.add_argument('--lr-weights', type=float, default=0.008, metavar='LR',
                    help='learning rate for hidden weights updates')
parser.add_argument('--lr-half-epochs', type=int, default=8, metavar='LH',
                    help='number of epochs after which learning rate if halfed')
parser.add_argument('--mu', type=float, default=0.003, metavar='M',
                    help='initial mu parameter')
parser.add_argument('--d-mu', type=float, default=0.0/300, metavar='M',
                    help='increase in mu after every mini-batch')
parser.add_argument('--batch-size', type=int, default=200, metavar='B',
                    help='input batch size for training')
parser.add_argument('--lr-decay', type=float, default=1.0, metavar='LD',
                    help='learning rate decay factor per epoch (default: 1.0)')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--output-classes', type=int, default=1,
                    help='output between 0 and 1 or class label')
parser.add_argument('--dataset', type=str, choices=['cern','delphes'], default = 'cern',
                    help="Choose which dataset to train the data on" ) 
parser.add_argument('--above-eight', type=int, choices=[0,1], default = '0',
                    help="Choose which dataset to train the data on" ),
parser.add_argument('--cpp', type=int, choices=[0,1], default = '0',
                    help="Choose train method" ) 
args = parser.parse_args()

if __name__ == "__main__":
    # Save everything in a `ddict`
    SAV = ddict(args=args.__dict__)

    train_data = TrackDatasetDelphes("../datasets/delphes/TTbarFull/Train/train.pkl", args.above_eight)
    val_data = TrackDatasetDelphes("../datasets/delphes/TTbarFull/Val/val.pkl", args.above_eight)
    test_data = TrackDatasetDelphes("../datasets/delphes/TTbarFull/Test/test.pkl", args.above_eight)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5000, shuffle=True,num_workers=1)
    val_loader = torch.utils.data.DataLoader(train_data, batch_size=5000, shuffle=True,num_workers=1)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size=5000, shuffle=True,num_workers=1)

    model = simpleNN(5, [25,30], 1)
    criterion = nn.BCELoss()

    if args.strategy == 'altmin':
        #Ignore flatten for now
        
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': args.lr_weights},
                         scheduler=lambda epoch: 1/2**(epoch//args.lr_half_epochs))
        model = model[1:]
        #model[-1].optimizer.param_groups[0]['lr'] = args.lr_out
        model, train_acc, val_acc = train_using_altmin(model, train_loader,val_loader, criterion, 'models/initialise_model_on_TTFull_using_altmin_for_', args.epochs, args.cpp)

    elif args.strategy =='sgd': 
        model, train_acc, val_acc = train_using_SGD(model, train_loader,val_loader, criterion, 'models/initialise_model_on_TTFull_using_sgd_for_', args.epochs)
        
    SAV.train_acc = train_acc
    SAV.val_acc = val_acc
    SAV.eval_model = eval_model(model, test_loader, criterion, label=" - Training")

    torch.save(SAV, 'output/initialise_model_on_TTbarFullTrain_using_'+ args.strategy+'_for_'+str(args.epochs)+'_epochs.pt')
    torch.save(model.state_dict(), 'models/initialise_model_on_TTBarFullTrain_using_'+ args.strategy+'_for_'+str(args.epochs)+'_epochs.pt')
    