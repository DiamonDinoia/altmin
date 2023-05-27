import torch
from torchvision import datasets, transforms
import pandas as pd

class TrackDatasetDelphes(torch.utils.data.Dataset):
    """ TrackDataset pytorch dataset
    Takes dataset_dir as a string input
    """
    # Training features used by model, for new ones, define them in dataset generator
    training_features = ['trk_eta',
                         'trk_phi',
                         'trk_pt',
                         'trk_z0',
                         "trk_dz"
                         ]
        
    # Binary target features
    target_feature = ['trk_PU']

    def __init__(self, dataset_dir, pt_above_8 = False, transform=None ):
        self.dataset_dir = dataset_dir
        self.dataframe = pd.read_pickle(dataset_dir)
        

        if pt_above_8:
            print("ao")
            self.dataframe  = self.dataframe [self.dataframe.trk_pt >= 8]

        self.transform = transform

        if self.transform:
            self.dataframe = self.transform(self.dataframe)

        self.X_data = torch.from_numpy(self.dataframe[self.training_features].to_numpy(dtype='float'))
        self.targets = torch.from_numpy(self.dataframe[self.target_feature].to_numpy(dtype='long')).flatten()
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.X_data[idx] , self.targets[idx]

def load_dataset(namedataset='cern', batch_size=200, data_augmentation=False, conv_net=False, num_workers=1, local = True):
    if namedataset == 'mnist':

        DIR_DATASET = '~/data'

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]

        if not conv_net:
            transform_list.append(transforms.Lambda(lambda x: x.view(x.size(1) * x.size(2))))

        transform = transforms.Compose(transform_list)

        trainset = datasets.MNIST(DIR_DATASET, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = datasets.MNIST(DIR_DATASET, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=True, num_workers=num_workers)

        classes = tuple(range(10))
        n_inputs = 784

    elif namedataset == 'delphes':
        training_data = TrackDatasetDelphes("../../datasets/TTBarFullTrain.pkl")
        validation_data = TrackDatasetDelphes("../../datasets/TTBarFullVal.pkl")
        testing_data = TrackDatasetDelphes("../../datasets/TTBarFullTest.pkl")
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        
        n_inputs = 5

        return train_loader, val_loader, test_loader, n_inputs
    elif namedataset == 'delphes_smear':
        training_data = TrackDatasetDelphes("../../datasets/TTBarFullSmearTrain.pkl")
        validation_data = TrackDatasetDelphes("../../datasets/TTBarFullSmearVal.pkl")
        testing_data = TrackDatasetDelphes("../../datasets/TTBarFullSmearTest.pkl")
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        
        n_inputs = 5

        return train_loader, val_loader, test_loader, n_inputs

    else:
        raise ValueError('Dataset {} not recognized'.format(namedataset))

    print(n_inputs)
    print("hi")
    return train_loader, test_loader, n_inputs

class ddict(object):
    '''
    dd = ddict(lr=[0.1, 0.2], n_hiddens=[100, 500, 1000], n_layers=2)
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        return self.__dict__.__getitem__(item)

    def __repr__(self):
        return str(self.__dict__)
    
def get_devices(cuda_device="cuda:0", seed=1):
    device = torch.device(cuda_device)
    torch.manual_seed(seed)
    # Multi GPU?
    num_gpus = torch.cuda.device_count()
    if device.type != 'cpu':
        print('\033[93m'+'Using CUDA,', num_gpus, 'GPUs\033[0m')
        torch.cuda.manual_seed(seed)
    return device, num_gpus

