import torch.nn as nn
import torch

class simpleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(simpleNN, self).__init__()
        self.features = ()
        self.classifier = nn.Sequential()
        linear1 = nn.Linear(input_dim, hidden_layers[0])
        nn.init.kaiming_uniform_(linear1.weight, nonlinearity='relu')
        self.classifier.add_module(str(len(self.classifier)), linear1)
        self.classifier.add_module(str(len(self.classifier)), nn.ReLU())
        linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        nn.init.kaiming_uniform_(linear2.weight, nonlinearity='relu')
        self.classifier.add_module(str(len(self.classifier)), linear2)
        self.classifier.add_module(str(len(self.classifier)), nn.ReLU())
        linear3 = nn.Linear(hidden_layers[1], output_dim)
        nn.init.xavier_uniform_(linear3.weight)
        self.classifier.add_module(str(len(self.classifier)), linear3)
        self.classifier.add_module(str(len(self.classifier)), nn.Sigmoid())
        self.n_inputs = input_dim
        self.n_outputs = output_dim
        self.classifier.double()

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #x = torch.flatten(x)
        return x

class LinMod(nn.Linear):
    '''Linear modules with or without batchnorm, all in one module
    '''
    def __init__(self, n_inputs, n_outputs, bias=False, batchnorm=False):
        super(LinMod, self).__init__(n_inputs, n_outputs, bias=bias)
        if batchnorm:
            self.bn = nn.BatchNorm1d(n_outputs, affine=True)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.batchnorm = batchnorm
        self.bias_flag = bias

    def forward(self, inputs):
        outputs = super(LinMod, self).forward(inputs)
        if hasattr(self, 'bn'):
            outputs = self.bn(outputs)
        return outputs

    def extra_repr(self):
        return '{n_inputs}, {n_outputs}, bias={bias_flag}, batchnorm={batchnorm}'.format(**self.__dict__)


class FFNet(nn.Module):
    '''Feed-forward all-to-all connected network
    '''
    def __init__(self, n_inputs, n_hiddens, n_hidden_layers=2, n_outputs=10, nlin=nn.ReLU, bias=False, batchnorm=False):
        super(FFNet, self).__init__()

        self.features = ()  # Skip convolutional features

        self.classifier = nn.Sequential(nn.Linear(n_inputs, n_hiddens, bias=bias), nlin())
        for i in range(n_hidden_layers - 1):
            self.classifier.add_module(str(2 * i + 2), nn.Linear(n_hiddens, n_hiddens, bias=bias))
            self.classifier.add_module(str(2 * i + 3), nlin())
        self.classifier.add_module(str(len(self.classifier)), nn.Linear(n_hiddens, n_outputs))

        self.batchnorm = batchnorm
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.classifier.double()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(model, optimizer, train_loader, criterion=nn.CrossEntropyLoss(), log_times=10):
    model.train()
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Outputs to terminal
        if batch_idx % (len(train_loader) // log_times) == 0:
            print('  training progress: {}/{} ({:.0f}%)\tloss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, data_loader, criterion=nn.CrossEntropyLoss(), label=''):
    '''Compute model accuracy
    '''
    model.eval()
    device = next(model.parameters()).device

    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            test_loss += criterion(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    accuracy = float(correct) / len(data_loader.dataset)
    test_loss /= len(data_loader)  # loss function already averages over batch size
    if label:
        print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            label, test_loss, correct, len(data_loader.dataset), 100. * accuracy))
    return accuracy