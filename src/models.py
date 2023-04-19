import torch.nn as nn


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
