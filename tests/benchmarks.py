# class benchmarkNN(nn.Module):
#     def __init__(self, input_dim, hidden_layers, output_dim):
#         super(benchmarkNN, self).__init__()
#         self.features = ()
#         self.classifier = nn.Sequential()
#         linear1 = nn.Linear(input_dim, hidden_layers[0])
#         nn.init.kaiming_uniform_(linear1.weight, nonlinearity='relu')
#         self.classifier.add_module(str(len(self.classifier)), linear1)
#         self.classifier.add_module(str(len(self.classifier)), nn.ReLU())
#         linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
#         nn.init.kaiming_uniform_(linear2.weight, nonlinearity='relu')
#         self.classifier.add_module(str(len(self.classifier)), linear2)
#         self.classifier.add_module(str(len(self.classifier)), nn.ReLU())

#         linear3 = nn.Linear(hidden_layers[1], hidden_layers[2])
#         nn.init.kaiming_uniform_(linear3.weight, nonlinearity='relu')
#         self.classifier.add_module(str(len(self.classifier)), linear3)
#         self.classifier.add_module(str(len(self.classifier)), nn.ReLU())

#         linear4 = nn.Linear(hidden_layers[2], hidden_layers[3])
#         nn.init.kaiming_uniform_(linear4.weight, nonlinearity='relu')
#         self.classifier.add_module(str(len(self.classifier)), linear4)
#         self.classifier.add_module(str(len(self.classifier)), nn.ReLU())


#         linear5 = nn.Linear(hidden_layers[3], output_dim)
#         nn.init.xavier_uniform_(linear5.weight)
#         self.classifier.add_module(str(len(self.classifier)), linear5)
#         self.classifier.add_module(str(len(self.classifier)), nn.Sigmoid())
#         self.n_inputs = input_dim
#         self.n_outputs = output_dim
#         self.classifier.double()

#     def forward(self, x):
#         #x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         #x = torch.flatten(x)
#         return x

 # def test_benchmark(self):
    #     n_iter = 1
    #     lr = 0.3
    #     mu = 0.003
    #     criterion = nn.BCELoss()

    #     #model = benchmarkNN(200, [300,500,400,150],1)
    #     model = simpleNN(200, [400,300], 1)

    #     model = get_mods(model, optimizer='Adam', optimizer_params={'lr': 0.008},
    #                  scheduler=lambda epoch: 1/2**(epoch//8))
    #     model[-1].optimizer.param_groups[0]['lr'] = 0.008
    #     model = model[1:]

    #     neural_network = fast_altmin.NeuralNetworkBCE()
    #     fast_altmin.create_model_class(model, neural_network, 5000, 0)

    #     import time 
    #     tot = 0.0
    #     for it in range(250):    
    #         in_tensor = torch.rand(5000,200,dtype = torch.double)
    #         targets = torch.round(torch.rand(5000, 1, dtype=torch.double))
    #         output_cpp = neural_network.get_codes(in_tensor, True) 
    #         neural_network.update_codes(targets)
    #         start = time.time()
    #         neural_network.update_weights_parallel(in_tensor, targets)
    #         end=time.time()
    #         tot+=(end-start)
    #     print("parallel: "+str(tot))
    #     tot = 0.0
        
    #     for it in range(250):    
    #         in_tensor = torch.rand(5000,200,dtype = torch.double)
    #         targets = torch.round(torch.rand(5000, 1, dtype=torch.double))
    #         output_cpp = neural_network.get_codes(in_tensor, True)
    #         neural_network.update_codes(targets)
    #         start = time.time()
    #         neural_network.update_weights_not_parallel(in_tensor, targets)
    #         end=time.time()
    #         tot+=(end-start)
    #     print("not parallel: "+str(tot))