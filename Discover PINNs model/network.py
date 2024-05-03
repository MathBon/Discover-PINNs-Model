
import torch


class NET(torch.nn.Module):
    def __init__(self, genome, io_dim):
        super(NET, self).__init__()
        self.io_dim = io_dim
        self.layer_num = genome.structure['layer_num']
        self.residual_layer = genome.structure['residual_layer']
        self.neuron_num = genome.structure['neuron_num']
        self.activation = genome.activation

        layers = list()
        layers.append(torch.nn.Linear(self.io_dim[0], self.neuron_num))
        for i in range(self.layer_num-2):
            layers.append(torch.nn.Linear(self.neuron_num, self.neuron_num))
        layers.append(torch.nn.Linear(self.neuron_num, self.io_dim[1]))
        self.layers = torch.nn.Sequential(*layers)

        self.max_params = sum([2 if len(node.children) == 0 else 1 for node in self.activation.list_nodes()])
        self.act_param = torch.nn.Parameter(torch.ones(self.layer_num-1, len(self.activation.param_edges)))  # layer-wise

    def forward(self, x):

        param = torch.ones(self.layer_num - 1, self.max_params).to(x.device)

        k = 0
        for i in range(self.max_params):
            if i in self.activation.param_edges:
                param[:, i] = self.act_param[:, k]
                k += 1

        for i in range(self.layer_num - 1):
            if i in self.residual_layer[1]:
                x = x + temp

            if i in self.residual_layer[0]:
                if i == 0:
                    padding = torch.zeros(x.shape[0], self.neuron_num - self.io_dim[0]).to(x.device)
                    temp = torch.cat((x, padding), 1)
                else:
                    temp = x

            x = self.activation.call(self.layers[i](x), param[i, :])

        if self.layer_num - 1 in self.residual_layer[1]:
            x = x + temp

        #print(param)

        return self.layers[-1](x)

