
import torch


class NET(torch.nn.Module):
    def __init__(self, io_dim, layer_num, residual_layer, neuron_num, activation):
        super(NET, self).__init__()
        self.io_dim = io_dim
        self.layer_num = layer_num
        self.residual_layer = residual_layer
        self.neuron_num = neuron_num
        self.activation = activation[0]
        self.actpara_num = activation[1]

        layers = list()
        layers.append(torch.nn.Linear(self.io_dim[0], self.neuron_num))
        for i in range(self.layer_num-2):
            layers.append(torch.nn.Linear(self.neuron_num, self.neuron_num))
        layers.append(torch.nn.Linear(self.neuron_num, self.io_dim[1]))
        self.layers = torch.nn.Sequential(*layers)

        if self.actpara_num > 0:
            self.act_param = torch.nn.Parameter(torch.ones(self.layer_num-1, self.actpara_num))   # layer-wise
        else:
            assert self.actpara_num == 0


    def forward(self, x):

        for i in range(self.layer_num - 1):
            if i in self.residual_layer[1]:
                x = x + temp

            if i in self.residual_layer[0]:
                if i == 0:
                    padding = torch.zeros(x.shape[0], self.neuron_num - self.io_dim[0]).to(x.device)
                    temp = torch.cat((x, padding), 1)
                else:
                    temp = x

            if self.actpara_num > 0:
                x = self.activation(self.act_param[i, :], self.layers[i](x))
            else:
                x = self.activation(self.layers[i](x))

        if self.layer_num - 1 in self.residual_layer[1]:
            x = x + temp

        # if self.actpara_num > 0:
        #     print(self.act_param)

        return self.layers[-1](x)

