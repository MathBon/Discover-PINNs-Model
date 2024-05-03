
import torch
import math


def xavier_normal(m):
    if type(m) == torch.nn.Linear:
        # print(m)
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        torch.nn.init._no_grad_normal_(m.weight, 0., std)
        if m.bias is not None:
            torch.nn.init._no_grad_zero_(m.bias)


def xavier_uniform(m):
    if type(m) == torch.nn.Linear:
        # print(m)
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        torch.nn.init._no_grad_uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            torch.nn.init._no_grad_zero_(m.bias)


def kaiming_normal(m):
    if type(m) == torch.nn.Linear:
        # print(m)
        fan = torch.nn.init._calculate_correct_fan(m.weight, 'fan_in')
        std = math.sqrt(2.0 / float(fan))
        torch.nn.init._no_grad_normal_(m.weight, 0., std)
        if m.bias is not None:
            torch.nn.init._no_grad_zero_(m.bias)


def kaiming_uniform(m):
    if type(m) == torch.nn.Linear:
        # print(m)
        fan = torch.nn.init._calculate_correct_fan(m.weight, 'fan_in')
        std = math.sqrt(2.0 / float(fan))
        bound = math.sqrt(3.0) * std
        torch.nn.init._no_grad_uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            torch.nn.init._no_grad_zero_(m.bias)

