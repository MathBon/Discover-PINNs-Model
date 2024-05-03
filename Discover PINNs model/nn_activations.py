
import torch


def sin(x):
    return torch.sin(x)

def sin_L1(param, x):
    return torch.sin(param[0] * x)

def asinh(x):
    return torch.asinh(x)

def cos(x):
    return torch.cos(x)

def cos_L1(param, x):
    return torch.cos(param[0] * x)

def tanh(x):
    return torch.tanh(x)

def tanh_L1(param, x):
    return torch.tanh(param[0] * x)

def atan(x):
    return torch.atan(x)

def sigmoid(x):
    return torch.sigmoid(x)

def sigmoid_L1(param, x):
    return torch.sigmoid(param[0] * x)

def fixedswish(x):
    return x * torch.sigmoid(x)

def swish_L3(param, x):
    return x * torch.sigmoid(param[0] * x)

def recu(x):
    return torch.relu(x)**3

def expn_p_1(x):
    return 1.0 + torch.exp(-x)

#--------------------------------------------------------------
def asinh_X_cos(x):
    return torch.asinh(x) * torch.cos(x)


def cos_X_erf_L1L4(param, x):
    return (param[0] * torch.cos(x)) * torch.erf(param[1] * x)


def sin_L0L1(param, x):
    return param[0] * torch.sin(param[1] * x)


def tanh_X_cos_L1L2(param, x):
    return (param[0] * torch.tanh(param[1] * x)) * torch.cos(x)


def x_D_exp_p_expn(x):
    return x / (torch.exp(x) + torch.exp(-x))


def sin_O_tanh_L2(param, x):
    return torch.sin(torch.tanh(param[0] * x))


def fixedswish_O_erf_L2(param, x):
    return fixedswish(torch.erf(param[0] * x))


def tanh_O_asinh_L0L1(param, x):
    return param[0]*torch.tanh(param[1]*torch.asinh(x))


def fixedswish_S_erf_L4(param, x):
    return fixedswish(x) - torch.erf(param[0] * x)


def cos_L0(param, x):
    return param[0] * torch.cos(x)


def nx_S_sin_L3(param, x):
    return -x - param[0] * torch.sin(x)


def sin_D_expn_p_1_L1(param, x):
    return (param[0] * torch.sin(x)) / expn_p_1(x)


def sigmoid_X_tanh(x):
    return sigmoid(x) * torch.tanh(x)


def erf_O_fixedswish_L1L2(param, x):
    return torch.erf(param[0] * fixedswish(param[1] * x))


def x_X_sigmoid_L1L3(param, x):
    return (param[0] * x) * sigmoid(param[1] * x)


def exp_p_expn_min_atan_L1L2L4(param, x):
    return torch.minimum(param[0] * (torch.exp(param[1]*x) + torch.exp(-param[1]*x)), torch.atan(param[2]*x) )


def tanh_O_fixedswish_L0(param, x):
    return param[0]*torch.tanh(fixedswish(x))


def tanh_O_fixedswish_L1(param, x):
    return torch.tanh(param[0]*fixedswish(x))


def tanh_L0(param, x):
    return param[0]*torch.tanh(x)


def sin_L0(param, x):
    return param[0]*torch.sin(x)


def tanh_O_fixedswish_L0L1(param, x):
    return param[0]* torch.tanh(param[1]* fixedswish(x))


def asinh_O_sin_L0L1L2(param, x):
    return param[0]*torch.asinh(param[1]*torch.sin(param[2]*x))


def atan_O_sin_L1(param, x):
    return torch.atan(param[0] * torch.sin(x))


def sin_O_fixedswish(x):
    return torch.sin(fixedswish(x))


def sin_X_sigmoid_L2(param, x):
    return torch.sin(param[0] * x) * torch.sigmoid(x)


def sigmoid_L0L1(param, x):
    return param[0] * torch.sigmoid(param[1] * x)


def neg_O_tanh_L0L2(param, x):
    return param[0]*(-torch.tanh(param[1]*x))


def fixedswish_L1(param, x):
    return fixedswish(param[0]*x)


def atan_O_fixedswish_L1L2(param, x):
    return torch.atan(param[0]*fixedswish(param[1] * x))


def asinh_O_fixedswish_L0L2(param, x):
    return param[0]*torch.asinh(fixedswish(param[1]*x))


def cos_X_atan_X_sigmoid_L0L4L7(param, x):
    return param[0]*(torch.cos(x)*(param[1]*torch.atan(x))*sigmoid(param[2]*x))


def atan_X_neg_O_sigmoid(x):
    return torch.atan(x) * (- torch.sigmoid(x))


