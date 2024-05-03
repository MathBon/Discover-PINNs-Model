
from train import relative_error
from loss import loss_func_pinns
import math
import numpy as np
import sys


def cal_fitness_pinns(genome, trained_model, data, equation):

    res = loss_func_pinns(trained_model, data.tr_set, equation).detach()  
    genome.fitness = (-res).cpu().numpy() 

    if math.isnan(genome.fitness):
        genome.fitness = np.array(-sys.float_info.max)  

    if equation in ['klein_gordon', 'burgers']:
        test_u = trained_model(data.test.x).detach()
        genome.error = relative_error(test_u, data.test.ua).detach().cpu().numpy()
    elif equation in ['lame']:
        test_u = trained_model(data.test.x).detach()
        test_u0 = test_u[:, 0:1]
        test_u1 = test_u[:, 1:2]
        error0 = relative_error(test_u0, data.test.u0a).detach().cpu().numpy()
        error1 = relative_error(test_u1, data.test.u1a).detach().cpu().numpy()
        genome.error = error0 + error1
    else:
        raise NameError('The equation is not defined')

