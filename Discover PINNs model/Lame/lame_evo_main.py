
import sys
sys.path.append("..")
import torch
import numpy as np
import random
import multiprocessing
from config import config
from evolution import evolution
from random_search import random_search
from lame_data import Data


def lame_evo():
    #set seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    pattern = 'evolution'
    # pattern = 'random_search'

    case = {'method': 'PINNs',
            'equation': 'lame',
            'io_dim': [2, 2] }

    """ data type """
    if config.DTYPE == 'float16':
        torch.set_default_dtype(torch.float16)
    elif config.DTYPE == 'float32':
        torch.set_default_dtype(torch.float32)
    elif config.DTYPE == 'float64':
        torch.set_default_dtype(torch.float64)

    data = Data(radius = [1.0, 2.0],
                elasticity_modulus = 2.1,
                poisson_ratio = 0.25,
                pressure = 23.0,
                shear_force = 3.0,
                nx_tr_interior = [100, 100],
                nx_tr_boundary = 200,
                nx_va_interior = [10, 10],
                nx_va_boundary = 10,
                nx_te = [200, 200],
                lambda_boundary = 10.0)

    if pattern == 'evolution':
        case['save_dir']= 'evo_save/'
        best_genome = evolution(case, data, seed)
    elif pattern == 'random_search':
        case['save_dir'] = 'rand_save/'
        best_genome = random_search(case, data, seed)
    else:
        raise NameError('The pattern is not defined')

    with open(case['save_dir'] + 'best_genome.txt', 'w') as file:
        file.write(best_genome.__str__())

    best_net = torch.load(case['save_dir'] + 'model/' + str(best_genome.generation) + '-'+ str(best_genome.id) )
    print('################################################')
    print('best_genome', best_genome)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    lame_evo()

