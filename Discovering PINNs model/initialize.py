
import random
from config import config
from genome import Genome
from DPAF import init_actfun
import math


def init_structure(layer_num):

    if random.random() < 0.5:  # random residual connections
        block_num = random.randint(0, layer_num-1)  # It is fully connected neural network if block_num == 0
        residual_layer_start = random.sample(range(0, layer_num-1), block_num)  # residual_layer_start == [] if block_num == 0
        residual_layer_start.sort()
        residual_layer_end = []

        for i in range(len(residual_layer_start)):
            if i < len(residual_layer_start) - 1:
                residual_layer_end.append(random.randint(residual_layer_start[i] + 1, residual_layer_start[i+1]))
            else:
                assert i == len(residual_layer_start) - 1
                residual_layer_end.append(random.randint(residual_layer_start[i] + 1, layer_num - 1))

    else:  # normal (regular) residual connections
        inner_layer_num = min(layer_num-1, random.randint(0, config.INNER_LAYER_NUM_THOLD)) # the number of layers in one block; It is fully connected neural network if inner_layer_num == 0
        residual_layer_start = []
        residual_layer_end = []

        if inner_layer_num > 0:
            for i in range(math.floor((layer_num-1)/inner_layer_num)):
                residual_layer_start.append(i*inner_layer_num)
                residual_layer_end.append((i+1)*inner_layer_num)

            if layer_num-1 not in residual_layer_end:
                residual_layer_start.append(residual_layer_end[-1])
                residual_layer_end.append(layer_num-1)

    residual_layer = [residual_layer_start, residual_layer_end]

    return residual_layer


def init_pop():
    
    pop = []

    # configurations
    for i in range(config.POPULATION_SIZE[0]):

        genome = Genome()

        genome.structure['layer_num'] = random.randint(config.LAYER_BOUND[0], config.LAYER_BOUND[1])
        genome.structure['neuron_num'] = random.choice(config.NEURON_CANDIDATES)
        genome.structure['residual_layer'] = init_structure(genome.structure['layer_num'])

        genome.activation = init_actfun(actfun = None, mutations_num = 0)

        pop.append(genome)

    return pop


if __name__ == '__main__':
    init_pop()

