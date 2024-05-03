
import torch
import copy
from config import config
from fitness import cal_fitness_pinns
from network import NET
from train import train_pinns
import random
import numpy as np
import weight_init
import time
import multiprocessing as mp


def parallel(case, pops_perdevice, i_device, data, ig, genome_id, epochs, seed, queue):
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    """ data type """
    if config.DTYPE == 'float16':
        torch.set_default_dtype(torch.float16)
    elif config.DTYPE == 'float32':
        torch.set_default_dtype(torch.float32)
    elif config.DTYPE == 'float64':
        torch.set_default_dtype(torch.float64)

    device = torch.device(f'cuda:{i_device}')
    data.data_to_gpu(device)

    for genome in pops_perdevice:
        print('################################################')
        print('GPU-ID:', i_device)


        if len(genome.structure['residual_layer'][0]) > 0:
            for i, r in enumerate(genome.structure['residual_layer'][0]):
                assert r < genome.structure['residual_layer'][1][i]
            a0 = copy.deepcopy(genome.structure['residual_layer'][0])
            a0.sort()
            assert genome.structure['residual_layer'][0] == a0
            a1 = copy.deepcopy(genome.structure['residual_layer'][1])
            a1.sort()
            assert genome.structure['residual_layer'][1] == a1

            assert genome.structure['layer_num'] - 1 >= genome.structure['residual_layer'][1][-1]


        assert len(set(genome.activation.param_edges)) == len(genome.activation.param_edges)
        max_param_num = sum([2 if len(node.children) == 0 else 1 for node in genome.activation.list_nodes()])
        assert sum(i > max_param_num-1 for i in genome.activation.param_edges) == 0
        assert sum(i < 0 for i in genome.activation.param_edges) == 0
        assert len(genome.activation.param_edges) <= max_param_num

        Net = NET(genome, case['io_dim']).to(device)

        # Net.apply(weight_init.xavier_normal)
        # Net.apply(weight_init.xavier_uniform)
        # Net.apply(weight_init.kaiming_normal)
        Net.apply(weight_init.kaiming_uniform)

        genome.para_num = sum(p.numel() for p in Net.parameters())

        if case['method'] == 'PINNs':
            trained_model, train_history = train_pinns(Net, data, case['equation'], config, epochs)
            cal_fitness_pinns(genome, trained_model, data, case['equation'])
        else:
            raise NameError('The method is not defined')

        with genome_id.get_lock():
            genome.id = genome_id.value
            genome_id.value += 1

        genome.generation = ig

        if genome.ori_generation == None:
            assert genome.ori_id == None
            genome.ori_generation = genome.generation
            genome.ori_id = genome.id

        torch.save(trained_model, case['save_dir'] + 'model/' + str(genome.generation) + '-' + str(genome.id) )

        print(genome)

        queue.put(genome)


def device_assign(case, pop, data, ig, genome_id, epochs, seed):

    genomes_num = len(pop)

    ngenomes_perdevice = round(genomes_num / config.DEVICE_NUM)

    genomes_device = []  # genomes on each device
    for k in range(config.DEVICE_NUM - 1):
        genomes_device.append(pop[k * ngenomes_perdevice: (k + 1) * ngenomes_perdevice])
    genomes_device.append(pop[(config.DEVICE_NUM - 1) * ngenomes_perdevice: genomes_num])

    queue = mp.Queue()

    jobs = []
    pop = []
    for i_device in range(config.DEVICE_NUM):
        p = mp.Process(target=parallel, args=(case, genomes_device[i_device], i_device, data, ig, genome_id, epochs, seed, queue))
        jobs.append(p)
        p.start()

    n_get = 0
    while True:
        time.sleep(1.0)
        if not queue.empty():
            pop.append(queue.get())  # the index of genome in pop may be different with the id of the genome
            n_get += 1
        if n_get == genomes_num:
            break

    for j in jobs:
        j.join()

    return pop

