
import numpy as np
import math
import random
from copy import deepcopy
from config import config
from crossover import crossover
from mutation import mutation
from initialize import init_pop
import multiprocessing as mp
from parallel import device_assign


def evolution(case, data, seed):
    """ Evolution"""

    beforemerge_pophistory = []
    aftermerge_pophistory = []

    # Initialize population
    new_pop = init_pop()

    for ig in range(config.GENERATIONS_NUM):

        genome_id = mp.Value('i', 0)

        #---------------------------------------------------------------------------------------------
        '''Train (with config.EPOCHS_ALL[g] epochs) and evaluate all the genomes in current generation(g)'''
        new_pop = device_assign(case, new_pop, data, ig, genome_id, config.EPOCHS_ALL[ig], seed)
        beforemerge_pophistory.append(new_pop)
        np.save(case['save_dir'] + 'beforemerge_pophistory.npy', np.array(beforemerge_pophistory, dtype=object))

        if ig > 0:
            '''Re-train (with config.EPOCHS_ALL[g] epochs) and re-evaluate the top-config.POPULATION_SIZE[g]*config.RETRAIN_RATIO 
               genomes in previous generation(g-1)'''
            sorted_pop = sorted(pop, key=lambda k: k.fitness, reverse=True)
            pop = sorted_pop[:math.ceil(config.POPULATION_SIZE[ig]*config.RETRAIN_RATIO)]
            pop = device_assign(case, pop, data, ig, genome_id, config.EPOCHS_ALL[ig], seed)
            merged_pop = pop + new_pop  # Merge the genomes
            sorted_merged_pop = sorted(merged_pop, key=lambda k: k.fitness, reverse=True)
            new_pop = sorted_merged_pop[:config.POPULATION_SIZE[ig]]

        aftermerge_pophistory.append(new_pop)
        np.save(case['save_dir'] + 'aftermerge_pophistory.npy', np.array(aftermerge_pophistory, dtype=object))

        pop = deepcopy(new_pop)

        # ---------------------------------------------------------------------------------------------
        '''Select, Cross, Mutate'''
        if ig < config.GENERATIONS_NUM - 1:
            new_pop = []
            pop_fitness = [i.fitness for i in pop]

            # Linear ranking
            ranking = sorted(range(len(pop_fitness)), key=lambda k: pop_fitness[k])
            positions = [ranking.index(j) for j in range(len(pop_fitness))]
            pop_prob = [(i + 1) / len((pop_fitness)) for i in positions]
            adjusted_pop_prob = [i / sum(pop_prob) for i in pop_prob]

            for i in range(math.ceil(config.POPULATION_SIZE[ig+1] / 2)):
                # Select
                assert config.POPULATION_SIZE[ig] == len(pop)
                parents_ids = np.random.choice(np.arange(config.POPULATION_SIZE[ig]), size=2, replace=False, p=adjusted_pop_prob)

                parent_1 = deepcopy(pop[parents_ids[0]])
                parent_2 = deepcopy(pop[parents_ids[1]])

                if random.random() < config.CROSS_MUTATE_RATIO:
                    # Cross
                    child_1, child_2 = crossover(parent_1, parent_2)
                else:
                    # Mutate
                    child_1 = mutation(parent_1)
                    child_2 = mutation(parent_2)

                new_pop.append(child_1)
                new_pop.append(child_2)

            new_pop = new_pop[:config.POPULATION_SIZE[ig+1]]

    # ---------------------------------------------------------------------------------------------
    # Find the best genome in the last generation with multiple evaluations

    sorted_pop = sorted(pop, key=lambda k: k.fitness, reverse=True)
    pop = sorted_pop[:config.TOP_N]

    top_genomes_copies = []
    for ip in range(len(pop)):
        for _ in range(config.TOP_EVAL_N):
            top_genomes_copies.append(deepcopy(pop[ip]))

    genome_id = mp.Value('i', 0)
    top_genomes_copies = device_assign(case, top_genomes_copies, data, 'reeval', genome_id, config.EPOCHS_ALL[ig], seed)

    ori_g_ids = [str(tgc.ori_generation) + '-' + str(tgc.ori_id) for tgc in top_genomes_copies]
    ori_g_ids = list(set(ori_g_ids))

    mean_fits = []
    for ogi in ori_g_ids:
        copies_fits = [tgc.fitness for tgc in top_genomes_copies if str(tgc.ori_generation) + '-' + str(tgc.ori_id) == ogi]
        mean_fits.append(sum(copies_fits)/len(copies_fits))

    best_ind = mean_fits.index(max(mean_fits))
    best_ogi = ori_g_ids[best_ind]

    best_genome = [p for p in pop if str(p.ori_generation) + '-' + str(p.ori_id) == best_ogi][0]

    return best_genome

