
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


def random_search(case, data, seed):
    """ Random search"""

    beforemerge_pophistory = []
    aftermerge_pophistory = []

    # Initialize population
    new_pop = init_pop()

    for ig in range(config.GENERATIONS_NUM):

        genome_id = mp.Value('i', 0)
        # ---------------------------------------------------------------------------------------------

        '''Train (with config.EPOCHS_ALL[g] epochs) and evaluate all the genomes in current generation(g)'''
        new_pop = device_assign(case, new_pop, data, ig, genome_id, config.EPOCHS_ALL[ig], seed)
        beforemerge_pophistory.append(new_pop)
        np.save(case['save_dir'] + 'beforemerge_pophistory.npy', np.array(beforemerge_pophistory, dtype=object))

        if ig > 0:
            '''Re-train (with config.EPOCHS_ALL[g] epochs) and re-evaluate the random config.POPULATION_SIZE[g]*config.RETRAIN_RATIO  
               genomes in previous generation(g-1)'''
            pop = random.sample(pop, math.ceil(config.POPULATION_SIZE[ig]*config.RETRAIN_RATIO))  # sampling without repetition
            pop = device_assign(case, pop, data, ig, genome_id, config.EPOCHS_ALL[ig], seed)
            merged_pop = pop + new_pop  # Merge the genomes
            new_pop = random.sample(merged_pop, config.POPULATION_SIZE[ig])

        aftermerge_pophistory.append(new_pop)
        np.save(case['save_dir'] + 'aftermerge_pophistory.npy', np.array(aftermerge_pophistory, dtype=object))

        pop = deepcopy(new_pop)

        # ---------------------------------------------------------------------------------------------
        '''Select, Cross, Mutate'''
        if ig < config.GENERATIONS_NUM - 1:
            new_pop = []

            for i in range(math.ceil(config.POPULATION_SIZE[ig+1] / 2)):
                # Select
                assert config.POPULATION_SIZE[ig] == len(pop)
                parents_ids = np.random.choice(np.arange(config.POPULATION_SIZE[ig]), size=2, replace=False)

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
    # Find the best genome in all the genomes with multiple evaluations

    candidates = []
    for ap in aftermerge_pophistory:
        candidates.append(deepcopy(sorted(ap, key=lambda k: k.fitness, reverse=True)[0])) # select the best genome in each generation

    genome_id = mp.Value('i', 0)
    candidates = device_assign(case, candidates, data, 'candidates', genome_id, config.EPOCHS_ALL[ig], seed)
    np.save(case['save_dir'] + 'best_candidates.npy', np.array(candidates, dtype=object))
    pop = sorted(candidates, key=lambda k: k.fitness, reverse=True)[:config.TOP_N]

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
