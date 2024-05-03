
import random
from config import config
from copy import deepcopy


def mutation(genome):

    ''' The schematic diagram about 4-layers neural network '''
    #  ········· (0) Input
    #  XXXXXXXXX (Linear transform + activation function)
    #  ········· (1)
    #  XXXXXXXXX (Linear transform + activation function)
    #  ········· (2)
    #  XXXXXXXXX (Linear transform + activation function)
    #  ········· (3)
    #  XXXXXXXXX (Linear transform)
    #  ········· (4) Output

    genome_before_mutation = deepcopy(genome)

    ''' Mutation of structure '''
    print('----------Structure before mutation:', genome.structure)

    if random.random() < config.NEURON_NUM_MUTATION_RATE:
        # Mutate neuron_num
        if genome.structure['neuron_num'] == max(config.NEURON_CANDIDATES):
            genome.structure['neuron_num'] = config.NEURON_CANDIDATES[-2]
        elif genome.structure['neuron_num'] == min(config.NEURON_CANDIDATES):
            genome.structure['neuron_num'] = config.NEURON_CANDIDATES[1]
        else:
            neuron_index = config.NEURON_CANDIDATES.index(genome.structure['neuron_num'])
            new_neuron_index = random.choice([neuron_index - 1, neuron_index + 1])
            genome.structure['neuron_num'] = config.NEURON_CANDIDATES[new_neuron_index]

    if random.random() < config.LAYER_NUM_MUTATION_RATE:
        # Mutate layer_num

        layer_num_mutation_type = random.choice(['add', 'substract'])

        if genome.structure['layer_num'] == config.LAYER_BOUND[0]:
            layer_num_mutation_type = 'add'
        elif genome.structure['layer_num'] == config.LAYER_BOUND[-1]:
            layer_num_mutation_type = 'substract'

        if layer_num_mutation_type == 'add': # choose the insert position of the new layer, and adjust the corresponding residual layers
            positions_candidates = list(range(genome.structure['layer_num'] + 1))
            insert_position = random.choice(positions_candidates)
            if insert_position in genome.structure['residual_layer'][0] and \
               insert_position not in genome.structure['residual_layer'][1]:

                block_index = genome.structure['residual_layer'][0].index(insert_position)
                insert_type = random.choice(['before', 'after'])
                if insert_type == 'before': # insert a layer before the start position of the corresponding block
                    for i in range(len(genome.structure['residual_layer'][0])):
                        if i >= block_index:
                            genome.structure['residual_layer'][0][i] += 1
                            genome.structure['residual_layer'][1][i] += 1
                else:
                    assert insert_type == 'after' # insert a layer after the start position of the corresponding block
                    for i in range(len(genome.structure['residual_layer'][0])):
                        if i == block_index:
                            genome.structure['residual_layer'][1][i] += 1
                        elif i > block_index:
                            genome.structure['residual_layer'][0][i] += 1
                            genome.structure['residual_layer'][1][i] += 1


            elif insert_position in genome.structure['residual_layer'][1] and \
                 insert_position not in genome.structure['residual_layer'][0]:

                block_index = genome.structure['residual_layer'][1].index(insert_position)
                insert_type = random.choice(['before', 'after'])
                if insert_type == 'before':  # insert a layer before the end position of the corresponding block
                    for i in range(len(genome.structure['residual_layer'][0])):
                        if i == block_index:
                            genome.structure['residual_layer'][1][i] += 1
                        elif i > block_index:
                            genome.structure['residual_layer'][0][i] += 1
                            genome.structure['residual_layer'][1][i] += 1
                else:
                    assert insert_type == 'after'  # insert a layer after the end position of the corresponding block
                    for i in range(len(genome.structure['residual_layer'][0])):
                        if i > block_index:
                            genome.structure['residual_layer'][0][i] += 1
                            genome.structure['residual_layer'][1][i] += 1


            elif insert_position in genome.structure['residual_layer'][0] and \
                 insert_position in genome.structure['residual_layer'][1]:

                block_index = genome.structure['residual_layer'][1].index(insert_position)
                insert_type = random.choice(['before', 'middle', 'after'])
                if insert_type == 'before':  # insert a layer before the end position of the previous block
                    for i in range(len(genome.structure['residual_layer'][0])):
                        if i == block_index:
                            genome.structure['residual_layer'][1][i] += 1
                        elif i > block_index:
                            genome.structure['residual_layer'][0][i] += 1
                            genome.structure['residual_layer'][1][i] += 1

                elif insert_type == 'middle': # insert a layer between the two adjacent blocks
                    for i in range(len(genome.structure['residual_layer'][0])):
                        if i > block_index:
                            genome.structure['residual_layer'][0][i] += 1
                            genome.structure['residual_layer'][1][i] += 1

                else:
                    assert insert_type == 'after'  # insert a layer after the start position of the latter block
                    for i in range(len(genome.structure['residual_layer'][0])):
                        if i == block_index + 1:
                            genome.structure['residual_layer'][1][i] += 1
                        elif i > block_index + 1:
                            genome.structure['residual_layer'][0][i] += 1
                            genome.structure['residual_layer'][1][i] += 1

            else:
                assert insert_position not in genome.structure['residual_layer'][0] and \
                       insert_position not in genome.structure['residual_layer'][1]

                if len(genome.structure['residual_layer'][0]) > 0:
                    for i in range(len(genome.structure['residual_layer'][0])):
                        if genome.structure['residual_layer'][0][i] > insert_position:
                            genome.structure['residual_layer'][0][i] += 1
                        if genome.structure['residual_layer'][1][i] > insert_position:
                            genome.structure['residual_layer'][1][i] += 1

            genome.structure['layer_num'] = genome.structure['layer_num'] + 1


        else:
            assert layer_num_mutation_type == 'substract'

            positions_candidates = list(range(genome.structure['layer_num'] - 1))
            remove_position = random.choice(positions_candidates)

            for i in range(len(genome.structure['residual_layer'][0])):
                if genome.structure['residual_layer'][0][i] > remove_position:
                    genome.structure['residual_layer'][0][i] -= 1
                if genome.structure['residual_layer'][1][i] > remove_position:
                    genome.structure['residual_layer'][1][i] -= 1

            for i in range(len(genome.structure['residual_layer'][0])-1, -1, -1): # remove the block which includes no layer
                if genome.structure['residual_layer'][0][i] == genome.structure['residual_layer'][1][i]:
                    genome.structure['residual_layer'][0].pop(i)
                    genome.structure['residual_layer'][1].pop(i)

            genome.structure['layer_num'] = genome.structure['layer_num'] - 1


    if random.random() < config.RESIDUAL_MUTATION_RATE:
        # Mutate the residual_layer

        # Three kinds of mutations of residual_layer
        residual_layer_mutation_type = random.choice(['change', 'add', 'remove'])

        max_block_num = genome.structure['layer_num'] - 1  # Maximum number of blocks
        if len(genome.structure['residual_layer'][0]) == max_block_num:
            residual_layer_mutation_type = 'remove'  # Saturated residual layers, forcing a remove mutation...
        elif len(genome.structure['residual_layer'][0]) == 0:
            residual_layer_mutation_type = 'add' # No residual layer, forcing an add mutation...


        if residual_layer_mutation_type == 'change':  # do not change the number of blocks, only change the position of start or end of the chosen block
            block_index = list(range(len(genome.structure['residual_layer'][0])))
            random.shuffle(block_index)

            n_change = 0
            for ib in block_index:

                change_type = ['start-shallower', 'start-deeper', 'end-shallower', 'end-deeper']
                random.shuffle(change_type)

                for ct in change_type:
                    n_change += 1
                    if ct=='start-shallower':
                        if genome.structure['residual_layer'][0][ib] > 0: # there is at least one layer ahead
                            if ib > 0: # there is at least one block ahead
                                if genome.structure['residual_layer'][1][ib-1] - genome.structure['residual_layer'][0][ib-1] > 1: # if the number of inner layers in previous block > 1
                                    new_start = random.randint(genome.structure['residual_layer'][0][ib-1] + 1,
                                                               genome.structure['residual_layer'][0][ib] - 1 ) # start position of previous block +1 ~ start position of current block -1
                                    genome.structure['residual_layer'][0][ib] = new_start
                                    if new_start < genome.structure['residual_layer'][1][ib-1]:  # if new start position of current block < end position of previous block
                                        genome.structure['residual_layer'][1][ib-1] = new_start  # change the end position of previous block
                                    break
                                elif genome.structure['residual_layer'][1][ib-1] - genome.structure['residual_layer'][0][ib-1] == 1 and \
                                     genome.structure['residual_layer'][1][ib-1] < genome.structure['residual_layer'][0][ib]:
                                    genome.structure['residual_layer'][0][ib] = random.randint(genome.structure['residual_layer'][1][ib-1],
                                                                                               genome.structure['residual_layer'][0][ib]-1)
                                    break
                            else:
                                genome.structure['residual_layer'][0][ib] = random.randint(0, genome.structure['residual_layer'][0][ib]-1) # 0 ~ start position of current block -1
                                break

                    elif ct=='start-deeper':
                        if genome.structure['residual_layer'][1][ib] - genome.structure['residual_layer'][0][ib] > 1:
                            genome.structure['residual_layer'][0][ib] = random.randint(genome.structure['residual_layer'][0][ib]+1,
                                                                                       genome.structure['residual_layer'][1][ib]-1)
                            break

                    elif ct=='end-shallower':
                        if genome.structure['residual_layer'][1][ib] - genome.structure['residual_layer'][0][ib] > 1:
                            genome.structure['residual_layer'][1][ib] = random.randint(genome.structure['residual_layer'][0][ib]+1,
                                                                                       genome.structure['residual_layer'][1][ib]-1)
                            break

                    else:
                        assert ct=='end-deeper'
                        if genome.structure['residual_layer'][1][ib] < genome.structure['layer_num']-1:  # there are at least two layers behind
                            if ib < len(genome.structure['residual_layer'][0])-1:  # there is at least one block behind
                                if genome.structure['residual_layer'][1][ib+1] - genome.structure['residual_layer'][0][ib+1] > 1: # if the number of inner layers in latter block > 1
                                    new_end = random.randint(genome.structure['residual_layer'][1][ib] + 1,
                                                             genome.structure['residual_layer'][1][ib+1] - 1 ) # end position of current block +1 ~ end position of latter block -1
                                    genome.structure['residual_layer'][1][ib] = new_end
                                    if new_end > genome.structure['residual_layer'][0][ib+1]:
                                        genome.structure['residual_layer'][0][ib+1] = new_end  # change the start position of latter block
                                    break
                                elif genome.structure['residual_layer'][1][ib+1] - genome.structure['residual_layer'][0][ib+1] == 1 and \
                                     genome.structure['residual_layer'][1][ib] < genome.structure['residual_layer'][0][ib+1]:
                                    genome.structure['residual_layer'][1][ib] = random.randint(genome.structure['residual_layer'][1][ib]+1,
                                                                                               genome.structure['residual_layer'][0][ib+1])
                                    break
                            else:
                                genome.structure['residual_layer'][1][ib] = random.randint(genome.structure['residual_layer'][1][ib]+1, genome.structure['layer_num']-1)  # end position of current block +1 ~ layer_num-1
                                break

                else:
                    if n_change == len(block_index) * len(change_type):
                        print('Residual layer has not been changed')
                    continue
                break


        elif residual_layer_mutation_type == 'add':  # add a block

            blocks_ranges = [[i, j] for i, j in zip(genome.structure['residual_layer'][0], genome.structure['residual_layer'][1])]
            ranges_candidates = []

            if len(blocks_ranges) > 0:
                for i in range(len(blocks_ranges)):
                    if i==0 and blocks_ranges[i][0] > 0:  # a range before the existing first block
                        ranges_candidates.append([0, blocks_ranges[i][0]])
                    if i > 0 and blocks_ranges[i][0]-blocks_ranges[i-1][1] > 0:  # a range between the existing blocks
                        ranges_candidates.append([blocks_ranges[i-1][1], blocks_ranges[i][0]])
                    if i==len(blocks_ranges)-1 and blocks_ranges[i][1] < genome.structure['layer_num']-1:  # a range after the existing last block
                        ranges_candidates.append([blocks_ranges[i][1], genome.structure['layer_num']-1])
            else:
                ranges_candidates.append([0, genome.structure['layer_num'] - 1])

            if len(ranges_candidates) > 0:
                add_range = random.choice(ranges_candidates)
                add_block = random.sample(list(range(add_range[0], add_range[1]+1)), 2)
                add_block.sort()

                genome.structure['residual_layer'][0].append(add_block[0])
                genome.structure['residual_layer'][1].append(add_block[1])

                genome.structure['residual_layer'][0].sort()
                genome.structure['residual_layer'][1].sort()
            else:
                assert len(ranges_candidates) == 0
                print('Block has not been added')

        else:
            assert residual_layer_mutation_type == 'remove'
            assert len(genome.structure['residual_layer'][0]) > 0
            remove_index = random.choice(range(len(genome.structure['residual_layer'][0])))
            genome.structure['residual_layer'][0].pop(remove_index)
            genome.structure['residual_layer'][1].pop(remove_index)

    print('++++++++++Structure after mutation:', genome.structure)

    ''' Mutation of activation function '''

    if random.random() < config.ACTIVATION_MUTATION_RATE:
        genome.activation.mutate()


    genome.fitness = None
    genome.error = None
    genome.generation = None
    genome.id = None
    genome.para_num = None


    if genome.compare(genome_before_mutation):
        genome.ori_generation = genome_before_mutation.ori_generation
        genome.ori_id = genome_before_mutation.ori_id
    else:
        genome.ori_generation = None
        genome.ori_id = None


    return genome
