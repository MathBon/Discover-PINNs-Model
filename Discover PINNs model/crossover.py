
import random
from genome import Genome
from config import config


def crossover(parent_1, parent_2):

    child_1 = Genome()
    child_2 = Genome()      

    if random.random() < config.CROSSOVER_RATE:
        # Single-point Crossover
        child_1.structure = parent_1.structure
        child_1.activation = parent_2.activation
        
        child_2.structure = parent_2.structure
        child_2.activation = parent_1.activation

    else: 
        child_1.structure = parent_1.structure
        child_1.activation = parent_1.activation
        
        child_2.structure = parent_2.structure
        child_2.activation = parent_2.activation


    if parent_1.compare(parent_2):
        assert child_1.compare(child_2)
        child_1.ori_generation = parent_1.ori_generation
        child_1.ori_id = parent_1.ori_id
        child_2.ori_generation = parent_2.ori_generation
        child_2.ori_id = parent_2.ori_id

    else:
        if child_1.compare(parent_1):
            child_1.ori_generation = parent_1.ori_generation
            child_1.ori_id = parent_1.ori_id
        elif child_1.compare(parent_2):
            child_1.ori_generation = parent_2.ori_generation
            child_1.ori_id = parent_2.ori_id
        else:
            child_1.ori_generation = None
            child_1.ori_id = None

        if child_2.compare(parent_1):
            child_2.ori_generation = parent_1.ori_generation
            child_2.ori_id = parent_1.ori_id
        elif child_2.compare(parent_2):
            child_2.ori_generation = parent_2.ori_generation
            child_2.ori_id = parent_2.ori_id
        else:
            child_2.ori_generation = None
            child_2.ori_id = None


    return child_1, child_2

