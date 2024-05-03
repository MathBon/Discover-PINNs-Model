

class config:

    DEVICE_NUM = 1  # Number of GPUs used

    OPTIM_TYPE = 'lbfgs'  

    TIME_LIMIT = 4.0  # training time limit of EPOCHS_ONCE epoches (setting according to different equations)  Klein-Gordon/Burgers:4.0; Lame:6.0 

    LEARNING_RATE = 1.0
    DTYPE = 'float64'


    #  for Klein-Gordon and Lame (evo-w/-DPSTE; random search)
    EPOCHS_ALL = [100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    EPOCHS_ONCE = 20
    POPULATION_SIZE = [1000, 250, 125, 85, 65, 50, 40, 30, 25, 20, 15, 15, 15, 10, 10]
    GENERATIONS_NUM = len(POPULATION_SIZE)    # number of generations

    #  for Klein-Gordon and Lame (evo-w/o-DPSTE)
    # EPOCHS_ALL = [5000, 5000, 5000, 5000, 5000, 5000]
    # EPOCHS_ONCE = 20
    # POPULATION_SIZE = [30, 30, 30, 30, 30, 30]
    # GENERATIONS_NUM = len(POPULATION_SIZE)

    #  for Burgers (evo-w/-DPSTE; random search)
    # EPOCHS_ALL = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2700, 3000]
    # EPOCHS_ONCE = 20
    # POPULATION_SIZE = [1000, 200, 100, 65, 50, 40, 35, 30, 25, 20, 20, 20, 15, 15, 15]
    # GENERATIONS_NUM = len(POPULATION_SIZE)

	#  for Burgers (evo-w/o-DPSTE)
    # EPOCHS_ALL = [3000, 3000, 3000, 3000, 3000, 3000]
    # EPOCHS_ONCE = 20
    # POPULATION_SIZE = [40, 40, 40, 40, 40, 40]
    # GENERATIONS_NUM = len(POPULATION_SIZE)



    TOP_N = 3   # number of candidates N_c    
    TOP_EVAL_N = 4  # number of times about evaluating each candidate N_e   

    CROSS_MUTATE_RATIO = 0.5  # R_cm

    CROSSOVER_RATE = 1.0   #  R_c

    NEURON_NUM_MUTATION_RATE = 0.3  #  R_n
    LAYER_NUM_MUTATION_RATE = 0.3   #  R_l
    RESIDUAL_MUTATION_RATE = 0.3    #  R_s
    ACTIVATION_MUTATION_RATE = 0.7  #  R_a

    INITIAL_BASEACT_RATIO = 0.2

    RETRAIN_RATIO = 0.25  #  proportion of elitists R_e

    LAYER_BOUND = [3, 11]
    NEURON_CANDIDATES = list(range(20, 50 + 1, 2))  # The lower bound should not be less than the dimension of input

    INNER_LAYER_NUM_THOLD = 5   
    ACT_PARA_THOLD = 3 # maximum number of learnable parameters in activation function  M_E
    ACT_MAX_NODES = 7  # maximum number of nodes in activation function M_N    

