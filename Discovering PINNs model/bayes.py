
import sys
sys.path.append("..")
import torch
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import weight_init
import time
import math
from train import train_pinns
from loss import loss_func_pinns
import nn_models
from nn_activations import tanh, tanh_L1, atan, sin, sin_L1, cos, cos_L1, asinh, sigmoid, sigmoid_L1, recu, fixedswish, swish_L3
from train import relative_error
from copy import deepcopy


def train_evaluate(iter, conf, args, neuron_num_dict, activation_dict, data, device):

    #print(conf)

    layer_num = round(conf['layer_num'])
    neuron_num = neuron_num_dict[str(round(conf['neuron_num']))]
    activation = activation_dict[str(round(conf['activation']))]

    assert type(layer_num) == int
    assert type(neuron_num) == int

    #--------------------------------------------------------
    if args.BO_structure == 'BO-FcResNet':
        inner_layer_num = 2
        residual_layer_start = []
        residual_layer_end = []
        for i in range(math.floor((layer_num-1)/inner_layer_num)):
            residual_layer_start.append(i*inner_layer_num)
            residual_layer_end.append((i+1)*inner_layer_num)

        if layer_num-1 not in residual_layer_end:
            residual_layer_start.append(residual_layer_end[-1])
            residual_layer_end.append(layer_num-1)

        residual_layer = [residual_layer_start, residual_layer_end]

    else:
        assert args.BO_structure == 'BO-FcNet'
        residual_layer = [[],[]]

    # --------------------------------------------------------

    Net = nn_models.NET(io_dim = args.io_dim,
                        layer_num = layer_num,
                        residual_layer = residual_layer,
                        neuron_num = neuron_num,
                        activation = activation).to(device)

    print(neuron_num, layer_num, activation[0].__name__, activation[1])

    # Net.apply(weight_init.xavier_normal)
    # Net.apply(weight_init.xavier_uniform)
    # Net.apply(weight_init.kaiming_normal)
    Net.apply(weight_init.kaiming_uniform)

    print('Number of parameters: %d' % (sum(param.numel() for param in Net.parameters())))
    print('Activation function: %s' % activation[0].__name__)
    print('Activation function para_num: %s' % activation[1])

    start_time = time.time()
    # Train neural network
    net, _ = train_pinns(Net, data, args.equation, args, args.EPOCHS_ALL[0])
    elapsed = time.time() - start_time
    print('training time: %.2f' % (elapsed))

    res = loss_func_pinns(net, data.tr_set, args.equation).detach()
    print('loss: %.3e' % res)

    torch.save(net, args.save_dir + 'model/' + str(iter) + '_' + str(layer_num) +
               '_' + str(neuron_num) + '_' + activation[0].__name__ + '_' + str(activation[1]))

    if res.isnan() or res > 10000:
        res = torch.tensor(10000.0)

    # Error
    if args.equation in ['klein_gordon', 'burgers']:
        test_u = net(data.test.x).detach()
        error_u = relative_error(test_u, data.test.ua).detach()
        print('error_u: %.3e' % (error_u))

    elif args.equation in ['lame']:
        test_u = net(data.test.x).detach()
        test_u0 = test_u[:, 0:1]
        test_u1 = test_u[:, 1:2]
        error_u0 = relative_error(test_u0, data.test.u0a).detach()
        error_u1 = relative_error(test_u1, data.test.u1a).detach()
        error_u = error_u0 + error_u1
        print('error_u: %.3e' % (error_u))

    else:
        raise NameError('The equation is not defined')

    if error_u.isnan() or error_u > 10000:
        error_u = torch.tensor(10000.0)

    return (-res).cpu().numpy(), error_u.cpu().numpy(), elapsed


def bayes_main(args, data, device, seed):

    optimizer = BayesianOptimization(f=None,
                                     pbounds={'layer_num': (3, 11), 'neuron_num': (1, 16), 'activation': (1, 13)},
                                     verbose=2,
                                     random_state=seed,)

    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    neuron_num_dict = {'1': 20, '2': 22, '3': 24, '4': 26, '5': 28, '6': 30, '7': 32, '8': 34, '9': 36,
                       '10': 38, '11': 40, '12': 42, '13': 44, '14': 46, '15': 48, '16': 50 }
    activation_dict = {'1':[tanh, 0], '2':[atan, 0], '3':[sin, 0], '4':[cos, 0],
                       '5':[asinh, 0], '6':[sigmoid, 0], '7':[recu, 0], '8':[fixedswish, 0],
                       '9':[tanh_L1, 1], '10':[sin_L1, 1], '11':[cos_L1, 1], '12':[sigmoid_L1, 1], '13':[swish_L3, 1]}

    opt_history = np.zeros([args.iter_num, 3])

    confs = []
    for i in range(args.iter_num):
        print('-------------------------------------------------------')
        conf = optimizer.suggest(utility)
        target, error_u, elapsed = train_evaluate(i, conf, args, neuron_num_dict, activation_dict, data, device)
        opt_history[i, 0] = target
        opt_history[i, 1] = error_u
        opt_history[i, 2] = elapsed
        optimizer.register(params=conf, target=target)
        #print(target, conf)
        confs.append(conf)

    fits = opt_history[:, 0]
    top_inds = np.argsort(fits)[-1:-1-args.topconf_num:-1]
    top_confs = [confs[i] for i in top_inds]

    np.save(args.save_dir + 'opt_history.npy', opt_history)

    # Multiple evaluations
    reeval_results = np.zeros([args.topconf_num, args.reeval_num, 3])

    for i, tc in enumerate(top_confs):
        print('################################################################')
        for j in range(args.reeval_num):
            print('-------------------------------------------------------')
            target, error_u, elapsed = train_evaluate('reeval'+str(args.reeval_num*i+j), tc, args, neuron_num_dict, activation_dict, data, device)
            reeval_results[i, j, 0] = target
            reeval_results[i, j, 1] = error_u
            reeval_results[i, j, 2] = elapsed

    mean_fits = []
    for i in range(len(top_confs)):
        fits = deepcopy(reeval_results[i, :, 0])
        mean_fits.append(sum(fits)/len(fits))

    best_ind = mean_fits.index(max(mean_fits))
    best_conf = top_confs[best_ind]

    best_layer_num = round(best_conf['layer_num'])
    best_neuron_num = neuron_num_dict[str(round(best_conf['neuron_num']))]
    best_activation = activation_dict[str(round(best_conf['activation']))]

    print('*****************************')
    print('best_layer_num:', best_layer_num, '\n'
          'best_neuron_num:', best_neuron_num, '\n'
          'best_activation:', best_activation[0].__name__ + '_' + str(best_activation[1]) )

    return best_layer_num, best_neuron_num, best_activation
