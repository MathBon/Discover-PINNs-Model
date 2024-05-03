
import sys
sys.path.append("..")
import torch
import numpy as np
from burgers_data import Data
import argparse
from bayes import bayes_main


def main(seed):
    # Configurations
    parser = argparse.ArgumentParser(description='PINNs for solving Burgers equation')
    parser.add_argument('--BO_structure', type=str, default='BO-FcResNet', help='BO-FcNet or BO-FcResNet')
    parser.add_argument('--equation', type=str, default='burgers', help='the equation to be solved')
    parser.add_argument('--bounds_x', type=float, default= [[0.0, 1.0], [0.0, 1.0]],  help='spatial bounds')
    parser.add_argument('--bounds_t', type=float, default= [0.0, 2.0], help='temporal bounds')
    parser.add_argument('--nx_tr_initial', type=int, default=[15, 15], help='spatial size of training data at initial time')
    parser.add_argument('--nx_tr_interior', type=int, default=[10, 10], help='spatial size of training data in interior domain')
    parser.add_argument('--nt_tr_interior', type=int, default=20, help='temporal size of training data in interior domain')
    parser.add_argument('--nx_tr_boundary', type=int, default=[15,15], help='spatial size of training data on boundary')
    parser.add_argument('--nt_tr_boundary', type=int, default=30, help='temporal size of training data on boundary')
    parser.add_argument('--nx_te', type=int, default=[25, 25], help='spatial size of testing data')
    parser.add_argument('--nt_te', type=int, default=50, help='temporal size of testing data')
    parser.add_argument('--lambda_initial', type=float, default=100.0, help='penalty coefficient for initial loss term')
    parser.add_argument('--lambda_boundary', type=float, default=100.0, help='penalty coefficient for boundary loss term')
    parser.add_argument('--coef', type=float, default=0.1, help='coefficient in equation')
    parser.add_argument('--io_dim', type=int, default=[3, 1], help='dimensions of input and output')
    parser.add_argument('--OPTIM_TYPE', type=str, default='lbfgs', help='optimizer type')
    parser.add_argument('--EPOCHS_ALL', type=int, default=[3000], help='total number of training epochs')
    parser.add_argument('--EPOCHS_ONCE', type=int, default=100, help='number of training epochs before each evaluation')
    parser.add_argument('--LEARNING_RATE', type=float, default=1.0, help='learning rate')
    parser.add_argument('--TIME_LIMIT', type=float, default=20.0, help='threshold of training time between two evaluations')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether or not to use CUDA')
    parser.add_argument('--dtype', type=str, default='float64', help='data type')
    parser.add_argument('--iter_num', type=int, default=300, help='number of iterations')
    parser.add_argument('--topconf_num', type=int, default=3, help='number of top configurations')
    parser.add_argument('--reeval_num', type=int, default=4, help='number of re-evaluations of top configurations')
    parser.add_argument('--save_dir', type=str, default='bayes_save/', help='saving path')
    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    if args.dtype == 'float16':
        torch.set_default_dtype(torch.float16)
    elif args.dtype == 'float32':
        torch.set_default_dtype(torch.float32)
    elif args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)

    data = Data(bounds_x = args.bounds_x,
                bounds_t = args.bounds_t,
                nx_tr_initial = args.nx_tr_initial,
                nx_tr_interior = args.nx_tr_interior,
                nt_tr_interior = args.nt_tr_interior,
                nx_tr_boundary = args.nx_tr_boundary,
                nt_tr_boundary = args.nt_tr_boundary,
                nx_te = args.nx_te,
                nt_te = args.nt_te,
                lambda_initial = args.lambda_initial,
                lambda_boundary = args.lambda_boundary,
                coef = args.coef)

    data.data_to_gpu(device)

    best_layer_num, best_neuron_num, best_activation = bayes_main(args, data, device, seed)

    with open(args.save_dir + 'best_conf.txt', 'w') as file:
        file.write('best_layer_num: '+ str(best_layer_num) + '\n')
        file.write('best_neuron_num: '+ str(best_neuron_num) + '\n')
        file.write('best_activation: ' + best_activation[0].__name__ + '_' + str(best_activation[1]) )


if __name__ == '__main__':
    # set seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main(seed)

