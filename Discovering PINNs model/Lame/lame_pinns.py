
import sys
sys.path.append("..")
import argparse
import torch
import numpy as np
import time
from train import train_pinns
from train import relative_error
from lame_data import Data
import nn_models
import nn_config
import weight_init


def main():
    # Configurations
    parser = argparse.ArgumentParser(description='PINNs for solving Lame equation')
    parser.add_argument('--equation', type=str, default='lame', help='the equation to be solved')
    parser.add_argument('--radius', type=float, default=[1.0, 2.0], help='inner and outer radiuses of annulus')
    parser.add_argument('--elasticity_modulus', type=float, default=2.1, help='elasticity modulus')
    parser.add_argument('--poisson_ratio', type=float, default=0.25, help='Poisson’s ratio')
    parser.add_argument('--pressure', type=float, default=23.0, help='pressure acting on external boundary')
    parser.add_argument('--shear_force', type=float, default=3.0, help='shear force acting on external boundary')
    parser.add_argument('--nx_tr_interior', type=int, default=[100, 100], help='spatial size of training data in interior domain')
    parser.add_argument('--nx_tr_boundary', type=int, default=200, help='spatial size of training data on inner boundary')
    parser.add_argument('--nx_te', type=int, default=[200, 200], help='spatial size of testing data')
    parser.add_argument('--lambda_boundary', type=float, default=10.0, help='penalty coefficient for boundary loss term')
    parser.add_argument('--io_dim', type=int, default=[2, 2], help='dimensions of input and output')
    parser.add_argument('--OPTIM_TYPE', type=str, default='lbfgs', help='optimizer type')
    parser.add_argument('--EPOCHS_ALL', type=int, default=[5000], help='total number of training epochs')
    parser.add_argument('--EPOCHS_ONCE', type=int, default=100, help='number of training epochs before each evaluation')
    parser.add_argument('--LEARNING_RATE', type=float, default=1.0, help='learning rate')
    parser.add_argument('--TIME_LIMIT', type=float, default=60.0, help='threshold of training time between two evaluations')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether or not to use CUDA')
    parser.add_argument('--dtype', type=str, default='float64', help='data type')
    parser.add_argument('--tests_num', type=int, default=10, help='number of tests about one configuration')
    parser.add_argument('--save_dir', type=str, default='multi_save/', help='saving path')
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

    data = Data(radius = args.radius,
                elasticity_modulus = args.elasticity_modulus,
                poisson_ratio = args.poisson_ratio,
                pressure = args.pressure,
                shear_force = args.shear_force,
                nx_tr_interior = args.nx_tr_interior,
                nx_tr_boundary = args.nx_tr_boundary,
                nx_te = args.nx_te,
                lambda_boundary = args.lambda_boundary)

    data.data_to_gpu(device)

    errors_u0 = torch.zeros(len(nn_config.config), args.tests_num)
    errors_u1 = torch.zeros(len(nn_config.config), args.tests_num)
    time_cost = torch.zeros(len(nn_config.config), args.tests_num)
    train_history = torch.zeros(len(nn_config.config), args.tests_num, int(args.EPOCHS_ALL[0]/args.EPOCHS_ONCE) + 1, 4)

    for ic in range(len(nn_config.config)):
        print('################################################################')
        for it in range(args.tests_num):
            print('-------------------------------------------------------')
            Net = nn_models.NET(io_dim = args.io_dim,
                                layer_num=nn_config.config[ic][0],
                                residual_layer=nn_config.config[ic][1],
                                neuron_num=nn_config.config[ic][2],
                                activation=nn_config.config[ic][3]).to(device)

            # Net.apply(weight_init.xavier_normal)
            # Net.apply(weight_init.xavier_uniform)
            # Net.apply(weight_init.kaiming_normal)
            Net.apply(weight_init.kaiming_uniform)

            print('Number of parameters: %d' % (sum(param.numel() for param in Net.parameters())))
            print('Activation function: %s' % Net.activation.__name__)
            print('Activation function para_num: %s' % Net.actpara_num)

            start_time = time.time()
            # Train neural network
            net, train_history[ic, it, :, :] = train_pinns(Net, data, args.equation, args, args.EPOCHS_ALL[0])
            elapsed = time.time() - start_time
            print('training time: %.2f' % (elapsed))
            torch.save(net, args.save_dir + 'model/' + str(ic) + '-' + str(it))
            time_cost[ic, it] = elapsed

            # Error
            test_u = net(data.test.x).detach()
            test_u0 = test_u[:, 0:1]
            test_u1 = test_u[:, 1:2]
            errors_u0[ic, it] = relative_error(test_u0, data.test.u0a).detach()
            errors_u1[ic, it] = relative_error(test_u1, data.test.u1a).detach()

            print('error_u0: %.3e, error_u1: %.3e' %(errors_u0[ic, it], errors_u1[ic, it]))

    torch.save(data, args.save_dir + 'data.pt')
    torch.save(errors_u0, args.save_dir + 'errors_u0.pt')
    torch.save(errors_u1, args.save_dir + 'errors_u1.pt')
    torch.save(time_cost, args.save_dir + 'time_cost.pt')
    torch.save(train_history, args.save_dir + 'train_history.pt')


if __name__ == '__main__':
    # set seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main()

