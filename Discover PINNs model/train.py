
import torch
import time
from loss import loss_func_pinns


def relative_error(u, ua):
    """ relative L2 error"""
    return (((u-ua)**2).sum() / ((ua**2).sum()+1e-16)) ** 0.5


def train_pinns(Net, data, equation, config, epochs):

    print('Train Neural Network')

    if config.OPTIM_TYPE=='adam':
        optim = torch.optim.Adam(Net.parameters(), lr=config.LEARNING_RATE)
    if config.OPTIM_TYPE=='lbfgs':
        optim = torch.optim.LBFGS(Net.parameters(), lr=config.LEARNING_RATE, max_iter=config.EPOCHS_ONCE,
                                  tolerance_grad=1e-16, tolerance_change=1e-16,
                                  line_search_fn='strong_wolfe')

    if equation in ['klein_gordon', 'burgers']:
        loss_history = torch.zeros(int(epochs/config.EPOCHS_ONCE)+1)
        valid_history = torch.zeros(int(epochs/config.EPOCHS_ONCE)+1)
        error_history = torch.zeros(int(epochs/config.EPOCHS_ONCE)+1)

        loss_history[0] = loss_func_pinns(Net, data.tr_set, equation).detach()
        valid_history[0] = loss_func_pinns(Net, data.va_set, equation).detach()
        test_u = Net(data.test.x).detach()
        error_history[0] = relative_error(test_u, data.test.ua).detach()
        print('epoch: %d, loss: %.3e, valid: %.3e, error: %.3e' %(0, loss_history[0], valid_history[0], error_history[0]))

    elif equation in ['lame']:
        loss_history = torch.zeros(int(epochs/config.EPOCHS_ONCE)+1)
        valid_history = torch.zeros(int(epochs/config.EPOCHS_ONCE)+1)
        error0_history = torch.zeros(int(epochs/config.EPOCHS_ONCE)+1)
        error1_history = torch.zeros(int(epochs/config.EPOCHS_ONCE) + 1)

        loss_history[0] = loss_func_pinns(Net, data.tr_set, equation).detach()
        valid_history[0] = loss_func_pinns(Net, data.va_set, equation).detach()
        test_u = Net(data.test.x).detach()
        test_u0 = test_u[:, 0:1]
        test_u1 = test_u[:, 1:2]
        error0_history[0] = relative_error(test_u0, data.test.u0a).detach()
        error1_history[0] = relative_error(test_u1, data.test.u1a).detach()
        print('epoch: %d, loss: %.3e, valid: %.3e, error0: %.3e, error1: %.3e' %(0, loss_history[0], valid_history[0], error0_history[0], error1_history[0]))

    else:
        raise NameError('The equation is not defined')

    optimal_loss = loss_history[0]
    optimal_state = Net
    
    """ Training """
    for it in range(int(epochs/config.EPOCHS_ONCE)):
        start_time = time.time()

        if config.OPTIM_TYPE == 'adam':
            for it_i in range(config.EPOCHS_ONCE):
                optim.zero_grad()
                loss = loss_func_pinns(Net, data.tr_set, equation)
                loss.backward()
                optim.step()

        if config.OPTIM_TYPE == 'lbfgs' or config.OPTIM_TYPE == 'bfgs':
            def closure():
                optim.zero_grad()
                loss = loss_func_pinns(Net, data.tr_set, equation)
                loss.backward()
                return loss
            optim.step(closure)

        if equation in ['klein_gordon', 'burgers']:
            loss_history[it+1] = loss_func_pinns(Net, data.tr_set, equation).detach()
            valid_history[it+1] = loss_func_pinns(Net, data.va_set, equation).detach()
            test_u = Net(data.test.x).detach()
            error_history[it+1] = relative_error(test_u, data.test.ua).detach()

            elapsed = time.time() - start_time
            print('epoch: %d, loss: %.3e, valid: %.3e, error: %.3e, time: %.2f'
                  %((it+1)*config.EPOCHS_ONCE, loss_history[it+1], valid_history[it+1], error_history[it+1], elapsed))

        elif equation in ['lame']:
            loss_history[it+1] = loss_func_pinns(Net, data.tr_set, equation).detach()
            valid_history[it+1] = loss_func_pinns(Net, data.va_set, equation).detach()
            test_u = Net(data.test.x).detach()
            test_u0 = test_u[:, 0:1]
            test_u1 = test_u[:, 1:2]
            error0_history[it+1] = relative_error(test_u0, data.test.u0a).detach()
            error1_history[it+1] = relative_error(test_u1, data.test.u1a).detach()
            elapsed = time.time() - start_time
            print('epoch: %d, loss: %.3e, valid: %.3e, error0: %.3e, error1: %.3e, time: %.2f'
                  %((it+1)*config.EPOCHS_ONCE, loss_history[it+1], valid_history[it+1], error0_history[it+1], error1_history[it+1], elapsed))
        else:
            raise NameError('The equation is not defined')

        if loss_history[it+1] < optimal_loss:
            optimal_loss = loss_history[it+1]
            optimal_state = Net

        if elapsed > config.TIME_LIMIT or loss_history[it+1].isnan():
            break

    if equation in ['klein_gordon', 'burgers']:
        train_history = torch.stack((loss_history, valid_history, error_history), 1)
    elif equation in ['lame']:
        train_history = torch.stack((loss_history, valid_history, error0_history, error1_history), 1)
    else:
        raise NameError('The equation is not defined')

    return optimal_state, train_history
