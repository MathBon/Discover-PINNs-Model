
import torch


def loss_func_pinns(Net, dataset, equation):
    """ loss function for training """

    if equation == 'klein_gordon':

        initial = dataset['initial']
        interior = dataset['interior']
        boundary = dataset['boundary']

        initial.x.requires_grad = True
        interior.x.requires_grad = True

        interior_u = Net(interior.x)
        interior_ux, = torch.autograd.grad(interior_u, interior.x,
                                           create_graph=True, retain_graph=True,
                                           grad_outputs=torch.ones_like(interior_u),
                                           allow_unused=True)
        if interior_ux == None:
            interior_ux0x0 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
            interior_ux1x1 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)  # t is denoted by x1
        else:
            interior_ux0 = interior_ux[:, 0:1]
            interior_ux1 = interior_ux[:, 1:2]

            interior_ux0x, = torch.autograd.grad(interior_ux0, interior.x,
                                                 create_graph=True, retain_graph=True,
                                                 grad_outputs=torch.ones_like(interior_ux0),
                                                 allow_unused=True)
            if interior_ux0x == None:
                interior_ux0x = torch.zeros(interior.x.shape).to(interior.x.device)
            interior_ux0x0 = interior_ux0x[:, 0:1]

            interior_ux1x, = torch.autograd.grad(interior_ux1, interior.x,
                                                 create_graph=True, retain_graph=True,
                                                 grad_outputs=torch.ones_like(interior_ux1),
                                                 allow_unused=True)
            if interior_ux1x == None:
                interior_ux1x = torch.zeros(interior.x.shape).to(interior.x.device)
            interior_ux1x1 = interior_ux1x[:, 1:2]

        interior_res = interior_ux1x1 - interior_ux0x0 + interior_u ** 3 - interior.r

        # ---------------------------------------------------
        initial_u = Net(initial.x)
        initial_res_u = initial_u - initial.r

        initial_ux, = torch.autograd.grad(initial_u, initial.x,
                                          create_graph=True, retain_graph=True,
                                          grad_outputs=torch.ones_like(initial_u),
                                          allow_unused=True)
        if initial_ux == None:
            initial_ux = torch.zeros(initial.x.shape).to(initial.x.device)
        initial_ux1 = initial_ux[:, 1:2]
        initial_res_ux1 = initial_ux1 - initial.r_t

        boundary_res = Net(boundary.d_x) - boundary.d_r

        # ---------------------------------------------------
        loss = (interior_res ** 2).mean() + \
               initial.lambda_initial * (initial_res_u ** 2).mean() + initial.lambda_initial * (initial_res_ux1 ** 2).mean() + \
               boundary.lambda_boundary * (boundary_res ** 2).mean()

        initial.x.requires_grad = False
        interior.x.requires_grad = False


    elif equation == 'lame':

        interior = dataset['interior']
        boundary = dataset['boundary']

        interior.x.requires_grad = True
        boundary.n_x.requires_grad = True

        interior_u = Net(interior.x)

        interior_u0 = interior_u[:,0:1]
        interior_u1 = interior_u[:,1:2]

        interior_u0x, = torch.autograd.grad(interior_u0, interior.x,
                                            create_graph=True, retain_graph=True,
                                            grad_outputs=torch.ones_like(interior_u0),
                                            allow_unused = True)
        if interior_u0x is None:
            interior_u0x0x0 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
            interior_u0x0x1 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
            interior_u0x1x1 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
        else:
            interior_u0x0 = interior_u0x[:, 0:1]
            interior_u0x0x, = torch.autograd.grad(interior_u0x0, interior.x,
                                                  create_graph=True, retain_graph=True,
                                                  grad_outputs=torch.ones_like(interior_u0x0),
                                                  allow_unused = True)
            if interior_u0x0x is None:
                interior_u0x0x = torch.zeros(interior.x.shape).to(interior.x.device)
            interior_u0x0x0 = interior_u0x0x[:, 0:1]
            interior_u0x0x1 = interior_u0x0x[:, 1:2]

            interior_u0x1 = interior_u0x[:, 1:2]
            interior_u0x1x, = torch.autograd.grad(interior_u0x1, interior.x,
                                                  create_graph=True, retain_graph=True,
                                                  grad_outputs=torch.ones_like(interior_u0x1),
                                                  allow_unused = True)
            if interior_u0x1x is None:
                interior_u0x1x = torch.zeros(interior.x.shape).to(interior.x.device)
            interior_u0x1x1 = interior_u0x1x[:, 1:2]

        # ---------------------------------------------------

        interior_u1x, = torch.autograd.grad(interior_u1, interior.x,
                                            create_graph=True, retain_graph=True,
                                            grad_outputs=torch.ones_like(interior_u1),
                                            allow_unused = True)
        if interior_u1x is None:
            interior_u1x0x0 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
            interior_u1x0x1 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
            interior_u1x1x1 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
        else:
            interior_u1x0 = interior_u1x[:, 0:1]
            interior_u1x0x, = torch.autograd.grad(interior_u1x0, interior.x,
                                                  create_graph=True, retain_graph=True,
                                                  grad_outputs=torch.ones_like(interior_u1x0),
                                                  allow_unused = True)
            if interior_u1x0x is None:
                interior_u1x0x = torch.zeros(interior.x.shape).to(interior.x.device)
            interior_u1x0x0 = interior_u1x0x[:, 0:1]
            interior_u1x0x1 = interior_u1x0x[:, 1:2]

            interior_u1x1 = interior_u1x[:, 1:2]
            interior_u1x1x, = torch.autograd.grad(interior_u1x1, interior.x,
                                                  create_graph=True, retain_graph=True,
                                                  grad_outputs=torch.ones_like(interior_u1x1),
                                                  allow_unused = True)
            if interior_u1x1x is None:
                interior_u1x1x = torch.zeros(interior.x.shape).to(interior.x.device)
            interior_u1x1x1 = interior_u1x1x[:, 1:2]


        interior_res0 = (interior.elasticity_modulus / (1.0 - interior.poisson_ratio**2)) * (interior_u0x0x0 + 0.5*(1.0 - interior.poisson_ratio) * interior_u0x1x1 + 0.5*(1.0 + interior.poisson_ratio) * interior_u1x0x1)
        interior_res1 = (interior.elasticity_modulus / (1.0 - interior.poisson_ratio**2)) * (interior_u1x1x1 + 0.5*(1.0 - interior.poisson_ratio) * interior_u1x0x0 + 0.5*(1.0 + interior.poisson_ratio) * interior_u0x0x1)

        # ---------------------------------------------------
        boundary_d_u = Net(boundary.d_x)

        boundary_d_u0 = boundary_d_u[:,0:1]
        boundary_d_u1 = boundary_d_u[:,1:2]

        boundary_d_res0 = boundary_d_u0
        boundary_d_res1 = boundary_d_u1


        boundary_n_u = Net(boundary.n_x)
        boundary_n_u0 = boundary_n_u[:,0:1]
        boundary_n_u1 = boundary_n_u[:,1:2]

        boundary_n_u0x, = torch.autograd.grad(boundary_n_u0, boundary.n_x,
                                              create_graph=True, retain_graph=True,
                                              grad_outputs=torch.ones_like(boundary_n_u0),
                                              allow_unused=True)
        if boundary_n_u0x == None:
            boundary_n_u0x = torch.zeros(boundary.n_x.shape).to(boundary.n_x.device)
        boundary_n_u0x0 = boundary_n_u0x[:, 0:1]
        boundary_n_u0x1 = boundary_n_u0x[:, 1:2]

        boundary_n_u1x, = torch.autograd.grad(boundary_n_u1, boundary.n_x,
                                              create_graph=True, retain_graph=True,
                                              grad_outputs=torch.ones_like(boundary_n_u1),
                                              allow_unused=True)
        if boundary_n_u1x == None:
            boundary_n_u1x = torch.zeros(boundary.n_x.shape).to(boundary.n_x.device)
        boundary_n_u1x0 = boundary_n_u1x[:, 0:1]
        boundary_n_u1x1 = boundary_n_u1x[:, 1:2]


        boundary_n_0 = (boundary.elasticity_modulus / (1.0 - boundary.poisson_ratio ** 2))*(boundary.n_x[:, 0:1]/boundary.r_n_x *(boundary_n_u0x0 + boundary.poisson_ratio*boundary_n_u1x1) + boundary.n_x[:, 1:2]/boundary.r_n_x *(1.0 - boundary.poisson_ratio)*0.5 *(boundary_n_u0x1 + boundary_n_u1x0))
        boundary_n_1 = (boundary.elasticity_modulus / (1.0 - boundary.poisson_ratio ** 2))*(boundary.n_x[:, 1:2]/boundary.r_n_x *(boundary_n_u1x1 + boundary.poisson_ratio*boundary_n_u0x0) + boundary.n_x[:, 0:1]/boundary.r_n_x *(1.0 - boundary.poisson_ratio)*0.5 *(boundary_n_u1x0 + boundary_n_u0x1))

        boundary_n_res0 = boundary_n_0 - boundary.n_r0
        boundary_n_res1 = boundary_n_1 - boundary.n_r1

        loss = (interior_res0 ** 2).mean() + \
               (interior_res1 ** 2).mean() + \
               boundary.lambda_boundary * (boundary_d_res0 ** 2).mean() + \
               boundary.lambda_boundary * (boundary_d_res1 ** 2).mean() + \
               boundary.lambda_boundary * (boundary_n_res0 ** 2).mean() + \
               boundary.lambda_boundary * (boundary_n_res1 ** 2).mean()


        interior.x.requires_grad = False
        boundary.n_x.requires_grad = False


    elif equation == 'burgers':

        initial = dataset['initial']
        interior = dataset['interior']
        boundary = dataset['boundary']

        interior.x.requires_grad = True

        interior_u = Net(interior.x)
        interior_ux, = torch.autograd.grad(interior_u, interior.x,
                                           create_graph=True, retain_graph=True,
                                           grad_outputs=torch.ones_like(interior_u),
                                           allow_unused=True)
        if interior_ux == None:
            interior_ux0 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
            interior_ux1 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
            interior_ux2 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)  # t is denoted by x2
            interior_ux0x0 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
            interior_ux1x1 = torch.zeros(interior.x.shape[0], 1).to(interior.x.device)
        else:
            interior_ux0 = interior_ux[:, 0:1]
            interior_ux1 = interior_ux[:, 1:2]
            interior_ux2 = interior_ux[:, 2:3]

            interior_ux0x, = torch.autograd.grad(interior_ux0, interior.x,
                                                 create_graph=True, retain_graph=True,
                                                 grad_outputs=torch.ones_like(interior_ux0),
                                                 allow_unused=True)
            if interior_ux0x == None:
                interior_ux0x = torch.zeros(interior.x.shape).to(interior.x.device)
            interior_ux0x0 = interior_ux0x[:, 0:1]

            interior_ux1x, = torch.autograd.grad(interior_ux1, interior.x,
                                                 create_graph=True, retain_graph=True,
                                                 grad_outputs=torch.ones_like(interior_ux1),
                                                 allow_unused=True)
            if interior_ux1x == None:
                interior_ux1x = torch.zeros(interior.x.shape).to(interior.x.device)
            interior_ux1x1 = interior_ux1x[:, 1:2]

        interior_res = interior_ux2 + interior_u*(interior_ux0 + interior_ux1) - interior.coef*(interior_ux0x0 + interior_ux1x1)

        # ---------------------------------------------------
        initial_res = Net(initial.x) - initial.r

        boundary_res = Net(boundary.d_x) - boundary.d_r

        # ---------------------------------------------------
        loss = (interior_res ** 2).mean() + \
               initial.lambda_initial * (initial_res ** 2).mean() + \
               boundary.lambda_boundary * (boundary_res ** 2).mean()

        interior.x.requires_grad = False


    else:
        raise NameError('The equation is not defined')

    return loss
