
import torch
import numpy as np
from matplotlib import pyplot as plt


def uu(x):
    """true solution"""

    return x[:,0:1] * torch.cos(5.0 * np.pi * x[:,1:2]) + (x[:,0:1] * x[:,1:2])**3   # UNDERSTANDING AND MITIGATING GRADIENT FLOW PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS --- paper
    #return x[:,0:1] * torch.cos(4.0 * np.pi * x[:,1:2]) + (x[:,0:1] * x[:,1:2])**3   # generalization-1
    #return x[:, 0:1] * torch.cos(6.0 * np.pi * x[:, 1:2]) + 1.2*(x[:, 0:1] * x[:, 1:2])**3  #generalization-2


def rr(x):
    """ right hand side of governing equation """

    x.requires_grad = True

    u = uu(x)
    u_xt, = torch.autograd.grad(u, x, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(u))
    u_x0 = u_xt[:,0:1]
    u_t = u_xt[:,1:2]

    u_x0_xt, = torch.autograd.grad(u_x0, x, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones_like(u_x0))
    u_x0_x0 = u_x0_xt[:,0:1]
    u_t_xt, = torch.autograd.grad(u_t, x, create_graph=True, retain_graph=True,
                                  grad_outputs=torch.ones_like(u_t))
    u_t_t = u_t_xt[:,1:2]

    x.requires_grad = False

    return ( u_t_t - u_x0_x0 + u ** 3).detach()


class InteriorSet():
    def __init__(self, bounds_x, bounds_t, nx, nt):
        self.bounds_x = bounds_x
        self.bounds_t = bounds_t
        self.nx = nx
        self.nt = nt

        self.size = self.nx[0] * self.nt
        self.dim = self.bounds_x.shape[0]
        self.x = torch.zeros(self.size, self.dim+1)
        self.hx = (self.bounds_x[:, 1] - self.bounds_x[:, 0]) / self.nx
        self.ht = (self.bounds_t[1] - self.bounds_t[0]) / self.nt

        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nt):
                self.x[m, 0] = self.bounds_x[0, 0] + (i + 0.5) * self.hx[0]
                self.x[m, 1] = self.bounds_t[0] + (j + 0.5) * self.ht
                m = m + 1

        self.r = rr(self.x)

        # plt.plot(self.x[:,0], self.x[:,1],'.')
        # plt.axis('equal')
        # plt.show()


class InitialSet():
    def __init__(self, bounds_x, nx, lambda_initial):
        self.bounds_x = bounds_x
        self.nx = nx
        self.lambda_initial = lambda_initial
        self.size = (self.nx[0] + 1)
        self.dim = self.bounds_x.shape[0]
        self.x = torch.zeros(self.size,self.dim+1)
        self.hx = (self.bounds_x[:,1]-self.bounds_x[:,0])/self.nx
        m = 0
        for i in range(self.nx[0]+1):
            self.x[m,0] = self.bounds_x[0,0] + i*self.hx[0]
            m += 1

        self.x.requires_grad = True

        self.r = uu(self.x)
        r_xt, = torch.autograd.grad(self.r, self.x, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones_like(self.r))
        self.r = self.r.detach()
        self.r_t = r_xt[:, 1:2].detach()

        self.x.requires_grad = False

        # plt.plot(self.x[:,0], self.x[:,1],'.')
        # plt.axis('equal')
        # plt.show()


class BoundarySet():
    def __init__(self, bounds_x, bounds_t, nt, lambda_boundary):
        self.bounds_x = bounds_x
        self.bounds_t = bounds_t
        self.nt = nt
        self.lambda_boundary = lambda_boundary

        self.d_size = 2*(self.nt+1)
        self.dim = self.bounds_x.shape[0]
        self.d_x = torch.zeros(self.d_size,self.dim+1)
        self.ht = (self.bounds_t[1] - self.bounds_t[0]) / self.nt

        m = 0
        for i in range(self.nt+ 1):
            self.d_x[m,0] = self.bounds_x[0,0]
            self.d_x[m,1] = self.bounds_t[0] + i*self.ht
            m += 1
        for j in range(self.nt + 1):
            self.d_x[m,0] = self.bounds_x[0,1]
            self.d_x[m,1] = self.bounds_t[0] + j*self.ht
            m += 1

        self.d_r = uu(self.d_x)

        # plt.plot(self.d_x[:,0], self.d_x[:,1],'.')
        # plt.axis('equal')
        # plt.show()


class TestSet():
    def __init__(self, bounds_x, bounds_t, nx, nt):
        self.bounds_x = bounds_x
        self.bounds_t = bounds_t
        self.nx = nx
        self.nt = nt

        self.size = (self.nx[0]+1)*(self.nt+1)
        self.dim = self.bounds_x.shape[0]
        self.x = torch.zeros(self.size,self.dim+1)
        self.hx = (self.bounds_x[:,1]-self.bounds_x[:,0])/self.nx
        self.ht = (self.bounds_t[1]-self.bounds_t[0])/self.nt

        m = 0
        for i in range(self.nx[0]+1):
            for j in range(self.nt+1):
                self.x[m,0] = self.bounds_x[0,0] + i*self.hx[0]
                self.x[m,1] = self.bounds_t[0] + j*self.ht
                m += 1

        self.ua = uu(self.x)

        # plt.plot(self.x[:,0], self.x[:,1],'.')
        # plt.axis('equal')
        # plt.show()


#----------------------------------------------------------------------------------------------------
class Data():
    def __init__(self,
                 bounds_x = [[0.0, 1.0]],
                 bounds_t = [0.0, 1.0],
                 nx_tr_initial=[80],
                 nx_tr_interior=[60],
                 nt_tr_interior=60,
                 nt_tr_boundary=80,
                 nx_va_initial=[10],
                 nx_va_interior=[10],
                 nt_va_interior=10,
                 nt_va_boundary=10,
                 nx_te = [100],
                 nt_te = 100,
                 lambda_initial = 100.0,
                 lambda_boundary = 100.0):

        bounds_x = torch.tensor(bounds_x)
        bounds_t = torch.tensor(bounds_t)
        nx_tr_initial = torch.tensor(nx_tr_initial).int()
        nx_tr_interior = torch.tensor(nx_tr_interior).int()
        nt_tr_interior = torch.tensor(nt_tr_interior).int()
        nt_tr_boundary = torch.tensor(nt_tr_boundary).int()
        nx_va_initial = torch.tensor(nx_va_initial).int()
        nx_va_interior = torch.tensor(nx_va_interior).int()
        nt_va_interior = torch.tensor(nt_va_interior).int()
        nt_va_boundary = torch.tensor(nt_va_boundary).int()
        nx_te = torch.tensor(nx_te).int()
        nt_te = torch.tensor(nt_te).int()
        lambda_initial = torch.tensor(lambda_initial)
        lambda_boundary = torch.tensor(lambda_boundary)

        self.tr_set = {'initial': InitialSet(bounds_x, nx_tr_initial, lambda_initial),
                       'interior': InteriorSet(bounds_x, bounds_t, nx_tr_interior, nt_tr_interior),
                       'boundary': BoundarySet(bounds_x, bounds_t, nt_tr_boundary, lambda_boundary)}
        self.va_set = {'initial': InitialSet(bounds_x, nx_va_initial, lambda_initial),
                       'interior': InteriorSet(bounds_x, bounds_t, nx_va_interior, nt_va_interior),
                       'boundary': BoundarySet(bounds_x, bounds_t, nt_va_boundary, lambda_boundary)}

        self.test = TestSet(bounds_x, bounds_t, nx_te, nt_te)
        self.bounds_x = bounds_x
        self.bounds_t = bounds_t

    def data_to_gpu(self, device):

        self.tr_set['interior'].x = self.tr_set['interior'].x.to(device)
        self.tr_set['interior'].r = self.tr_set['interior'].r.to(device)
        self.tr_set['initial'].x = self.tr_set['initial'].x.to(device)
        self.tr_set['initial'].r = self.tr_set['initial'].r.to(device)
        self.tr_set['initial'].r_t = self.tr_set['initial'].r_t.to(device)
        self.tr_set['boundary'].d_x = self.tr_set['boundary'].d_x.to(device)
        self.tr_set['boundary'].d_r = self.tr_set['boundary'].d_r.to(device)

        self.va_set['interior'].x = self.va_set['interior'].x.to(device)
        self.va_set['interior'].r = self.va_set['interior'].r.to(device)
        self.va_set['initial'].x = self.va_set['initial'].x.to(device)
        self.va_set['initial'].r = self.va_set['initial'].r.to(device)
        self.va_set['initial'].r_t = self.va_set['initial'].r_t.to(device)
        self.va_set['boundary'].d_x = self.va_set['boundary'].d_x.to(device)
        self.va_set['boundary'].d_r = self.va_set['boundary'].d_r.to(device)

        self.test.x = self.test.x.to(device)
        self.test.ua = self.test.ua.to(device)

