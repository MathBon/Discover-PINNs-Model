
import torch
import numpy as np
from matplotlib import pyplot as plt


def uu(x, radius, elasticity_modulus, poisson_ratio, pressure, shear_force):
    """true solution"""

    r = (x[:, 0:1]**2 + x[:, 1:2]**2)**0.5

    G = 0.5*elasticity_modulus / (1.0 + poisson_ratio)

    ra = radius[0]
    rb = radius[1]

    A = (1.0 - poisson_ratio**2) * pressure * rb**2 / ( elasticity_modulus * (rb**2 * (1.0 + poisson_ratio) + ra**2 * (1.0 - poisson_ratio)))

    B = shear_force * rb**2 / (2.0 * G * ra**2)


    return A * ((ra/r)**2 - 1.0) * x[:, 0:1] + B * (1.0 - (ra/r)**2) * x[:, 1:2], \
           A * ((ra/r)**2 - 1.0) * x[:, 1:2] - B * (1.0 - (ra/r)**2) * x[:, 0:1]  # for pressure and shear force case


class InteriorSet():
    def __init__(self, radius, elasticity_modulus, poisson_ratio, nx):

        self.radius = radius
        self.elasticity_modulus = elasticity_modulus
        self.poisson_ratio = poisson_ratio
        self.dim = 2

        self.nx = nx
        x = torch.linspace(-self.radius[1], self.radius[1], self.nx[0])
        y = torch.linspace(-self.radius[1], self.radius[1], self.nx[1])
        mx, my = torch.meshgrid(x, y)
        ind = torch.logical_and((mx ** 2 + my ** 2) > self.radius[0] ** 2,
                                (mx ** 2 + my ** 2) < self.radius[1] ** 2)
        mxf = mx[ind]
        myf = my[ind]
        self.size = len(mxf)
        self.x = torch.zeros(self.size, self.dim)
        self.x[:,0] = mxf
        self.x[:,1] = myf

        # plt.plot(self.x[:,0], self.x[:,1],'.')
        # plt.axis('equal')
        # plt.show()


class BoundarySet():
    def __init__(self, radius, elasticity_modulus, poisson_ratio, pressure, shear_force, nx, lambda_boundary):

        self.radius = radius
        self.elasticity_modulus = elasticity_modulus
        self.poisson_ratio = poisson_ratio
        self.pressure = pressure
        self.shear_force = shear_force
        self.inner_nx = nx
        self.lambda_boundary = lambda_boundary
        th_inner_interval = 2 * np.pi / self.inner_nx
        th_outer_interval = th_inner_interval * (self.radius[0]/self.radius[1])

        self.d_x = torch.stack((torch.cos(torch.arange(0, 2 * np.pi, th_inner_interval)) * self.radius[0],
                                torch.sin(torch.arange(0, 2 * np.pi, th_inner_interval)) * self.radius[0]), 1)

        self.n_x = torch.stack((torch.cos(torch.arange(0, 2 * np.pi, th_outer_interval)) * self.radius[1],
                                torch.sin(torch.arange(0, 2 * np.pi, th_outer_interval)) * self.radius[1]), 1)

        self.r_n_x = (self.n_x[:, 0:1] ** 2 + self.n_x[:, 1:2] ** 2) ** 0.5


        # for pressure and shear force case
        self.n_r0 = -self.pressure * self.n_x[:, 0:1] / self.r_n_x + self.shear_force * self.n_x[:, 1:2] / self.r_n_x
        self.n_r1 = -self.pressure * self.n_x[:, 1:2] / self.r_n_x - self.shear_force * self.n_x[:, 0:1] / self.r_n_x

        # plt.plot(self.d_x[:,0], self.d_x[:,1], '.')
        # plt.plot(self.n_x[:,0], self.n_x[:,1], '.')
        # plt.axis('equal')
        # plt.show()


class TestSet():
    def __init__(self, radius, elasticity_modulus, poisson_ratio, pressure, shear_force, nx):

        self.radius = radius
        self.elasticity_modulus = elasticity_modulus
        self.poisson_ratio = poisson_ratio
        self.pressure = pressure
        self.shear_force = shear_force
        self.dim = 2
        self.nx = nx

        x = torch.linspace(-self.radius[1], self.radius[1], self.nx[0])
        y = torch.linspace(-self.radius[1], self.radius[1], self.nx[1])
        mx, my = torch.meshgrid(x, y)
        ind = torch.logical_and((mx ** 2 + my ** 2) >= self.radius[0] ** 2,
                                (mx ** 2 + my ** 2) <= self.radius[1] ** 2)
        mxf = mx[ind]
        myf = my[ind]
        self.size = len(mxf)
        self.x = torch.zeros(self.size, self.dim)
        self.x[:,0] = mxf
        self.x[:,1] = myf

        self.u0a, self.u1a = uu(self.x, self.radius, self.elasticity_modulus, self.poisson_ratio, pressure, shear_force)

        # plt.plot(self.x[:,0], self.x[:,1],'.')
        # plt.axis('equal')
        # plt.show()


#----------------------------------------------------------------------------------------------------
class Data():
    def __init__(self,
                 radius=[1.0, 2.0],
                 elasticity_modulus = 2.1,
                 poisson_ratio = 0.25,
                 pressure = 23.0,
                 shear_force = 3.0,
                 nx_tr_interior = [100, 100],
                 nx_tr_boundary = 200,
                 nx_va_interior = [10, 10],
                 nx_va_boundary = 10,
                 nx_te = [200, 200],
                 lambda_boundary = 10.0):

        self.radius = torch.tensor(radius)
        self.elasticity_modulus = torch.tensor(elasticity_modulus)
        self.poisson_ratio = torch.tensor(poisson_ratio)
        self.pressure = torch.tensor(pressure)
        self.shear_force = torch.tensor(shear_force)
        nx_tr_interior = torch.tensor(nx_tr_interior).int()
        nx_tr_boundary = torch.tensor(nx_tr_boundary).int()
        nx_va_interior = torch.tensor(nx_va_interior).int()
        nx_va_boundary = torch.tensor(nx_va_boundary).int()
        nx_te = torch.tensor(nx_te).int()
        lambda_boundary = torch.tensor(lambda_boundary)

        self.tr_set = {'interior': InteriorSet(self.radius, self.elasticity_modulus, self.poisson_ratio, nx_tr_interior),
                       'boundary': BoundarySet(self.radius, self.elasticity_modulus, self.poisson_ratio, self.pressure, self.shear_force, nx_tr_boundary, lambda_boundary)}
        self.va_set = {'interior': InteriorSet(self.radius, self.elasticity_modulus, self.poisson_ratio, nx_va_interior),
                       'boundary': BoundarySet(self.radius, self.elasticity_modulus, self.poisson_ratio, self.pressure, self.shear_force, nx_va_boundary, lambda_boundary)}
        self.test = TestSet(self.radius, self.elasticity_modulus, self.poisson_ratio, self.pressure, self.shear_force, nx_te)


    def data_to_gpu(self, device):

        self.tr_set['interior'].x = self.tr_set['interior'].x.to(device)
        self.tr_set['boundary'].d_x = self.tr_set['boundary'].d_x.to(device)
        self.tr_set['boundary'].n_x = self.tr_set['boundary'].n_x.to(device)
        self.tr_set['boundary'].r_n_x = self.tr_set['boundary'].r_n_x.to(device)
        self.tr_set['boundary'].n_r0 = self.tr_set['boundary'].n_r0.to(device)
        self.tr_set['boundary'].n_r1 = self.tr_set['boundary'].n_r1.to(device)

        self.va_set['interior'].x = self.va_set['interior'].x.to(device)
        self.va_set['boundary'].d_x = self.va_set['boundary'].d_x.to(device)
        self.va_set['boundary'].n_x = self.va_set['boundary'].n_x.to(device)
        self.va_set['boundary'].r_n_x = self.va_set['boundary'].r_n_x.to(device)
        self.va_set['boundary'].n_r0 = self.va_set['boundary'].n_r0.to(device)
        self.va_set['boundary'].n_r1 = self.va_set['boundary'].n_r1.to(device)

        self.test.x = self.test.x.to(device)
        self.test.u0a = self.test.u0a.to(device)
        self.test.u1a = self.test.u1a.to(device)

