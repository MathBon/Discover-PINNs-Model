
import torch
from matplotlib import pyplot as plt


def uu(x, coef):
    """true solution"""

    return 1.0/ (1.0 + torch.exp((0.5/coef)*(x[:,0:1]+x[:,1:2]-x[:,2:3])))



class InteriorSet():
    def __init__(self, bounds_x, bounds_t, nx, nt, coef):
        self.bounds_x = bounds_x
        self.bounds_t = bounds_t
        self.nx = nx
        self.nt = nt
        self.coef = coef

        # J shape
        self.size = (3* self.nx[0] * self.nx[1]) * self.nt
        self.dim = self.bounds_x.shape[0]
        self.x = torch.zeros(self.size, self.dim+1)
        self.hx = (self.bounds_x[:, 1] - self.bounds_x[:, 0]) / (2*self.nx)
        self.ht = (self.bounds_t[1] - self.bounds_t[0]) / self.nt

        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nt):
                    self.x[m, 0] = self.bounds_x[0, 0] + (i + 0.5) * self.hx[0]
                    self.x[m, 1] = self.bounds_x[1, 0] + (j + 0.5) * self.hx[1]
                    self.x[m, 2] = self.bounds_t[0] + (k + 0.5) * self.ht
                    m = m + 1

        for i in range(self.nx[0], 2*self.nx[0]):
            for j in range(2*self.nx[1]):
                for k in range(self.nt):
                    self.x[m, 0] = self.bounds_x[0, 0] + (i + 0.5) * self.hx[0]
                    self.x[m, 1] = self.bounds_x[1, 0] + (j + 0.5) * self.hx[1]
                    self.x[m, 2] = self.bounds_t[0] + (k + 0.5) * self.ht
                    m = m + 1


        # ax = plt.subplot(projection='3d')
        # ax.set_title('3d_image')
        # ax.scatter(self.x[:,0], self.x[:,1], self.x[:,2], c='b')
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('t')
        # plt.show()


class InitialSet():
    def __init__(self, bounds_x, nx, lambda_initial, coef):
        self.bounds_x = bounds_x
        self.nx = nx
        self.lambda_initial = lambda_initial
        self.coef = coef


        # J shape
        self.size = (self.nx[0] + 1)* (self.nx[1] + 1) + self.nx[0] * (2*self.nx[1]+1)
        self.dim = self.bounds_x.shape[0]
        self.x = torch.zeros(self.size,self.dim+1)
        self.hx = (self.bounds_x[:,1]-self.bounds_x[:,0])/ (2*self.nx)
        m = 0
        for i in range(self.nx[0]+1):
            for j in range(self.nx[1]+1):
                self.x[m,0] = self.bounds_x[0,0] + i*self.hx[0]
                self.x[m,1] = self.bounds_x[1,0] + j*self.hx[1]
                m += 1

        for i in range(self.nx[0]+1, 2*self.nx[0]+1):
            for j in range(2*self.nx[1]+1):
                self.x[m,0] = self.bounds_x[0,0] + i*self.hx[0]
                self.x[m,1] = self.bounds_x[1,0] + j*self.hx[1]
                m += 1


        self.r = uu(self.x, self.coef)

        # ax = plt.subplot(projection='3d')
        # ax.set_title('3d_image')
        # ax.scatter(self.x[:,0], self.x[:,1], self.x[:,2], c='b')
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('t')
        # plt.show()


class BoundarySet():
    def __init__(self, bounds_x, bounds_t, nx, nt, lambda_boundary, coef):
        self.bounds_x = bounds_x
        self.bounds_t = bounds_t
        self.nx = nx
        self.nt = nt
        self.lambda_boundary = lambda_boundary
        self.coef = coef


        # J shape
        self.d_size = 4*(self.nx[0] + self.nx[1])*(self.nt+1)
        self.dim = self.bounds_x.shape[0]
        self.d_x = torch.zeros(self.d_size,self.dim+1)
        self.hx = (self.bounds_x[:, 1] - self.bounds_x[:, 0]) / (2*self.nx)
        self.ht = (self.bounds_t[1] - self.bounds_t[0]) / self.nt

        m = 0
        for i in range(2*self.nx[0]):
            for k in range(self.nt + 1):
                self.d_x[m,0] = self.bounds_x[0,0] + i*self.hx[0]
                self.d_x[m,1] = self.bounds_x[1,0]
                self.d_x[m,2] = self.bounds_t[0] + k*self.ht
                m += 1
        for j in range(2*self.nx[1]):
            for k in range(self.nt + 1):
                self.d_x[m,0] = self.bounds_x[0,1]
                self.d_x[m,1] = self.bounds_x[1,0] + j*self.hx[1]
                self.d_x[m,2] = self.bounds_t[0] + k*self.ht
                m += 1
        for i in range(self.nx[0]):
            for k in range(self.nt + 1):
                self.d_x[m,0] = self.bounds_x[0,1] - i*self.hx[0]
                self.d_x[m,1] = self.bounds_x[1,1]
                self.d_x[m,2] = self.bounds_t[0] + k*self.ht
                m += 1
        for j in range(self.nx[1]):
            for k in range(self.nt + 1):
                self.d_x[m,0] = (self.bounds_x[0,0] + self.bounds_x[0,1])/2
                self.d_x[m,1] = self.bounds_x[1,1] - j*self.hx[1]
                self.d_x[m,2] = self.bounds_t[0] + k*self.ht
                m += 1
        for i in range(self.nx[0]):
            for k in range(self.nt + 1):
                self.d_x[m,0] = (self.bounds_x[0,0]+self.bounds_x[0,1])/2 - i*self.hx[0]
                self.d_x[m,1] = (self.bounds_x[1,0] + self.bounds_x[1,1])/2
                self.d_x[m,2] = self.bounds_t[0] + k*self.ht
                m += 1
        for j in range(self.nx[1]):
            for k in range(self.nt + 1):
                self.d_x[m,0] = self.bounds_x[0,0]
                self.d_x[m,1] = (self.bounds_x[1,0] + self.bounds_x[1,1])/2 - j*self.hx[1]
                self.d_x[m,2] = self.bounds_t[0] + k*self.ht
                m += 1


        self.d_r = uu(self.d_x, self.coef)

        # ax = plt.subplot(projection='3d')
        # ax.set_title('3d_image')
        # ax.scatter(self.d_x[:,0], self.d_x[:,1], self.d_x[:,2], c='b')
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('t')
        # plt.show()


class TestSet():
    def __init__(self, bounds_x, bounds_t, nx, nt, coef):
        self.bounds_x = bounds_x
        self.bounds_t = bounds_t
        self.nx = nx
        self.nt = nt
        self.coef = coef


        # J shape
        self.size = ((self.nx[0])*(self.nx[1]+1) + (self.nx[0]+1)*(2*self.nx[1]+1))*(self.nt+1)
        self.dim = self.bounds_x.shape[0]
        self.x = torch.zeros(self.size,self.dim+1)
        self.hx = (self.bounds_x[:,1]-self.bounds_x[:,0])/(2*self.nx)
        self.ht = (self.bounds_t[1]-self.bounds_t[0])/self.nt

        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]+1):
                for k in range(self.nt+1):
                    self.x[m,0] = self.bounds_x[0,0] + i*self.hx[0]
                    self.x[m,1] = self.bounds_x[1,0] + j*self.hx[1]
                    self.x[m,2] = self.bounds_t[0] + k*self.ht
                    m += 1

        for i in range(self.nx[0], 2*self.nx[0]+1):
            for j in range(2*self.nx[1]+1):
                for k in range(self.nt+1):
                    self.x[m,0] = self.bounds_x[0,0] + i*self.hx[0]
                    self.x[m,1] = self.bounds_x[1,0] + j*self.hx[1]
                    self.x[m,2] = self.bounds_t[0] + k*self.ht
                    m += 1


        self.ua = uu(self.x, self.coef)

        # ax = plt.subplot(projection='3d')
        # ax.set_title('3d_image')
        # ax.scatter(self.x[:,0], self.x[:,1], self.x[:,2], c='b')
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('t')
        # plt.show()


#----------------------------------------------------------------------------------------------------
class Data():
    def __init__(self,
                 bounds_x = [[0.0, 1.0], [0.0, 1.0]],
                 bounds_t = [0.0, 2.0],
                 nx_tr_initial=[15, 15],
                 nx_tr_interior=[10, 10],
                 nt_tr_interior=20,
                 nx_tr_boundary= [15,15],
                 nt_tr_boundary=30,
                 nx_va_initial=[5,5],
                 nx_va_interior=[5,5],
                 nt_va_interior=5,
                 nx_va_boundary=[5, 5],
                 nt_va_boundary=5,
                 nx_te = [25,25],
                 nt_te = 50,
                 lambda_initial = 100.0,
                 lambda_boundary = 100.0,
                 coef = 0.1):

        bounds_x = torch.tensor(bounds_x)
        bounds_t = torch.tensor(bounds_t)
        nx_tr_initial = torch.tensor(nx_tr_initial).int()
        nx_tr_interior = torch.tensor(nx_tr_interior).int()
        nt_tr_interior = torch.tensor(nt_tr_interior).int()
        nx_tr_boundary = torch.tensor(nx_tr_boundary).int()
        nt_tr_boundary = torch.tensor(nt_tr_boundary).int()
        nx_va_initial = torch.tensor(nx_va_initial).int()
        nx_va_interior = torch.tensor(nx_va_interior).int()
        nt_va_interior = torch.tensor(nt_va_interior).int()
        nx_va_boundary = torch.tensor(nx_va_boundary).int()
        nt_va_boundary = torch.tensor(nt_va_boundary).int()
        nx_te = torch.tensor(nx_te).int()
        nt_te = torch.tensor(nt_te).int()
        lambda_initial = torch.tensor(lambda_initial)
        lambda_boundary = torch.tensor(lambda_boundary)
        coef = torch.tensor(coef)

        self.tr_set = {'initial': InitialSet(bounds_x, nx_tr_initial, lambda_initial, coef),
                       'interior': InteriorSet(bounds_x, bounds_t, nx_tr_interior, nt_tr_interior, coef),
                       'boundary': BoundarySet(bounds_x, bounds_t, nx_tr_boundary, nt_tr_boundary, lambda_boundary, coef)}
        self.va_set = {'initial': InitialSet(bounds_x, nx_va_initial, lambda_initial, coef),
                       'interior': InteriorSet(bounds_x, bounds_t, nx_va_interior, nt_va_interior, coef),
                       'boundary': BoundarySet(bounds_x, bounds_t, nx_va_boundary, nt_va_boundary, lambda_boundary, coef)}

        self.test = TestSet(bounds_x, bounds_t, nx_te, nt_te, coef)
        self.bounds_x = bounds_x
        self.bounds_t = bounds_t

    def data_to_gpu(self, device):

        self.tr_set['interior'].x = self.tr_set['interior'].x.to(device)
        self.tr_set['initial'].x = self.tr_set['initial'].x.to(device)
        self.tr_set['initial'].r = self.tr_set['initial'].r.to(device)
        self.tr_set['boundary'].d_x = self.tr_set['boundary'].d_x.to(device)
        self.tr_set['boundary'].d_r = self.tr_set['boundary'].d_r.to(device)

        self.va_set['interior'].x = self.va_set['interior'].x.to(device)
        self.va_set['initial'].x = self.va_set['initial'].x.to(device)
        self.va_set['initial'].r = self.va_set['initial'].r.to(device)
        self.va_set['boundary'].d_x = self.va_set['boundary'].d_x.to(device)
        self.va_set['boundary'].d_r = self.va_set['boundary'].d_r.to(device)

        self.test.x = self.test.x.to(device)
        self.test.ua = self.test.ua.to(device)
