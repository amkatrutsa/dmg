import autograd.numpy as np
import scipy.sparse as spsp
from numba import jit
from autograd.extend import primitive, defvjp
import numpy as pure_np
import copy

@primitive
@jit(nopython=True)
def A_matvec_1D(A, X):
    n, m = X.shape
    Y = pure_np.zeros((n, m))
    for j in range(m):
        for i in range(n):
            # Out of bound values lead to the following rules to define limit in s index
            # s_min = -1 for all rows except the zeroth and 0 for the zero row
            # s_max = 2 for all rows except the last one and 1 for the last row
            s_min = max(-1, -i)
            s_max = min(2, n - i)
            for s in range(s_min, s_max):  # this loop simplifies differentiation
                Y[i, j] += A[i, s + 1] * X[i + s, j]
    return Y

@jit(nopython=True)
def A_matvec_1D_A_vjp_inner(g, A, X):
    Y = pure_np.zeros_like(A)
    m = X.shape[1]
    n = A.shape[0]
    for j in range(m):
        for i in range(n):
            s_min = max(-1, -i)
            s_max = min(2, n - i)
            for s in range(s_min, s_max):
                Y[i, s + 1] += g[i, j] * X[i + s, j]
    return Y

@primitive
def A_matvec_1D_A_vjp(g, ans, A, X):
    return A_matvec_1D_A_vjp_inner(g, A, X)

@jit(nopython=True)
def A_matvec_1D_X_vjp_inner(g, A, X):
    Y = pure_np.zeros_like(X)
    m = X.shape[1]
    n = A.shape[0]
    for j in range(m):
        for i in range(n):
            s_min = max(-1, -i)
            s_max = min(2, n - i)
            for s in range(s_min, s_max):
                Y[i + s, j] += g[i, j] * A[i, s + 1]
    return Y

@primitive
def A_matvec_1D_X_vjp(g, ans, A, X):
    return A_matvec_1D_X_vjp_inner(g, A, X)

defvjp(A_matvec_1D, lambda ans, A, X: lambda g: A_matvec_1D_A_vjp(g, ans, A, X), 
                       lambda ans, A, X: lambda g: A_matvec_1D_X_vjp(g, ans, A, X))

@primitive
@jit(nopython=True)
def A_matvec_2D(A, X):
    _, m = X.shape
    n = A.shape[0]
    Y = pure_np.zeros((n**2, m))
    for k in range(m):
        for i in range(n):
            for j in range(n):
                si_min = max(-1, -i)
                si_max = min(2, n - i)
                sj_min = max(-1, -j)
                sj_max = min(2, n - j)
                # This loops simplify differentiation
                for si in range(si_min, si_max):
                    for sj in range(sj_min, sj_max):
                        Y[n*i + j, k] += A[i, j, si + 1, sj + 1] * X[n*i + j + n*si + sj, k]
    return Y

@jit(nopython=True)
def A_matvec_2D_A_vjp_inner(g, A, X):
    Y = pure_np.zeros_like(A)
    m = X.shape[1]
    n = A.shape[0]
    for k in range(m):
        for i in range(n):
            for j in range(n):
                si_min = max(-1, -i)
                si_max = min(2, n - i)
                sj_min = max(-1, -j)
                sj_max = min(2, n - j)
                for si in range(si_min, si_max):
                    for sj in range(sj_min, sj_max):
                        Y[i, j, si + 1, sj + 1] += g[n*i + j, k] * X[n*i + j + n*si + sj, k]
    return Y

@primitive
def A_matvec_2D_A_vjp(g, ans, A, X):
    return A_matvec_2D_A_vjp_inner(g, A, X)

@jit(nopython=True)
def A_matvec_2D_X_vjp_inner(g, A, X):
    Y = pure_np.zeros_like(X)
    m = X.shape[1]
    n = A.shape[0]
    for k in range(m):
        for i in range(n):
            for j in range(n):
                si_min = max(-1, -i)
                si_max = min(2, n - i)
                sj_min = max(-1, -j)
                sj_max = min(2, n - j)
# this loops simplify differentiation
                for si in range(si_min, si_max):
                    for sj in range(sj_min, sj_max):
                        Y[n*i + j + n*si + sj, k] += A[i, j, si + 1, sj + 1] * g[n*i + j, k]
    return Y

@primitive
def A_matvec_2D_X_vjp(g, ans, A, X):
    return A_matvec_2D_X_vjp_inner(g, A, X)

defvjp(A_matvec_2D, lambda ans, A, X: lambda g: A_matvec_2D_A_vjp(g, ans, A, X),
                       lambda ans, A, X: lambda g: A_matvec_2D_X_vjp(g, ans, A, X))

@primitive
@jit(nopython=True)
def generate_full_A_1D(A):
    n = A.shape[0]
    A_mat = pure_np.zeros((n, n))
    for i in range(n):
        #0 <= i + s < n
        smin = max(-1, -i)
        smax = min(2, n-i)
        for s in range(smin, smax):
            A_mat[i, i+s] = A[i, s+1]
    return A_mat

@jit(nopython=True)
def generate_full_A_1D_vjp(g, ans, A):
    n = A.shape[0]
    res = pure_np.zeros((n, 3))
    for i in range(n):
        #0 <= i + s < n
        smin = max(-1, -i)
        smax = min(2, n-i)
        for s in range(smin, smax):
            res[i, s+1] = g[i, i+s] 
    return res

defvjp(generate_full_A_1D, lambda ans, A: lambda g: generate_full_A_1D_vjp(g, ans, A))

@primitive
@jit(nopython=True)
def generate_full_A_2D(A):
    n = A.shape[0]
    A_mat = pure_np.zeros((n**2, n**2))
    for i in range(n):
        for j in range(n):
            si_min = max(-1, -i)
            si_max = min(2, n-i)
            sj_min = max(-1, -j)
            sj_max = min(2, n-j)
            for si in range(si_min, si_max):
                for sj in range(sj_min, sj_max):
                    A_mat[n * i + j, n * i + j + si * n + sj] = A[i, j, 1 + si, 1 + sj]
    return A_mat

@jit(nopython=True)
def generate_full_A_2D_vjp(g, ans, A):
    n = A.shape[0]
    res = pure_np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            si_min = max(-1, -i)
            si_max = min(2, n-i)
            sj_min = max(-1, -j)
            sj_max = min(2, n-j)
            for si in range(si_min, si_max):
                for sj in range(sj_min, sj_max):
                    res[i, j, 1 + si, 1 + sj] = g[n * i + j, n * i + j + si * n + sj] 
    return res

defvjp(generate_full_A_2D, lambda ans, A: lambda g: generate_full_A_2D_vjp(g, ans, A))

    
    
class DiscretizationMatrix(object):
    def __init__(self, dim, stencil=None):
        self.__dim = dim
        if stencil is None:
            if len(dim) == 1:
                self.__A = np.zeros((dim[0], 3))
            elif len(dim) == 2:
                if dim[0] != dim[1]:
                    raise ValueError("Both dimensions must be the same")
                self.__A = np.zeros((dim[0], dim[1], 3, 3))
            else:
                raise NotImplementedError("3D case is not implemented yet")
        else:
            if len(stencil.shape) == 1:
                stencil = stencil[np.newaxis, :]
            elif len(dim) == 2 and len(stencil.shape) == 2 and stencil.shape[0] != 3 and stencil.shape[1] != 3:
                raise ValueError("For 2D problem the size of uniform stencil has to be 3 by 3")
            elif len(dim) == 3 and len(stencil.shape) == 3 and \
                 stencil.shape[0] != 3 and stencil.shape[1] != 3 and stencil.shape[2] != 3:
                raise ValueError("For 3D problem the size of stencil has to be 3 by 3 by 3")
            if len(dim) == 1 and stencil.shape[1] != 3:
                raise ValueError("For 1D problem the size of stencil has to be 3")
            if len(dim) == 1:
                n = dim[0]
                self.__A = np.zeros((n, 3))
                if stencil.shape[0] == 1:
                    for i in range(n):
                        self.__A[i, :] = stencil
                elif stencil.shape[0] == n:
                        self.__A = stencil.copy()
                self.__A[n - 1, 2] = None
                self.__A[0, 0] = None
            elif len(dim) == 2:
                if dim[0] != dim[1]:
                    raise ValueError("Both dimensions must be the same")
                n = dim[0]
                self.__A = np.zeros((n, n, 3, 3))
                if len(stencil.shape) == 2:
                    for i in range(n):
                        for j in range(n):
                            self.__A[i, j, :, :] = stencil
                            if i == 0:
                                self.__A[i, j, 0, :] = None
                            elif i == n-1:
                                self.__A[i, j, 2, :] = None
                            if j == 0:
                                self.__A[i, j, :, 0] = None
                            elif j == n - 1:
                                self.__A[i, j, :, 2] = None
                elif len(stencil.shape) == 4:
                    self.__A = stencil.copy()
                    self.__A[0, :, 0, :] = None
                    self.__A[n - 1, :, 2, :] = None
                    self.__A[:, 0, :, 0] = None
                    self.__A[:, n - 1, :, 2] = None
            elif len(dim) == 3:
                self.__A = np.zeros((dim[0], dim[1], dim[2], 3, 3, 3))
                # TODO
            self.__dim = dim
            self.__stencil = stencil
            
    def to_csr(self):
        if self.problem_dim == 1:
            n = self.__A.shape[0]
            A_mat = spsp.lil_matrix((n, n))
            for i in range(n):
                #0 <= i + s < n
                smin = max(-1, -i)
                smax = min(2, n-i)
                for s in range(smin, smax):
                    A_mat[i, i+s] = self.__A[i, s+1]
            return A_mat.tocsr()
        elif self.problem_dim == 2:
            n = self.__A.shape[0]
            A_mat = spsp.lil_matrix((n**2, n**2))
            for i in range(n):
                for j in range(n):
                    si_min = max(-1, -i)
                    si_max = min(2, n-i)
                    sj_min = max(-1, -j)
                    sj_max = min(2, n-j)
                    for si in range(si_min, si_max):
                        for sj in range(sj_min, sj_max):
                            A_mat[n * i + j, n * i + j + si * n + sj] = self.__A[i, j, 1 + si, 1 + sj]
            return A_mat.tocsr()
        elif self.problem_dim == 3:
            raise NotImplementedError("3D case is not implemented yet")
    
    def to_full(self):
        return self._to_full(self.__A)
    
    def _to_full(self, A):
        if self.problem_dim == 1:
            return generate_full_A_1D(A)
        elif self.problem_dim == 2:
            return generate_full_A_2D(A)

    def get_matrix(self):
        return self.__A.copy()
    
    def set_matrix(self, A):
        if len(A.shape) == 2:
            if A.shape[0] == self.__dim[0]:
                self.__A = A.copy()
            else:
                raise ValueError("Dimension of the setted matrix {} \
                                  has to be equal to the prepared dimension {}".format(A.shape[0], 
                                                                                       self.__dim[0]))
        elif len(A.shape) == 4:
            if A.shape[0] == A.shape[1] == self.__dim[0]:
                self.__A = A.copy()
            else:
                raise ValueError("Dimension of the setted matrix {} \
                                  has to be equal to the prepared dimension {}".format(A.shape, self.__dim[0]))
        else:
            raise NotImplementedError("3D case is not implemented yet")
    
    
    def dot(self, X):
        return self._dot(self.__A, X)
    
    def _dot(self, A, X):
        if self.problem_dim == 1:
            return A_matvec_1D(A, X)
        elif self.problem_dim == 2:
            return A_matvec_2D(A, X)
        else:
            raise NotImplementedError("3D case is not implemented yet")
    
    @property
    def shape(self):
        if len(self.__dim) == 1:
            return self.__dim[0]
        elif len(self.__dim) == 2:
            return self.__dim[0] * self.__dim[1]
        elif len(self.__dim) == 3:
            raise NotImplementedError("3D case is not implemented yet")
    
    @property
    def problem_dim(self):
        return len(self.__dim)
    
    @property
    def dim(self):
        return self.__dim
    
    def get_diagonal(self):
        if self.problem_dim == 1:
            return self.__A[:, 1].copy()
        elif self.problem_dim == 2:
            return self.__A[:, :, 1, 1].copy()
        else:
            raise NotImplementedError("3D case is not implemented yet")
    
    def __add__(self, other):
        A = copy.deepcopy(self)
        A.set_matrix(self.__A + other.get_matrix())
        return A
    
    def __mul__(self, other):
        A = copy.deepcopy(self)
        A.set_matrix(self.__A * other)
        return A
    
    __rmul__ = __mul__

def poisson(dim, ax=1, ay=1):
    '''
    -Delta u = f
    '''
    if len(dim) == 1:
        A = DiscretizationMatrix(dim, np.array([-1, 2, -1]))
        return A
    elif len(dim) == 2:
        A = DiscretizationMatrix(dim, np.array([[0, -ay, 0], [-ax, 2*ax + 2*ay, -ax], [0, -ay, 0]]))
        return A
    elif len(dim) == 3:
        raise NotImplementedError("3D case is not implemented yet")

def helmholtz(dim, k, ax=1, ay=1):
    '''
    -Delta u - k^2u = f
    '''
    if len(dim) == 1:
        helm_stencil = np.array([-1, 2 - k**2 / (dim[0] + 1)**2, -1])
    elif len(dim) == 2:
        helm_stencil = np.array([[0, -ay, 0], [-ax, 2*ax + 2*ay - k**2 / (dim[0]+1)**2, -ax], [0, -ay, 0]])
    elif len(dim) == 3:
        raise NotImplementedError("3D case is not implemented yet")
    A_helm = DiscretizationMatrix(dim, helm_stencil)
    return A_helm

def divkrad(dim, k):
    '''
    -div(k grad(u)) = f
    '''
    if len(dim) == 1:
        stencil = np.zeros((dim[0], 3))
        for i in range(dim[0]):
            stencil[i, :] = np.array([-k[i], k[i] + k[i+1], -k[i+1]])
    elif len(dim) == 2:
        stencil = np.zeros((dim[0], dim[1], 3, 3))
        if len(k.shape) == 2:
            if k.shape[0] - 1 != dim[0] or k.shape[1] - 1 != dim[1]:
                raise ValueError("Dimension of the K has to be equal dim + 1")
            for i in range(dim[0]):
                for j in range(dim[1]):
                    stencil[i, j, :, :] = np.array([[0, -k[i, j], 0],
                                                    [-k[i, j], 
                                                     (k[i, j] + k[i+1, j] + k[i, j+1] + k[i, j]), 
                                                     -k[i, j+1]],
                                                    [0, -k[i+1, j], 0]])
        elif k.shape[0] == 2:
            # 0, i, j == 'x'
            # 1, i, j == 'y'
            for i in range(dim[0]):
                for j in range(dim[1]):
                    stencil[i, j, :, :] = np.array([[0, -k[1, i, j], 0],
                                                    [-k[0, i, j], 
                                                     (k[0, i, j] + k[1, i+1, j] + k[0, i, j+1] + k[1, i, j]), 
                                                     -k[0, i, j+1]],
                                                    [0, -k[1, i+1, j], 0]])
        elif k.shape[2] == 4:
            raise NotImplementedError("Complete diffusion tensor is not implemented yet!")
    else:
        raise NotImplementedError("3D case is not implemented yet")
    A = DiscretizationMatrix(dim, stencil)
    return A
    

def convection_diffusion(dim, eps, ax=1., ay=1.):
    if len(dim) == 1:
        if ax > 0:
            conv_dif_stencil = np.array([-eps - ax / (dim[0] + 1), 2 * eps + ax / (dim[0] + 1), -eps])
        else:
            conv_dif_stencil = np.array([-eps, 2 * eps - ax / (dim[0] + 1), -eps + ax / (dim[0] + 1)])
    elif len(dim) == 2:
        h = 1. / (dim[0] + 1)
        a = ax
        b = ay
        conv_dif_stencil = np.array([[0, h * (b - np.abs(b)) / 2 - eps, 0],
                                     [-h * (a + np.abs(a)) / 2 - eps, 4 * eps + h * (np.abs(a) + np.abs(b)), h * (a - np.abs(a)) / 2 - eps],
                                     [0, -h * (b + np.abs(b)) / 2 - eps, 0]])
    elif len(dim) == 3:
        raise NotImplementedError("3D case is not implemented yet")
    A_dif = DiscretizationMatrix(dim, conv_dif_stencil)
    return A_dif
