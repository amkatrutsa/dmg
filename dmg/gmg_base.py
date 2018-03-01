import autograd.numpy as np
import scipy.sparse.linalg as spsplin
import copy

class GMG(object):
    def __init__(self, A, cycle, presmoother, presmooth_par,
                 postsmoother=None, postsmooth_par=None, max_levels=10):
        self._presmoother = copy.deepcopy(presmoother)
        self._presmooth_par = copy.deepcopy(presmooth_par)
        if postsmoother:
            self._postsmoother = copy.deepcopy(postsmoother)
            self._postsmooth_par = copy.deepcopy(postsmooth_par)
        else:
            self._postsmoother = copy.deepcopy(presmoother)
            self._postsmooth_par = copy.deepcopy(presmooth_par)
        if cycle in ["V", "W"]:
            self._cycle = cycle
        else:
            raise NotImplementedError("Only V and W cycles are implemented!")
        self._max_levels = max_levels - 1
        self._convergence = []
        self._A = A
    
    def solve(self, rhs, x0=None, tol=1e-6, max_iter=100):
        self._convergence = []
        it = 0
        if x0 is None:
            x = np.zeros((self._A.shape, 1))
        else:
            if len(x0.shape) == 1:
                x = x0.copy()
                x = x[:, np.newaxis]
            elif len(x0.shape) == 2:
                x = x0.copy()
            else:
                raise Exception("Dimension of x0 has to be 2!")
        if len(rhs.shape) == 1:
            rhs = rhs[:, np.newaxis]
        current_tol = np.linalg.norm(self._A.dot(x) - rhs)
        if self._cycle == "V":
            while current_tol > tol and it < max_iter:
                x = self._V_cycle(rhs, x)
                current_tol = np.linalg.norm(self._A.dot(x) - rhs)
                self._convergence.append(current_tol)
                it += 1
        elif self._cycle == "W":
            while current_tol > tol and it < max_iter:
                x = self._W_cycle(rhs, x)
                current_tol = np.linalg.norm(self._A.dot(x) - rhs)
                self._convergence.append(current_tol)
                it += 1
        return x
    
    def get_gmg_convergence(self):
        return np.array(self._convergence)
    
    def __generate_all_A(self):
        raise NotImplementedError("Method for generation matrices for all grids is not implemented")
        
    def _V_cycle(self, rhs, x):
        raise NotImplementedError("V cycle is not implemented")
    
    def _W_cycle(self, rhs, x, max_levels=None):
        raise NotImplementedError("W cycle is not implemented")
    
    def _iteration_matvec(self, x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        rhs = np.zeros_like(x)
        if self._cycle == "V":
            Cx = self._V_cycle(rhs, x)
        elif self._cycle == "W":
            Cx = self._W_cycle(rhs, x)
        return Cx
    
    def compute_rho(self):
        n = self._A.shape
        C_x = spsplin.LinearOperator(shape=(n, n), matvec=lambda x: self._iteration_matvec(x))
        rho = np.abs(spsplin.eigs(C_x, k=1, return_eigenvectors=False, tol=1e-16)) 
        return rho[0]
    
    def _power_method(self, Ax, n, tol=1e-10, maxiter=100):
        x = np.random.randn(n)
        current_tol = np.linalg.norm(Ax.dot(x) - x)
        print("Initial error = {}".format(current_tol))
        it = 0
        while current_tol > tol and it < maxiter:
            x_next = Ax.dot(x)
            x_next = x_next / np.linalg.norm(x_next)
            lam = x_next.dot(Ax.dot(x_next))
            current_tol = np.linalg.norm(Ax.dot(x) - lam * x)
            it += 1
            x = x_next
        return lam, x