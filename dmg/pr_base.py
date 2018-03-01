class PR(object):
    def __init__(self, R=None, P=None, coarse_dim=None, fine_dim=None, problem_dim=None):
        self._R = R
        self._P = P
        self._coarse_dim = coarse_dim
        self._fine_dim = fine_dim
        self._problem_dim = problem_dim
    
    @property
    def dim(self):
        return self._problem_dim
    
    @property
    def P_shape(self):
        if self.dim == 1:
            return (self._fine_dim, self._coarse_dim)
        elif self.dim == 2:
            return (self._fine_dim**2, self._coarse_dim**2)
        else:
            raise ValueError("Parameter dim is not 1 or 2, but {}".format(self.dim))
    
    @property        
    def R_shape(self):
        if self.dim == 1:
            return (self._coarse_dim, self._fine_dim)
        elif self.dim == 2:
            return (self._coarse_dim**2, self._fine_dim**2)
        else:
            raise ValueError("Parameter dim is not 1 or 2, but {}".format(self.dim))
    
    @property        
    def shape(self):
        if self.dim == 1:
            return (self._coarse_dim, )
        elif self.dim == 2:
            return (self._coarse_dim, self._coarse_dim)
        else:
            raise ValueError("Parameter dim is not 1 or 2, but {}".format(self.dim))
        
    def P_matvec(self, x):
        return self._P_matvec(self._P, x)
    def _P_matvec(self, _P, x):
        raise NotImplementedError("Matrix P by vector product is not implemented")
    
    def R_matvec(self, x):
        return self._R_matvec(self._R, x)
    def _R_matvec(self, _R, x):
        raise NotImplementedError("Matrix R by vector product is not implemented")
        
    def A_projection(self, A):
        return self._A_projection(self._R, A, self._P, self._coarse_dim)
    def _A_projection(self, _R, A, _P, coarse_dim):
        raise NotImplementedError("Galerkin projection of matrix A is not implemented")
