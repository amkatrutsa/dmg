from . import gmg_base
from . import smoothers
from . import gallery
import autograd.numpy as np
import importlib

class LinearGMG(gmg_base.GMG):
    
    def __init__(self, A, max_levels=10,
                 presmoother=smoothers.damped_jacobi, presmooth_par={"damp": 2./3, "max_iter": 2},
                 postsmoother=None, postsmooth_par=None,
                 cycle="V"):
        super().__init__(A, cycle, presmoother, presmooth_par,
                         postsmoother, postsmooth_par, max_levels)
        dim = A.problem_dim
        if dim == 1:
            stencil_type = "3p"
        elif dim == 2:
            stencil_type = "9p"
        else:
            raise NotImplementedError("3D case is not implemented yet!")
        PR = importlib.import_module(".pr_{}d_{}".format(dim, stencil_type), "dmg")
        self.__PR_list = PR.generate_PR_list(self._max_levels, A.dim[0])
        self._max_levels = len(self.__PR_list) - 1
        self.__A_list = self.__generate_all_A()
        
    def __generate_all_A(self):
        A_list = [[] for i in range(self._max_levels + 1)]
        A_list[self._max_levels] = self._A
        for l in range(self._max_levels - 1, -1, -1):
            current_A = gallery.DiscretizationMatrix(self.__PR_list[l+1].shape)
            current_A.set_matrix(self.__PR_list[l+1].A_projection(A_list[l+1].get_matrix()))
            A_list[l] = current_A
        return A_list
    
    def __str__(self):
        output = "Dimension of matrix A for levels:\n"
        for i, A in enumerate(self.__A_list):
            output = output + "Level = {}, dimension = {}\n".format(i, A.dim)
        return output
    
    def _V_cycle_recursive(self, rhs, x0, level):
        U = x0
        if level == 0:
            A_mat = self.__A_list[level].to_full()
            U = np.linalg.solve(A_mat, rhs)
        else:
            U = self._presmoother(self.__A_list[level], rhs, U, self._presmooth_par)
            Au = self.__A_list[level].dot(U)
            res = self.__PR_list[level].R_matvec(rhs - Au)
            e = np.zeros_like(res)
            e = self._V_cycle_recursive(rhs=res, x0=e, level=level - 1)
            U = U + self.__PR_list[level].P_matvec(e)
            U = self._postsmoother(self.__A_list[level], rhs, U, self._postsmooth_par)
        return U
    
    def _V_cycle_loop(self, rhs, x0):
        U0 = x0.copy()
        num_levels = len(self.__PR_list) - 1
        res = [0 for i in range(num_levels + 1)]
        res[0] = [0]
        res[-1] = rhs
        presmoothed_u = [0 for i in range(num_levels + 1)]
        presmoothed_u[-1] = U0
        presmoothed_u[0] = [0]

        for i in range(num_levels, 0, -1):
            presmoothed_u[i] = self._presmoother(self.__A_list[i], res[i], presmoothed_u[i], self._presmooth_par)
            Au = self.__A_list[i].dot(presmoothed_u[i])
            res[i-1] = self.__PR_list[i].R_matvec(res[i] - Au)
            presmoothed_u[i - 1] = np.zeros_like(res[i-1])

        A_mat = self.__A_list[0].to_full()
        presmoothed_u[0] = np.linalg.solve(A_mat, res[0])
        if len(presmoothed_u[0].shape) == 1:
            presmoothed_u[0] = presmoothed_u[0].reshape(presmoothed_u[0].shape[0], 1)

        for i in range(1, num_levels+1):
            presmoothed_u[i] = presmoothed_u[i] + self.__PR_list[i].P_matvec(presmoothed_u[i-1])
            presmoothed_u[i] = self._postsmoother(self.__A_list[i], res[i], presmoothed_u[i], self._postsmooth_par)
        U = presmoothed_u[-1]
        return U
    def _V_cycle(self, rhs, x0):
        # return self._V_cycle_recursive(rhs, x0, self._max_levels)
        return self._V_cycle_loop(rhs, x0)