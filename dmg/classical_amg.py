import pyamg
from . import gmg_base

class ClassicalAMG(gmg_base.GMG):
    def __init__(self, A, max_levels=10,
                 presmoother="jacobi", presmooth_par={"omega": 2./3, "iterations": 2, "withrho": False},
                 postsmoother=None, postsmooth_par=None,
                 cycle="V"):
        super().__init__(A, cycle, presmoother, presmooth_par,
                         postsmoother, postsmooth_par, max_levels)
        self._amg_solver = pyamg.classical.classical.ruge_stuben_solver(A.to_csr(), max_levels=max_levels, 
                                                                        max_coarse=1)
        pyamg.relaxation.smoothing.change_smoothers(self._amg_solver,
                                                    presmoother=(self._presmoother, self._presmooth_par),
                                                    postsmoother=(self._postsmoother, self._postsmooth_par)
                                                    )
    def _V_cycle(self, rhs, x):
        return self._amg_solver.solve(b=rhs, x0=x, maxiter=1, cycle="V")
    
    def __str__(self):
        return self._amg_solver.__str__()