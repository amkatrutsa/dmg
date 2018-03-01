from . import optimizers
from . import gmg_base
from . import smoothers
from . import gallery
import copy
import autograd.numpy as np
import autograd
import numpy as pure_np
import time
import importlib

class DeepMG(gmg_base.GMG):
    def __init__(self, A, max_levels, PR_stencil_type=None, K=10, solver=optimizers.adam,
                 presmoother=smoothers.damped_jacobi_dmg, presmooth_par={"damp": 2./3, "max_iter": 2},
                 postsmoother=None, postsmooth_par=None,
                 cycle="V"):
        super().__init__(A, cycle, presmoother, presmooth_par,
                         postsmoother, postsmooth_par, max_levels)
        dim = A.problem_dim
        self._input_max_levels = self._max_levels
        self._K = K
        if PR_stencil_type not in ["m3p", "m2p", "3p", "2p",
                                   "m9p", "9p"]:
            raise ValueError("Unknown PR stencil type")
        elif PR_stencil_type not in ["m3p", "m2p", "3p", "2p"] and dim == 1:
            raise ValueError("Stencil type {} is unavailable for 1D problem".format(PR_stencil_type))
        elif PR_stencil_type not in ["m9p", "9p"] and dim == 2:
            raise ValueError("Stencil type {} is unavailable for 2D problem".format(PR_stencil_type))
        else:
            self.PR = importlib.import_module(".pr_{}d_{}".format(dim, PR_stencil_type), "dmg")
        
        self.__PR_list = self.PR.generate_PR_list(self._max_levels, A.dim[0])
        self._init_P_list = [PR._P for PR in self.__PR_list[1:]]
        self._init_R_list = [PR._R for PR in self.__PR_list[1:]]
        self._max_levels = len(self.__PR_list) - 1
        self._initialized_A = copy.deepcopy(A)
        self._A_list = self._generate_all_A()
        self._pure_A = A.get_matrix()
        self._presmooth_par["Amatvec"] = A._dot
        self._postsmooth_par["Amatvec"] = A._dot
        self._solver = solver
        self._opt_time = 0
        
    def _generate_all_A4grad(self, A, P_list, R_list):
        A_list = [[] for i in range(self._max_levels + 1)]
        A_list[self._max_levels] = A
        for l in range(self._max_levels - 1, -1, -1):
            A_list[l] = self.__PR_list[l+1]._A_projection(R_list[l+1], 
                                                          A_list[l+1],
                                                          P_list[l+1],
                                                          self.__PR_list[l+1].shape[0])
        return A_list
    
    def _generate_all_A(self):
        A_list = [[] for i in range(self._max_levels + 1)]
        A_list[self._max_levels] = self._A
        for l in range(self._max_levels - 1, -1, -1):
            current_A = gallery.DiscretizationMatrix(self.__PR_list[l+1].shape)
            A_projected = self.__PR_list[l+1].A_projection(A_list[l+1].get_matrix())
            current_A.set_matrix(A_projected)
            A_list[l] = current_A
        return A_list
    
    def __str__(self):
        output = "Dimension of matrix A for levels:\n"
        for i, A in enumerate(self._A_list):
            output = output + "Level = {}, dimension = {}\n".format(i, A.dim)
        return output
    
    def _generate_batch(self, batch_size):
        u = np.random.randint(0, 2, size=(self._A.shape, batch_size))
        u = 2 * u - 1
        self._U0 = u.copy()
        
    def get_optimizer_convergence(self):
        return self._optimizer_convergence
    
    def get_optimization_time(self):
        return self._opt_time

    def _V_cycle(self, rhs, x0):
        P_list = [[]] + [PR._P for PR in self.__PR_list[1:]]
        R_list = [[]] + [PR._R for PR in self.__PR_list[1:]]
        A_list = self._generate_all_A4grad(self._pure_A, P_list, R_list)
        return self._V_cycle4grad(rhs, x0, A_list, P_list, R_list)
    
    def _V_cycle4grad(self, rhs, x0, A_list, P_list, R_list, damp=None):
        U0 = x0
        if damp:
            self._presmooth_par["damp"] = damp
            self._postsmooth_par["damp"] = damp
        num_levels = len(P_list) - 1
        res = [0 for i in range(num_levels + 1)]
        res[0] = [0]
        res[-1] = rhs
        presmoothed_u = [0 for i in range(num_levels + 1)]
        presmoothed_u[-1] = U0
        presmoothed_u[0] = [0]
    
        for i in range(num_levels, 0, -1):
            presmoothed_u[i] = self._presmoother(A_list[i], res[i], presmoothed_u[i], self._presmooth_par)
            Au = self._A._dot(A_list[i], presmoothed_u[i])
            res[i-1] = self.__PR_list[i]._R_matvec(R_list[i], res[i] - Au)
            presmoothed_u[i-1] = np.zeros_like(res[i-1])
            

        A_mat = self._A._to_full(A_list[0])
        presmoothed_u[0] = np.linalg.solve(A_mat, res[0])
        
        if len(presmoothed_u[0].shape) == 1:
            presmoothed_u[0] = presmoothed_u[0].reshape(presmoothed_u[0].shape[0], 1)
            
        for i in range(1, num_levels+1):
            presmoothed_u[i] = presmoothed_u[i] + self.__PR_list[i]._P_matvec(P_list[i], presmoothed_u[i-1])
            presmoothed_u[i] = self._postsmoother(A_list[i], res[i], presmoothed_u[i], self._postsmooth_par)
        U = presmoothed_u[-1]
        return U
    
    def objective(self, PRd_tuple, it, A=None):
        rhs = np.zeros_like(self._U0)
        P_list = [[]] + PRd_tuple[0]
        R_list = [[]] + PRd_tuple[1]
        damp = PRd_tuple[2]
        if A is None:
            A = self._pure_A
        A_list = self._generate_all_A4grad(A, P_list, R_list)
        U = self._U0.copy()
        for i in range(self._K):
            U = self._V_cycle4grad(rhs, U, A_list, P_list, R_list, damp)
        return np.log(np.power(np.linalg.norm(U, "fro"), 2) / U.shape[1])
    
    def update_prd(self, prd):
        self._presmooth_par["damp"] = prd[2]
        self._postsmooth_par["damp"] = prd[2]
        fine_dims = [PR._fine_dim for PR in self.__PR_list[1:]]
        self.__PR_list = self.PR.generate_from_P_R_lists(prd[0], prd[1], fine_dims)
        
    def update_matrix(self, A):
        self._A = copy.deepcopy(A)
        self._pure_A = A.get_matrix()
      
    def update_init_matrix(self, A):
        self._initialized_A = copy.deepcopy(A)
    
    def reset_prd(self):
        self.__PR_list = self.PR.generate_PR_list(self._input_max_levels, self._A.dim[0])
        print(len(self.__PR_list))
        self._presmooth_par["damp"] = 2./3
        self._postsmooth_par["damp"] = 2./3
    
    def optimize(self, init_point=None, objective=None, num_iter=100, step_size=1e-4, batch_size=10,
                 callback=None):
        self._optimizer_convergence = []
        if objective is None:
            objective = self.objective
        if init_point is None:
            P_list = [PR._P for PR in self.__PR_list[1:]]
            R_list = [PR._R for PR in self.__PR_list[1:]]
            damp = self._presmooth_par["damp"]
            init_point = (P_list, R_list, damp)
        self._generate_batch(batch_size)
        grad = autograd.grad(objective)
        if callback is None:
            callback = self._callback
        start = time.time()
        optimal_par = self._solver(grad, init_point, num_iters=num_iter, step_size=step_size, 
                                   callback=callback)
        self._opt_time = time.time() - start
        return optimal_par
    
    def _callback(self, par, it, g, num_iters):
        self._optimizer_convergence.append(par)
        if (it + 1) % 100 == 0:
            print("Iteration {}/{}".format(it + 1, num_iters))
            test_batch_size = 20
            self._generate_batch(test_batch_size)
            upper_bound = self.objective(par, it)
            print("Upper bound estimation = {}".format(np.power(np.exp(upper_bound), 1./(2 * self._K))))
    
    def _homotopy_callback(self, par, it, g, num_iters):
        self._optimizer_convergence.append(par)
    
    def homotopy_optimize(self, A_init, homotopy_step_size, acceptance_limit, accept_arg="rho", init_point=None, 
                          batch_size=10, adam_step_size=1e-4, num_iter=100, homotopy_arg="matrix", 
                          log_filename=None):
        self._homotopy_convergence = {}
        if init_point is None:
            P_list = [PR._P for PR in self.__PR_list[1:]]
            R_list = [PR._R for PR in self.__PR_list[1:]]
            damp = self._presmooth_par["damp"]
            init_point = (P_list, R_list, damp)
        self.update_prd(init_point)
        init_rho = self.compute_rho()
        
        if log_filename is not None:
            f = open(log_filename, "a")
            print("Max eigenvalue for iteration matrix in init param = {}".format(init_rho), end="\n", file=f)
            f.close()
        else:
            print("Max eigenvalue for iteration matrix in init param = {}".format(init_rho))
        
        print("Try direct optimization...")
        hom_obj = lambda prd, it: self.objective(prd, it)
        init_opt_PRd = self.optimize(init_point, objective=hom_obj, num_iter=500, step_size=adam_step_size,
                                       batch_size=batch_size, callback=self._homotopy_callback)
        if accept_arg == "rho": 
            self.update_prd(init_opt_PRd)
            current_rho_upper = self.compute_rho()
        elif accept_arg == "objective":
            current_obj = hom_obj(init_opt_PRd, 0)
            current_rho_upper = np.power(np.exp(current_obj), 1./(2 * self._K))
        elif accept_arg == "obj_ratio":
            current_obj = hom_obj(init_opt_PRd, 0)
            current_rho_upper = np.power(np.exp(current_obj), 1./(2 * self._K))
        
        print("Upper bound after direct optimization = {}".format(current_rho_upper))
        if current_rho_upper < 0.5:
            print("Direct optimization is successful!")
            return init_opt_PRd
        else:
            print("Direct optimization is failed!")
        
        active_alpha = 0
        alpha = 0
        active_alpha_list = [active_alpha]
        alpha_idx = 0
        
        if log_filename is not None:
            f = open(log_filename, "a")
            print("Improve init C", end="\n", file=f)
            f.close()
        else:
            print("Improve init C")
        
        if homotopy_arg == "matrix":
            A_current = alpha * self._initialized_A + (1 - alpha) * A_init
            self.update_matrix(A_current)
            hom_obj = lambda prd, it: self.objective(prd, it)
        elif homotopy_arg == "objective":
            hom_obj = lambda prd, it: alpha * self.objective(prd, it, self._initialized_A.get_matrix()) + \
                                      (1 - alpha) * self.objective(prd, it, A_init.get_matrix())
                
        current_opt_PRd = self.optimize(init_point, objective=hom_obj, num_iter=num_iter, step_size=adam_step_size,
                                       batch_size=batch_size, callback=self._homotopy_callback)
        
        if accept_arg == "rho": 
            self.update_prd(current_opt_PRd)
            current_rho_upper = self.compute_rho()
        elif accept_arg == "objective":
            current_obj = hom_obj(current_opt_PRd, 0)
            current_rho_upper = np.power(np.exp(current_obj), 1./(2 * self._K))
        elif accept_arg == "obj_ratio":
            current_obj = hom_obj(current_opt_PRd, 0)
            current_rho_upper = np.power(np.exp(current_obj), 1./(2 * self._K))
            
        print("Upper bound after initial optimization = {}".format(current_rho_upper))
        self._homotopy_convergence[alpha_idx] = self.get_optimizer_convergence()
        alpha_idx += 1
        
        if log_filename is not None:
            f = open(log_filename, "a")
            print("Start homotopy...", end="\n", file=f)
            f.close()
        else:
            print("Start homotopy...")
        
        prev_rho_upper = current_rho_upper
        while active_alpha < 1.0:
            alpha = min(active_alpha + homotopy_step_size, 1.0)
            
            if log_filename is not None:
                f = open(log_filename, "a")
                print("Current alpha = {}".format(alpha), end="\n", file=f)
                f.close()
            else:
                print("Current alpha = {}".format(alpha))
            
            if homotopy_arg == "matrix":
                A_current = alpha * self._initialized_A + (1 - alpha) * A_init
                self.update_matrix(A_current)
                hom_obj = lambda prd, it: self.objective(prd, it)
            elif homotopy_arg == "objective":
                hom_obj = lambda prd, it: alpha * self.objective(prd, it, self._initialized_A.get_matrix()) + \
                                          (1 - alpha) * self.objective(prd, it, A_init.get_matrix())

            step_power = 1
            current_alpha_prd = copy.deepcopy(current_opt_PRd)
            while True:
                current_opt_PRd = self.optimize(current_alpha_prd, objective=hom_obj, num_iter=num_iter, 
                                                step_size=adam_step_size, batch_size=batch_size, 
                                                callback=self._homotopy_callback)
                
                if accept_arg == "rho":
                    self.update_prd(current_opt_PRd)
                    test_accept = self.compute_rho()
                elif accept_arg == "objective":
                    current_obj = hom_obj(current_opt_PRd, 0)
                    test_accept = np.power(np.exp(current_obj), 1./(2 * self._K))
                elif accept_arg == "obj_ratio":
                    current_obj = hom_obj(current_opt_PRd, 0)
                    current_rho_upper = np.power(np.exp(current_obj), 1./(2 * self._K))
                    test_accept =  current_rho_upper / prev_rho_upper
                
                if log_filename is not None:
                    f = open(log_filename, "a")
                    print("Current eigenvalue upper bound for iteration matrix = {}".format(current_rho_upper), end="\n", file=f)
                    f.close()
                else:
                    print("Current eigenvalue upper bound for iteration matrix = {}".format(current_rho_upper))
                    print("Test acceptance = {}".format(test_accept))
                
                if test_accept < acceptance_limit:
                    active_alpha = alpha
                    active_alpha_list.append(active_alpha)
                    self._homotopy_convergence[alpha_idx] = self.get_optimizer_convergence()
                    alpha_idx += 1
                    prev_rho_upper = current_rho_upper
                    if log_filename is not None:
                        f = open(log_filename, "a")
                        print("New active alpha = {}".format(active_alpha), end="\n", file=f)
                        f.close()
                    else:
                        print("New active alpha = {}".format(active_alpha))
                    break
                while active_alpha + homotopy_step_size * 0.5**step_power > 1:
                    step_power += 1
                alpha = active_alpha + homotopy_step_size * 0.5**step_power
                
                if homotopy_arg == "matrix":
                    A_current = alpha * self._initialized_A + (1 - alpha) * A_init
                    self.update_matrix(A_current)
                    hom_obj = lambda prd, it: self.objective(prd, it)
                elif homotopy_arg == "objective":
                    hom_obj = lambda prd, it: alpha * self.objective(prd, it, self._initialized_A.get_matrix()) + \
                                          (1 - alpha) * self.objective(prd, it, A_init.get_matrix())
                
                if log_filename is not None:
                    f = open(log_filename, "a")
                    print("Alpha = {}".format(alpha), end="\n", file=f)
                    f.close()
                else:
                    print("Alpha = {}".format(alpha))
                
                step_power += 1
        self._active_alpha_list = copy.copy(active_alpha_list)
        return current_opt_PRd
    
    def get_init_p_list(self):
        return self._init_P_list
    
    def get_init_r_list(self):
        return self._init_R_list