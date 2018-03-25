import dmg.gallery as gallery
import dmg.dgmg as dgmg
import dmg.gmg_linear as gmg_linear
import dmg.classical_amg as classical_amg
import numpy as np

## 1D Poisson equation

print("1D Poisson equation")
print("Compute spectral radius for linear P and R operators...")
num_levels = 3
print("Number of levels = {}".format(num_levels))
ns = [2**i - 1 for i in range(3, 8)]
rhos_linear = []
print("Test dims: {}".format(ns))
for n in ns:
    print("Current dim = {}".format(n))
    A = gallery.poisson((n, ))
    linear_gmg = gmg_linear.LinearGMG(A, max_levels=num_levels)
    current_rho = linear_gmg.compute_rho()
    print("Spectral radius = {}".format(current_rho))
    rhos_linear.append(current_rho)

print("Compute spectral radius for linear P and R operators...Done")

print("Make spectral radius smaller with optimization of P and R...")
rhos_opt = []
K = 20
batch_size = 10
num_iter = 1000
step_size = 5e-5
PR_stencil_type = "3p"
init_point = None
convergence = {}
opt_par_n = {}
conv_time = {}
for i, n in enumerate(ns):
    print("Current dim = {}".format(n))
    print("Optimization...")
    A = gallery.poisson((n, ))
    deep_gmm = dgmg.DeepMG(A, K=K, PR_stencil_type=PR_stencil_type, max_levels=num_levels)
    opt_par = deep_gmm.optimize(num_iter=num_iter, step_size=step_size, batch_size=batch_size, init_point=init_point)
    print("Optimization...Done")
    opt_par_n[n] = opt_par
    convergence[n] = deep_gmm.get_optimizer_convergence()
    conv_time[n] = deep_gmm.get_optimization_time()
    deep_gmm.update_prd(opt_par)
    current_rho = deep_gmm.compute_rho()
    rhos_opt.append(current_rho)
    print("Linear spectral radius = {}".format(rhos_linear[i]))
    print("Optimized spectral radius = {}".format(current_rho))
    
print("Make spectral radius smaller with optimization of P and R...Done")
np.savez("./poisson_1d_opt_par", opt_par=opt_par)