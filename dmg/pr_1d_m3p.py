from numba import jit
import numpy as pure_np
import autograd.numpy as np
from autograd.extend import primitive, defvjp 
from . import pr_base

@primitive
@jit(nopython=True)
def P_matvec(P, X, fine_dim):
    l = X.shape[1]
    m = P.shape[0]
    n = fine_dim
    Y = pure_np.zeros((n, l))
    for j in range(l):
        for i in range(m):
            for s in range(3):
                Y[2*i+s, j] += P[i, s] * X[i, j]
    return Y

@jit(nopython=True)
def P_matvec_batch_P_inner(g, P, X):
    m, l = X.shape
    Y = pure_np.zeros_like(P)
    for j in range(l):
        for i in range(m):
            for s in range(3):
                Y[i, s] += g[2*i+s, j] * X[i, j]
    return Y

@primitive
def P_matvec_batch_P_vjp(g, ans, P, X, fine_dim):
    return P_matvec_batch_P_inner(g, P, X)

@jit(nopython=True)
def P_matvec_batch_X_vjp_inner(g, P, X):
    m, l = X.shape
    Y = pure_np.zeros_like(X)
    for j in range(l):
        for i in range(m):
            for s in range(3):
                Y[i, j] += P[i, s] * g[2*i+s, j]
    return Y

@primitive
def P_matvec_batch_X_vjp(g, ans, P, X, fine_dim):
    return P_matvec_batch_X_vjp_inner(g, P, X)

defvjp(P_matvec, lambda ans, P, X, fine_dim: lambda g: P_matvec_batch_P_vjp(g, ans, P, X, fine_dim),
                 lambda ans, P, X, fine_dim: lambda g: P_matvec_batch_X_vjp(g, ans, P, X, fine_dim))

@primitive
@jit(nopython=True)
def R_matvec(R, X, coarse_dim):
    l = X.shape[1]
    m = coarse_dim
    Y = pure_np.zeros((m, l))
    for j in range(l):
        for i in range(m):
            for s in range(3):
                Y[i, j] += R[i, s] * X[2*i+s, j]
    return Y

@jit(nopython=True)
def R_matvec_batch_R_vjp_inner(g, R, X):
    Y = pure_np.zeros_like(R)
    m, l = g.shape
    for j in range(l):
        for i in range(m):
            for s in range(3):
                Y[i, s] += g[i, j] * X[2*i+s, j]
    return Y

@primitive
def R_matvec_batch_R_vjp(g, ans, R, X, coarse_dim):
    return R_matvec_batch_R_vjp_inner(g, R, X)

@jit(nopython=True)
def R_matvec_batch_X_vjp_inner(g, R, X):
    Y = pure_np.zeros_like(X)
    m, l = g.shape
    for j in range(l):
        for i in range(m):
            for s in range(3):
                Y[2*i+s, j] += R[i, s] * g[i, j]
    return Y

@primitive
def R_matvec_batch_X_vjp(g, ans, R, X, coarse_dim):
    return R_matvec_batch_X_vjp_inner(g, R, X)

defvjp(R_matvec, lambda ans, R, X, coarse_dim: lambda g: R_matvec_batch_R_vjp(g, ans, R, X, coarse_dim),
                 lambda ans, R, X, coarse_dim: lambda g: R_matvec_batch_X_vjp(g, ans, R, X, coarse_dim))

@primitive
@jit(nopython=True)
def generate_RAP(R, A, P, coarse_dim): #Given R, A, P compute efficiently the Galerkin projection
    m = coarse_dim
    Anew = pure_np.zeros((m, 3))
    for i in range(m): ##P'AR is strange... P_{ki} A_{kl} R_{lj} -> it is also 3-diagonal
        # 0 <= i + s - 1 < m
        # s <= m + 1
        sb_min = max(0, 1-i)
        sb_max = min(3, m+1-i)
        for s in range(sb_min, sb_max):
            j = i + s - 1 #j index
            for s1 in range(3): #
                #-1 <= 2*j+s2-2*i-s1 <= 1
                smin = max(s1 + 2*i - 2*j - 1, 0) ##P[i, 2*i+s1] * P[j, 2*j+s2] A[2*i+s1, 2*j+s2]
                smax = min(s1 + 2*i - 2*j + 2, 3) #2*j+s2-2*i-s1+1 = (2*i+2*s-2)-(2*i)-s1+1 = 2*s-s1+1
                for s2 in range(smin, smax):
                    sloc = 2 * j + s2 - 2 * i - s1 + 1
                    Anew[i, s] = Anew[i, s] + R[i, s1] * P[j, s2] * A[2*i+s1, sloc]
                    #Anew[i, s] += P[i, s1] * P[j, s2] * A_mat[2*i+s1, 2*j+s2]#A[2*i+s1, sloc]#A[2*i+s1, 2*j+s2]
                    #Anew[i, s] += P[i, s1] * P[j, s2] * A[2*i+s1, 2*j+s2-2*i-s1+1]#
    return Anew

@jit(nopython=True)
def generate_RAP_R_vjp_inner(g, R, A, P):
    m = g.shape[0]
    res = pure_np.zeros((m, 3))
    for i in range(m): 
        # 0 <= i + s - 1 < m
        # s <= m + 1
        sb_min = max(0, 1-i)
        sb_max = min(3, m+1-i)
        for s in range(sb_min, sb_max):
            j = i + s - 1 #j index
            for s1 in range(3): #
                #-1 <= 2*j+s2-2*i-s1 <= 1
                smin = max(s1 + 2*i - 2*j - 1, 0) ##P[i, 2*i+s1] * P[j, 2*j+s2] A[2*i+s1, 2*j+s2]
                smax = min(s1 + 2*i - 2*j + 2, 3) #2*j+s2-2*i-s1+1 = (2*i+2*s-2)-(2*i)-s1+1 = 2*s-s1+1
                for s2 in range(smin, smax):
                    sloc = 2 * j + s2 - 2 * i - s1 + 1
                    res[i, s1] = res[i, s1] + g[i, s]*P[j, s2]*A[2*i+s1, sloc]
                    #Anew[i, s] = Anew[i, s] + R[i, s1] * P[j, s2] * A[2*i+s1, sloc]       
    return res
    
@primitive
def generate_RAP_R_vjp(g, ans, R, A, P, coarse_dim): #Ax -> A' g g is the size of output (m, 3) 
    #and result is the size of input (m, 3)
    return generate_RAP_R_vjp_inner(g, R, A, P)

@jit(nopython=True)
def generate_RAP_A_vjp_inner(g, R, A, P):
    n = A.shape[0]
    m = g.shape[0]
    res = pure_np.zeros((n, 3))
    for i in range(m): 
        # 0 <= i + s - 1 < m
        # s <= m + 1
        sb_min = max(0, 1-i)
        sb_max = min(3, m+1-i)
        for s in range(sb_min, sb_max):
            j = i + s - 1 #j index
            for s1 in range(3): #
                #-1 <= 2*j+s2-2*i-s1 <= 1
                smin = max(s1 + 2*i - 2*j - 1, 0) ##P[i, 2*i+s1] * P[j, 2*j+s2] A[2*i+s1, 2*j+s2]
                smax = min(s1 + 2*i - 2*j + 2, 3) #2*j+s2-2*i-s1+1 = (2*i+2*s-2)-(2*i)-s1+1 = 2*s-s1+1
                for s2 in range(smin, smax):
                    sloc = 2 * j + s2 - 2 * i - s1 + 1
                    res[2*i+s1, sloc] = res[2*i+s1, sloc] + R[i, s1] * P[j, s2] * g[i, s]
    return res

@primitive
def generate_RAP_A_vjp(g, ans, R, A, P, coarse_dim):
    return generate_RAP_A_vjp_inner(g, R, A, P)

@jit(nopython=True)
def generate_RAP_P_vjp_inner(g, R, A, P):
    m = g.shape[0]
    res = pure_np.zeros((m, 3))
    for i in range(m): 
        # 0 <= i + s - 1 < m
        # s <= m + 1
        sb_min = max(0, 1-i)
        sb_max = min(3, m+1-i)
        for s in range(sb_min, sb_max):
            j = i + s - 1 #j index
            for s1 in range(3): #
                #-1 <= 2*j+s2-2*i-s1 <= 1
                smin = max(s1 + 2*i - 2*j - 1, 0) ##P[i, 2*i+s1] * P[j, 2*j+s2] A[2*i+s1, 2*j+s2]
                smax = min(s1 + 2*i - 2*j + 2, 3) #2*j+s2-2*i-s1+1 = (2*i+2*s-2)-(2*i)-s1+1 = 2*s-s1+1
                for s2 in range(smin, smax):
                    sloc = 2 * j + s2 - 2 * i - s1 + 1
                    res[j, s2] = res[j, s2] + R[i, s1]*g[i, s]*A[2*i+s1, sloc]
                    #Anew[i, s] = Anew[i, s] + R[i, s1] * P[j, s2] * A[2*i+s1, sloc]       
    return res

@primitive
def generate_RAP_P_vjp(g, ans, R, A, P, coarse_dim):
    return generate_RAP_P_vjp_inner(g, R, A, P)

defvjp(generate_RAP, lambda ans, R, A, P, coarse_dim: lambda g: generate_RAP_R_vjp(g, ans, R, A, P, coarse_dim),
                     lambda ans, R, A, P, coarse_dim: lambda g: generate_RAP_A_vjp(g, ans, R, A, P, coarse_dim), 
                     lambda ans, R, A, P, coarse_dim: lambda g: generate_RAP_P_vjp(g, ans, R, A, P, coarse_dim))

class PR_1d_m3p(pr_base.PR):
    def __init__(self, n, P=None, R=None):
        m = (n+1)//2-1
        if m % 2 == 0:
            m -= 1
        if P is None and R is None:
            P = np.zeros((m, 3))
            R = np.zeros((m, 3))
            for i in range(m):
                P[i, :] = np.array([1.0, 2.0, 1.0])
                R[i, :] = np.array([1.0, 2.0, 1.0])
            P = 0.5 * P
            R = 0.25 * R
        super().__init__(R, P, m, n, problem_dim=1)
    def _P_matvec(self, P, X):
        return P_matvec(P, X, self._fine_dim)
    def _R_matvec(self, R, X):
        return R_matvec(R, X, self._coarse_dim)
    def _A_projection(self, R, A, P, coarse_dim):
        return generate_RAP(R, A, P, coarse_dim)
    
def generate_PR_list(num_levels, n):
    PR_list = []
    while num_levels > 0:
        num_levels -= 1
        if n % 2 == 0:
            n -= 1
        current_pr = PR_1d_m3p(n)
        PR_list.append(current_pr)
        n = (n+1) // 2 - 1
        if n < 3:
            break
    PR_list.append([0])
    return PR_list[::-1]

def generate_from_P_R_lists(P_list, R_list, fine_dims):
    PR_list = [[0]]
    for i, (P, R) in enumerate(zip(P_list, R_list)):
        current_pr = PR_1d_m3p(fine_dims[i], P, R)
        PR_list.append(current_pr)
    return PR_list