from numba import jit
import numpy as pure_np
import autograd.numpy as np
from . import pr_base
from autograd.extend import primitive, defvjp

@primitive
@jit(nopython=True)
def P_matvec(P, X, fine_dim):
    l = X.shape[1]
    m = P.shape[0]
    n_1d = fine_dim
    n = n_1d**2
    Y = pure_np.zeros((n, l))
    for k in range(l):
        for i in range(m):
            for j in range(m):
                for si in range(3):
                    for sj in range(3):
                        Y[2 * (n_1d * i + j) + n_1d * si + sj, k] += P[i, j, si, sj] * X[m * i + j, k]
    return Y

@jit(nopython=True)
def P_matvec_batch_P_inner(g, P, X):
    l = X.shape[1]
    m = P.shape[0]
    n_1d = int(pure_np.sqrt(g.shape[0]))
    Y = pure_np.zeros_like(P)
    for k in range(l):
        for i in range(m):
            for j in range(m):
                for si in range(3):
                    for sj in range(3):
                        Y[i, j, si, sj] += g[2 * (n_1d * i + j) + n_1d * si + sj, k] * X[m * i + j, k]
    return Y

@primitive
def P_matvec_batch_P_vjp(g, ans, P, X, fine_dim):
    return P_matvec_batch_P_inner(g, P, X)

@jit(nopython=True)
def P_matvec_batch_X_inner(g, P, X):
    l = X.shape[1]
    m = P.shape[0]
    n_1d = int(pure_np.sqrt(g.shape[0]))
    Y = pure_np.zeros_like(X)
    for k in range(l):
        for i in range(m):
            for j in range(m):
                for si in range(3):
                    for sj in range(3):
                        Y[m * i + j, k] += P[i, j, si, sj] * g[2 * (n_1d * i + j) + n_1d * si + sj, k]
    return Y

@primitive
def P_matvec_batch_X_vjp(g, ans, P, X, fine_dim):
    return P_matvec_batch_X_inner(g, P, X)

defvjp(P_matvec, lambda ans, P, X, fine_dim: lambda g: P_matvec_batch_P_vjp(g, ans, P, X, fine_dim),
                 lambda ans, P, X, fine_dim: lambda g: P_matvec_batch_X_vjp(g, ans, P, X, fine_dim))

@primitive
@jit(nopython=True)
def R_matvec(R, X, coarse_dim):
    n, l = X.shape
    n_1d = int(pure_np.sqrt(n))
    m = coarse_dim
    Y = pure_np.zeros((m * m, l))
    for k in range(l):
        for i in range(m):
            for j in range(m):
                for si in range(3):
                    for sj in range(3):
                        Y[m * i + j, k] += R[i, j, si, sj] * X[2 * (n_1d * i + j) + n_1d * si + sj, k]
    return Y

@jit(nopython=True)
def R_matvec_batch_R_vjp_inner(g, R, X):
    Y = pure_np.zeros_like(R)
    m = int(pure_np.sqrt(g.shape[0]))
    n, l = X.shape
    n_1d = int(pure_np.sqrt(n))
    for k in range(l):
        for i in range(m):
            for j in range(m):
                for si in range(3):
                    for sj in range(3):
                        Y[i, j, si, sj] += g[m * i + j, k] * X[2 * (n_1d * i + j) + n_1d * si + sj, k]
    return Y

@primitive
def R_matvec_batch_R_vjp(g, ans, R, X, coarse_dim):
    return R_matvec_batch_R_vjp_inner(g, R, X)

@jit(nopython=True)
def R_matvec_batch_X_vjp_inner(g, R, X):
    Y = pure_np.zeros_like(X)
    m = int(pure_np.sqrt(g.shape[0]))
    n, l = X.shape
    n_1d = int(pure_np.sqrt(n))
    for k in range(l):
        for i in range(m):
            for j in range(m):
                for si in range(3):
                    for sj in range(3):
                        Y[2 * (n_1d * i + j) + n_1d * si + sj, k] += R[i, j, si, sj] * g[m * i + j, k]
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
    Anew = pure_np.zeros((m, m, 3, 3))
    for i in range(m): ##P'AR is strange... P_{ki} A_{kl} R_{lj} -> it is also 3-diagonal
        # 0 <= i + s - 1 < m
        # s <= m + 1
        for j in range(m):
            sbi_min = max(0, 1-i)
            sbi_max = min(3, m+1-i)
            sbj_min = max(0, 1-j)
            sbj_max = min(3, m+1-j)
            for si in range(sbi_min, sbi_max):
                ji = i + si - 1 #j index
                for sj in range(sbj_min, sbj_max):
                    jj = j + sj - 1
                    for s1i in range(3): #
                        #-1 <= 2*j+s2-2*i-s1 <= 1
                        smin_i = max(s1i + 2*i - 2*ji - 1, 0) ##P[i, 2*i+s1] * P[j, 2*j+s2] A[2*i+s1, 2*j+s2]
                        smax_i = min(s1i + 2*i - 2*ji + 2, 3) #2*j+s2-2*i-s1+1 = (2*i+2*s-2)-(2*i)-s1+1 = 2*s-s1+1
                        for s1j in range(3):
                            smin_j = max(s1j + 2*j - 2*jj - 1, 0)
                            smax_j = min(s1j + 2*j - 2*jj + 2, 3)
                            for s2i in range(smin_i, smax_i):
                                sloci = 2 * ji + s2i - 2 * i - s1i + 1
                                for s2j in range(smin_j, smax_j):
                                    slocj = 2 * jj + s2j - 2 * j - s1j + 1
                                    Anew[i, j, si, sj] += R[i, j, s1i, s1j] * P[ji, jj, s2i, s2j] * A[2*i+s1i, 2*j + s1j, sloci, slocj]
    return Anew

@jit(nopython=True)
def generate_RAP_R_vjp_inner(g, R, A, P):
    m = g.shape[0]
    res = pure_np.zeros((m, m, 3, 3))
    for i in range(m): ##P'AR is strange... P_{ki} A_{kl} R_{lj} -> it is also 3-diagonal
        # 0 <= i + s - 1 < m
        # s <= m + 1
        for j in range(m):
            sbi_min = max(0, 1-i)
            sbi_max = min(3, m+1-i)
            sbj_min = max(0, 1-j)
            sbj_max = min(3, m+1-j)
            for si in range(sbi_min, sbi_max):
                ji = i + si - 1 #j index
                for sj in range(sbj_min, sbj_max):
                    jj = j + sj - 1
                    for s1i in range(3): #
                        #-1 <= 2*j+s2-2*i-s1 <= 1
                        smin_i = max(s1i + 2*i - 2*ji - 1, 0) ##P[i, 2*i+s1] * P[j, 2*j+s2] A[2*i+s1, 2*j+s2]
                        smax_i = min(s1i + 2*i - 2*ji + 2, 3) #2*j+s2-2*i-s1+1 = (2*i+2*s-2)-(2*i)-s1+1 = 2*s-s1+1
                        for s1j in range(3):
                            smin_j = max(s1j + 2*j - 2*jj - 1, 0)
                            smax_j = min(s1j + 2*j - 2*jj + 2, 3)
                            for s2i in range(smin_i, smax_i):
                                sloci = 2 * ji + s2i - 2 * i - s1i + 1
                                for s2j in range(smin_j, smax_j):
                                    slocj = 2 * jj + s2j - 2 * j - s1j + 1
                                    res[i, j, s1i, s1j] += g[i, j, si, sj] * P[ji, jj, s2i, s2j] * A[2*i+s1i, 2*j + s1j, sloci, slocj]
    return res
    
@primitive
def generate_RAP_R_vjp(g, ans, R, A, P, coarse_dim):
    return generate_RAP_R_vjp_inner(g, R, A, P)

@jit(nopython=True)
def generate_RAP_A_vjp_inner(g, R, A, P):
    n = A.shape[0]
    m = g.shape[0]
    res = pure_np.zeros((n, n, 3, 3))
    for i in range(m): ##P'AR is strange... P_{ki} A_{kl} R_{lj} -> it is also 3-diagonal
        # 0 <= i + s - 1 < m
        # s <= m + 1
        for j in range(m):
            sbi_min = max(0, 1-i)
            sbi_max = min(3, m+1-i)
            sbj_min = max(0, 1-j)
            sbj_max = min(3, m+1-j)
            for si in range(sbi_min, sbi_max):
                ji = i + si - 1 #j index
                for sj in range(sbj_min, sbj_max):
                    jj = j + sj - 1
                    for s1i in range(3): #
                        #-1 <= 2*j+s2-2*i-s1 <= 1
                        smin_i = max(s1i + 2*i - 2*ji - 1, 0) ##P[i, 2*i+s1] * P[j, 2*j+s2] A[2*i+s1, 2*j+s2]
                        smax_i = min(s1i + 2*i - 2*ji + 2, 3) #2*j+s2-2*i-s1+1 = (2*i+2*s-2)-(2*i)-s1+1 = 2*s-s1+1
                        for s1j in range(3):
                            smin_j = max(s1j + 2*j - 2*jj - 1, 0)
                            smax_j = min(s1j + 2*j - 2*jj + 2, 3)
                            for s2i in range(smin_i, smax_i):
                                sloci = 2 * ji + s2i - 2 * i - s1i + 1
                                for s2j in range(smin_j, smax_j):
                                    slocj = 2 * jj + s2j - 2 * j - s1j + 1
                                    res[2*i+s1i, 2*j + s1j, sloci, slocj] += R[i, j, s1i, s1j] * P[ji, jj, s2i, s2j] * g[i, j, si, sj]
    return res

@primitive
def generate_RAP_A_vjp(g, ans, R, A, P, coarse_dim):
    return generate_RAP_A_vjp_inner(g, R, A, P)

@jit(nopython=True)
def generate_RAP_P_vjp_inner(g, R, A, P):
    m = g.shape[0]
    res = pure_np.zeros((m, m, 3, 3))
    for i in range(m): ##P'AR is strange... P_{ki} A_{kl} R_{lj} -> it is also 3-diagonal
        # 0 <= i + s - 1 < m
        # s <= m + 1
        for j in range(m):
            sbi_min = max(0, 1-i)
            sbi_max = min(3, m+1-i)
            sbj_min = max(0, 1-j)
            sbj_max = min(3, m+1-j)
            for si in range(sbi_min, sbi_max):
                ji = i + si - 1 #j index
                for sj in range(sbj_min, sbj_max):
                    jj = j + sj - 1
                    for s1i in range(3): #
                        #-1 <= 2*j+s2-2*i-s1 <= 1
                        smin_i = max(s1i + 2*i - 2*ji - 1, 0) ##P[i, 2*i+s1] * P[j, 2*j+s2] A[2*i+s1, 2*j+s2]
                        smax_i = min(s1i + 2*i - 2*ji + 2, 3) #2*j+s2-2*i-s1+1 = (2*i+2*s-2)-(2*i)-s1+1 = 2*s-s1+1
                        for s1j in range(3):
                            smin_j = max(s1j + 2*j - 2*jj - 1, 0)
                            smax_j = min(s1j + 2*j - 2*jj + 2, 3)
                            for s2i in range(smin_i, smax_i):
                                sloci = 2 * ji + s2i - 2 * i - s1i + 1
                                for s2j in range(smin_j, smax_j):
                                    slocj = 2 * jj + s2j - 2 * j - s1j + 1
                                    res[ji, jj, s2i, s2j] += R[i, j, s1i, s1j] * g[i, j, si, sj] * A[2*i+s1i, 2*j + s1j, sloci, slocj]
    return res

@primitive
def generate_RAP_P_vjp(g, ans, R, A, P, coarse_dim):
    return generate_RAP_P_vjp_inner(g, R, A, P)

defvjp(generate_RAP, lambda ans, R, A, P, coarse_dim: lambda g: generate_RAP_R_vjp(g, ans, R, A, P, coarse_dim),
                     lambda ans, R, A, P, coarse_dim: lambda g: generate_RAP_A_vjp(g, ans, R, A, P, coarse_dim), 
                     lambda ans, R, A, P, coarse_dim: lambda g: generate_RAP_P_vjp(g, ans, R, A, P, coarse_dim))

class PR_2d_m9p(pr_base.PR):
    def __init__(self, n, R=None, P=None):
        m = (n+1) // 2 - 1
        if m % 2 == 0:
            m -= 1
        if R is None and P is None:
            P = np.zeros((m, m, 3, 3))
            R = np.zeros((m, m, 3, 3))
            for i in range(m):
                for j in range(m):
                    P[i, j, 0, :] = [0.25, 0.5, 0.25]
                    P[i, j, 1, :] = [0.5, 1, 0.5]
                    P[i, j, 2, :] = [0.25, 0.5, 0.25]
            R = 0.25 * P
        super().__init__(R, P, m, n, problem_dim=2)
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
        current_pr = PR_2d_m9p(n)
        PR_list.append(current_pr)
        n = (n+1) // 2 - 1
        if n < 3:
            break
    PR_list.append([0])
    return PR_list[::-1]

def generate_from_P_R_lists(P_list, R_list, fine_dims):
    PR_list = [[0]]
    for i, (P, R) in enumerate(zip(P_list, R_list)):
        current_pr = PR_2d_m9p(fine_dims[i], R, P)
        PR_list.append(current_pr)
    return PR_list