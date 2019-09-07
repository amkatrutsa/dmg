import autograd.numpy as np
import numpy as pure_np
import autograd
from autograd.extend import primitive, defvjp
import scipy.sparse as spsp
from numba import njit

########### CSR 3 mat ###########
@primitive
def csr_3mat(A_data, A_indptr, A_indices, 
               B_data, B_indptr, B_indices, 
               C_data, C_indptr, C_indices,
               m, n, k, l):
    A_csr = spsp.csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))
    B_csr = spsp.csr_matrix((B_data, B_indices, B_indptr), shape=(n, k))
    C_csr = spsp.csr_matrix((C_data, C_indices, C_indptr), shape=(k, l))
    D_csr = A_csr.dot(B_csr).dot(C_csr)
    D_csr.eliminate_zeros()
    D_csr.sort_indices()
    return D_csr.data, D_csr.indices, D_csr.indptr

@njit
def prune_csr_matrix(ref_indptr, ref_indices,
                     pruned_indptr, pruned_indices, pruned_data):
    A_grad = pure_np.zeros_like(ref_indices)
    value_counter = 0
    for i in range(ref_indptr.shape[0] - 1):
        num_col = len(ref_indices[ref_indptr[i]:ref_indptr[i+1]])
        for k in range(ref_indptr[i], ref_indptr[i+1]):
            for j in range(pruned_indptr[i], pruned_indptr[i+1]):
                if ref_indices[k] == pruned_indices[j]:
                    A_grad[k] = pruned_data[j]
        value_counter += num_col
    return A_grad

def csr_3mat_vjp_Adata(g, ans, A_data, A_indptr, A_indices, 
                       B_data, B_indptr, B_indices, 
                       C_data, C_indptr, C_indices,
                       m, n, k, l):
    g_data = g[0]
#     A_csr = spsp.csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))
    B_csr = spsp.csr_matrix((B_data, B_indices, B_indptr), shape=(n, k))
    C_csr = spsp.csr_matrix((C_data, C_indices, C_indptr), shape=(k, l))
    G_csr = spsp.csr_matrix((g_data, ans[1], ans[2]))
    BC = B_csr.dot(C_csr)
    A_grad_csr = G_csr.dot(BC.transpose().tocsr())
    A_grad_csr.sort_indices()
    A_grad = prune_csr_matrix(A_indptr, A_indices, A_grad_csr.indptr, A_grad_csr.indices, A_grad_csr.data)
    return A_grad

def csr_3mat_vjp_Bdata(g, ans, A_data, A_indptr, A_indices, 
                                 B_data, B_indptr, B_indices, 
                                 C_data, C_indptr, C_indices,
                                 m, n, k, l):
    g_data = g[0]
    A_csr = spsp.csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))
    G_csr = spsp.csr_matrix((g_data, ans[1], ans[2]))
    C_csr = spsp.csr_matrix((C_data, C_indices, C_indptr), shape=(k, l))
    A_csr_t = A_csr.transpose().tocsr()
    C_csr_t = C_csr.transpose().tocsr()
#     A_csr_t.sort_indices()
#     G_csr.sort_indices()
    B_grad_csr = A_csr_t.dot(G_csr).dot(C_csr_t)
#     print(A_grad_csr.has_sorted_indices)
#     B_grad_csr_t = B_grad_csr.transpose()
#     print("before sort", B_grad_csr.data)
    B_grad_csr.sort_indices()
#     print(B_grad_csr.data, B_grad_csr.indices, B_csr.indptr)
#     print("B grad shape", B_grad_csr.data.shape)
#     print("B shape", B_data.shape)
    B_grad = prune_csr_matrix(B_indptr, B_indices, B_grad_csr.indptr, B_grad_csr.indices, B_grad_csr.data)
    return B_grad

def csr_3mat_vjp_Cdata(g, ans, A_data, A_indptr, A_indices, 
                                 B_data, B_indptr, B_indices, 
                                 C_data, C_indptr, C_indices,
                                 m, n, k, l):
    g_data = g[0]
    A_csr = spsp.csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))
    G_csr = spsp.csr_matrix((g_data, ans[1], ans[2]))
    B_csr = spsp.csr_matrix((B_data, B_indices, B_indptr), shape=(n, k))
#     C_csr = spsp.csr_matrix((C_data, C_indices, C_indptr), shape=(k, l))
    AB = A_csr.dot(B_csr)
    AB_t = AB.transpose().tocsr()
    C_grad_csr = AB_t.dot(G_csr).tocsr()
#     print(A_grad_csr.has_sorted_indices)
#     B_grad_csr_t = B_grad_csr.transpose()
#     print("before sort", B_grad_csr.data)
    C_grad_csr.sort_indices()
    C_grad = prune_csr_matrix(C_indptr, C_indices, C_grad_csr.indptr, C_grad_csr.indices, C_grad_csr.data)
    return C_grad
#     return C_grad_csr.data

defvjp(csr_3mat, lambda ans, A_data, A_indptr, A_indices, 
                                 B_data, B_indptr, B_indices, 
                                 C_data, C_indptr, C_indices,
                                 m, n, k, l: 
                           lambda g: csr_3mat_vjp_Adata(g, ans, A_data, A_indptr, A_indices, 
                                 B_data, B_indptr, B_indices, 
                                 C_data, C_indptr, C_indices,
                                 m, n, k, l),
                   lambda ans, A_data, A_indptr, A_indices, 
                                 B_data, B_indptr, B_indices, 
                                 C_data, C_indptr, C_indices,
                                 m, n, k, l: 
                           lambda g: csr_3mat_vjp_Bdata(g, ans, A_data, A_indptr, A_indices, 
                                 B_data, B_indptr, B_indices, 
                                 C_data, C_indptr, C_indices,
                                 m, n, k, l),
                  lambda ans, A_data, A_indptr, A_indices, 
                                 B_data, B_indptr, B_indices, 
                                 C_data, C_indptr, C_indices,
                                 m, n, k, l: 
                           lambda g: csr_3mat_vjp_Cdata(g, ans, A_data, A_indptr, A_indices, 
                                 B_data, B_indptr, B_indices, 
                                 C_data, C_indptr, C_indices,
                                 m, n, k, l),
      argnums=[0, 3, 6])

########### CSR matvec ###########

@primitive
@njit
def csr_matvec(data, indptr, indices, x):
    n = indptr.shape[0] - 1
    y = pure_np.zeros((n, x.shape[1]))
    for j in range(x.shape[1]):
        for i in range(n):
            for k in range(indptr[i], indptr[i+1]):
#                 print(data[k], x[col_idx[k], j])
                y[i, j] += data[k] * x[indices[k], j]
    return y

@primitive
def csr_matvec_x_vjp(g, ans, data, indptr, indices, x):
    return csr_matvec_x_vjp_inner(g, data, indptr, indices, x)
@njit
def csr_matvec_x_vjp_inner(g, data, indptr, indices, x):
    n = indptr.shape[0] - 1
    y = pure_np.zeros_like(x)
    for j in range(x.shape[1]):
        for i in range(n):
            for k in range(indptr[i], indptr[i+1]):
                y[indices[k], j] += data[k] * g[i, j]
    return y

@primitive
def csr_matvec_data_vjp(g, ans, data, indptr, indices, x):
    return csr_matvec_data_vjp_inner(g, data, indptr, indices, x)

@njit
def csr_matvec_data_vjp_inner(g, data, indptr, indices, x):
    n = indptr.shape[0] - 1
    y = pure_np.zeros_like(data)
    for j in range(x.shape[1]):
        for i in range(n):
            for k in range(indptr[i], indptr[i+1]):
                y[k] += g[i, j] * x[indices[k], j]
    return y

defvjp(csr_matvec,
       lambda ans, data, indptr, indices, x: lambda g: csr_matvec_data_vjp(g, ans, data, indptr, indices, x),
       lambda ans, data, indptr, indices, x: lambda g: csr_matvec_x_vjp(g, ans, data, indptr, indices, x),
       argnums=[0, 3])

########### CSR diagonal extraction ###########

@primitive
@njit
def get_sparse_diag(A_values, A_indices, A_indptr, n):
    d = pure_np.zeros((n, 1))
    for i in range(A_indptr.shape[0] - 1):
        for k in range(A_indptr[i], A_indptr[i+1]): 
            if A_indices[k] == i:
                d[i] = A_values[k]
    return d

def get_sparse_diag_vjp_Avalues(g, ans, A_values, A_indices, A_indptr, n):
    return get_sparse_diag_vjp_Avalues_inner(g, A_values, A_indices, A_indptr, n)

@njit
def get_sparse_diag_vjp_Avalues_inner(g, A_values, A_indices, A_indptr, n):
    grad = pure_np.zeros_like(A_values)
    g_ravel = g.ravel()
    for i in range(A_indptr.shape[0] - 1):
        for k in range(A_indptr[i], A_indptr[i+1]): 
            if A_indices[k] == i:
                grad[k] = g_ravel[i]
    return grad

defvjp(get_sparse_diag,
       lambda ans, A_values, A_indices, A_indptr, n: 
       lambda g: get_sparse_diag_vjp_Avalues(g, ans, A_values, A_indices, A_indptr, n), argnums=[0])

########### CSR to dense conversion ###########

@primitive
@njit
def csr2dense(values, indices, indptr, n_col):
    n_row = indptr.shape[0] - 1
    A = pure_np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(indptr[i], indptr[i+1]):
            A[i, indices[j]] = values[j]
#     print(A)
    return A

@primitive
def csr2dense_vjp_values(g, ans, values, indices, indptr, n_col):
    return csr2dense_vjp_values_inner(g, values, indices, indptr, n_col)

@njit
def csr2dense_vjp_values_inner(g, values, indices, indptr, n_col):
#     print(ans)
    grad = pure_np.zeros_like(values)
    n_row = indptr.shape[0] - 1
    for i in range(n_row):
        for j in range(indptr[i], indptr[i+1]):
            grad[j] = g[i, indices[j]]
#     print(grad.shape, grad)
    return grad

defvjp(csr2dense,
       lambda ans, values, indices, indptr, n_col: lambda g: csr2dense_vjp_values(g, ans, values, indices, indptr, n_col))

