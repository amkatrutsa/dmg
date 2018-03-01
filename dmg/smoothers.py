import autograd.numpy as np

def damped_jacobi(A, rhs, U, par):
    n = A.shape
    if A.problem_dim == 1:
        inv_D_vec = 1.0 / A.get_diagonal().reshape(n, 1)
    elif A.problem_dim == 2:
        inv_D_vec = 1.0 / A.get_diagonal().reshape(n, 1)
    X = U
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    for i in range(par["max_iter"]):
        AX = A.dot(X)
        bAX = rhs - AX
        dX = inv_D_vec * bAX
        X = X + par["damp"] * dX
    return X

def damped_jacobi_dmg(A, rhs, U, par):
    n1d = A.shape[0]
    if len(A.shape) == 2:
        inv_D_vec = 1.0 / A[:, 1].reshape(n1d, 1)
    elif len(A.shape) == 4:
        inv_D_vec = 1.0 / A[:, :, 1, 1].reshape(n1d * n1d, 1)
    X = U
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    for i in range(par["max_iter"]):
        AX = par["Amatvec"](A, X)
        bAX = rhs - AX
        dX = inv_D_vec * bAX
        X = X + par["damp"] * dX
    return X