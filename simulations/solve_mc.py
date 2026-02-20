import numpy as np
import cvxpy as cp

from math import sqrt
from numpy.linalg import norm

# ************************************************************ #
#                         PCA solver                           #
# ************************************************************ #

def pca(cov):
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    # eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs
    


# ************************************************************ #
#                           Solvers                            #
# ************************************************************ #


def solveRight(Us, observed_entries, omega_indices, ncol, lambdaR=0,
               type='wc', **kwargs):
    """
    """
    # Options
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    solver = kwargs.get('solver', cp.SCS)
    verbose = kwargs.get('verbose', False)
    if verbose: print("\nRIGHT----")

    # Parameters
    num_e = len(Us)
    nrow = [U.shape[0] for U in Us]
    nrow_pool = sum(nrow)
    rank = Us[0].shape[1]
    
    # Problem
    V_T = cp.Variable((rank, ncol))
    errors = []
    for s in range(num_e):
        diff = (Us[s] @ V_T)[omega_indices[s]] - observed_entries[s]
        if type == 'wc':
            errors.append(cp.norm(diff, p='fro') / sqrt(nrow[s])) #/ len(observed_entries[s])) # TODO: this is new (observed entries)
        elif type == 'pool':
            errors.append(diff)
    if type == 'wc':
        obj = cp.Minimize(cp.max(cp.hstack(errors)) +\
                          lambdaR * cp.norm(V_T, 'fro') / sqrt(rank*ncol))
    elif type == 'pool':
        obj = cp.Minimize(cp.norm(cp.hstack(errors)) / sqrt(nrow_pool) +\
                           lambdaR * cp.norm(V_T, 'fro') / sqrt(rank*ncol))
    prob = cp.Problem(obj)
    prob.solve(solver=solver, verbose=verbose)

    if verbose: 
        print("right status:", prob.status)
        print(f"\nRIGHT objective value: {prob.value}\n")

        # err0 = errors[0].value
        # err1 = errors[1].value
        # op = "<" if err0 < err1 else ">" if err0 > err1 else "="
        # print(f"\t## ind errs: {err0:.5f} {op} {err1:.5f} \t prob: {prob.value:.5f} \t obj: {obj.value:.5f}")

    V = (V_T.value).T
    if returnObjectiveValue:
        return (V, prob.value)
    else:
        return V
     
    
def solveLeft_np(V, observed_entries, omega_indices, nrow, 
                 type='wc', lambdaL=0.01, **kwargs):
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    verbose = kwargs.get('verbose', False)
    
    ncol, rank = V.shape
    num_e = len(observed_entries)
    
    if verbose: print("\nLEFT [NUMPY] ----")
    
    Us = []
    errors = []

    for s in range(num_e):
        U = np.zeros((nrow[s], rank))
        rows, cols = omega_indices[s]
        values = observed_entries[s]

        for i in range(nrow[s]):
            # get indices j where row i has an observation
            mask = (rows == i)
            item_indices = cols[mask]
            ratings = values[mask]

            if len(item_indices) > 0:
                V_i = V[item_indices]  # shape (num_items_i, rank)
                y_i = ratings  # shape (num_items_i,)
                A = V_i.T @ V_i + lambdaL * np.eye(rank) / sqrt(rank * nrow[s])
                b = V_i.T @ y_i
                U[i, :] = np.linalg.solve(A, b)

        Us.append(U)
        recon = (U @ V.T)[rows, cols]
        err = np.linalg.norm(recon - values) ** 2 / nrow[s]
        errors.append(err)

    if returnObjectiveValue:
        return Us, np.sum(errors)
    else:
        return Us


def solveRight_np(Us, observed_entries, omega_indices, ncol, lambdaR=0,
                  **kwargs):
    """
    Numpy-only right factor solver for the pooled objective.

    The pool right step decouples column-by-column: column j of V only
    appears in residuals at column j, so we solve ncol independent least
    squares problems.

    With lambdaR=0  : plain lstsq (handles rank deficiency).
    With lambdaR>0  : ridge via normal equations.
    """
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    verbose = kwargs.get('verbose', False)

    if verbose: print("\nRIGHT [NUMPY pool] ----")

    num_e = len(Us)
    rank = Us[0].shape[1]
    V = np.zeros((ncol, rank))

    for j in range(ncol):
        U_j_list = []
        y_j_list = []

        for s in range(num_e):
            rows, cols = omega_indices[s]
            values = observed_entries[s]
            mask = (cols == j)
            user_indices = rows[mask]
            ratings = values[mask]
            if len(user_indices) > 0:
                U_j_list.append(Us[s][user_indices, :])
                y_j_list.append(ratings)

        if U_j_list:
            U_j = np.vstack(U_j_list)   # (n_obs_j, rank)
            y_j = np.concatenate(y_j_list)
            if lambdaR == 0:
                V[j, :] = np.linalg.lstsq(U_j, y_j, rcond=None)[0]
            else:
                A = U_j.T @ U_j + lambdaR * np.eye(rank) / sqrt(rank * ncol)
                b = U_j.T @ y_j
                V[j, :] = np.linalg.solve(A, b)

    if returnObjectiveValue:
        nrow = [U.shape[0] for U in Us]
        errors = []
        for s in range(num_e):
            rows, cols = omega_indices[s]
            values = observed_entries[s]
            recon = (Us[s] @ V.T)[rows, cols]
            errors.append(np.linalg.norm(recon - values) ** 2 / nrow[s])
        return V, np.sum(errors)
    else:
        return V


def solveLeft(V, observed_entries, omega_indices, nrow, lambdaL=0,
                             type='wc', **kwargs):
    """
    """
    # Options
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    solver = kwargs.get('solver', cp.SCS)
    verbose = kwargs.get('verbose', False)
    if verbose: print("\nLEFT----")

    if isinstance(verbose, int):
        if verbose > 1:
            verbose = True
        else:
            verbose = False
    
    rank = V.shape[1]
    num_e = len(observed_entries)

    # set up the problem
    Us = [cp.Variable((ne, rank)) for ne in nrow]
    errors = []
    for s in range(num_e):
        diff = (Us[s] @ V.T)[omega_indices[s]] - observed_entries[s]
        errors.append(diff)
    obj = cp.Minimize(cp.norm(cp.hstack(errors)))# TODO: / n
    
    prob = cp.Problem(obj)
    prob.solve(solver=solver, verbose=verbose)

    # compute errors to print out
    if verbose:
        U1 = Us[0].value
        U2 = Us[1].value
        err1 = norm((U1 @ V.T)[omega_indices[0]] - observed_entries[0])
        err2 = norm((U2 @ V.T)[omega_indices[1]] - observed_entries[1])
        
        print(f"\t## err1_np: {err1:.5f},  \t\terr2_np: {err2:.5f}")
        print(f"\t## fullerr_np_sum: {(sqrt(err1**2 + err2**2)):.5f}") #, \tfullerr_np: {fullerr:.5f}")

    Us = [U.value for U in Us]
    if returnObjectiveValue:
        errors = [error.value for error in errors]
        return (Us, obj.value)
    else:
        return Us

    
def get_mc_solution(observed_entries, omega_indices, nrow, ncol, rank, num_e,
                    lambdaL=0, lambdaR=0,
                    type='wc', store_history=False, 
                    init_type='random', init_values=None, 
                    max_iters=50, optCond=None, optTol=1e-4, 
                    verbose=True, opts=None):
    objPrevious = np.inf
    losses = [] if store_history else None
    UV_history = [] if store_history else None

    solveR = solveRight #solveRight_np if type == 'pool' else solveRight

    # Use the fast numpy left solver when lambdaL>0 (regularised, always
    # numerically stable) or when every row has at least `rank` observations
    # (system is overdetermined — check done once here, not per iteration).
    all_overdetermined = all(
        np.min(np.bincount(omega_indices[s][0], minlength=nrow[s])) >= rank
        for s in range(num_e)
    )
    solveL = solveLeft_np if lambdaL > 0 or all_overdetermined else solveLeft

    # set up starting point
    if init_type == 'random': 
        Us = [0.1 * np.random.randn(ne, rank) for ne in nrow]
    elif init_type == 'svd':
        Us = []
        for s in range(num_e):
            proj_matrix = np.zeros((nrow[s], ncol))
            proj_matrix[omega_indices[s]] = observed_entries[s]
            U, _, _ = np.linalg.svd(proj_matrix, full_matrices=False)
            Us.append(U[:, :rank])
    elif init_type == 'init':
        Us = init_values
    else:
        raise ValueError("Invalid init_type. Use 'random', 'svd', or 'init'.")
    
    # Optimize
    for T in range(max_iters):
        if verbose:
            print(f"Iteration {T}")
        V, objValue = solveR(Us, observed_entries, omega_indices, ncol, 
                                 lambdaR=lambdaR, type=type,
                                 **opts, returnObjectiveValue=True)
        Us, objValue = solveL(V, observed_entries, omega_indices, nrow, 
                                    lambdaL=lambdaL, type=type,
                                    **opts, returnObjectiveValue=True)

        if store_history:
            losses.append(objValue)
            UV_history.append((Us, V))

        if optCond(objValue, objPrevious) < optTol:
            if verbose:
                print("\n**************************************************")
                print("Optimality conditions satisfied.")
                print(f"Iteration {T}, Objective value = {objValue:5.3g}")
                print("**************************************************")
            break
        else:
            if verbose:
                print(f"Iteration {T}: Objective = {objValue}", end='\r')
            objPrevious = objValue

        if T == max_iters - 1:
            print("Warning: maximum number of iterations reached.")
            if verbose:
                print("\n#################################################")
                print("!!! Maximum number of iterations reached !!!")
                print(f"Iteration {T}, Objective value = {objValue:5.3g}")
                print("#################################################")

    if store_history:
        return Us, V, losses, UV_history
    else:
        return Us, V
    

def altMinSense(observed_entries, omega_indices, nrow, ncol, rank,
                lambdaL=0, lambdaR=0, # TODO document
                type='wc', store_history=False, 
                init_type='random', init_values=None, reruns=1,
                **kwargs):
    """
    altMinSense(observed_entries, omega_indices, nrow, ncol, rank, type, 
                outerr, store_history, **kwargs)
    The alternating minimization algorithm for matrix completion, either in the
    worst-case or pooled sense.

    It is assumed that there are two environments, each with a matrix of
    size m-by-n.

    Parameters:
    ----------
    observed_entries : list of arrays
        For each environment, the observed entries of the corresponding matrix.
    omega_indices : list of arrays
        The indices of the observed entries of each matrix in each environment,
        i.e., e entries of [rows, cols].
    nrow : int or list of int
        The number of rows in each environment of the output matrix M.
    ncol : int
        The number of columns in each environment of the output matrix M.
    rank : int
        The rank of the output matrix M_hat.
    lambdaL : float, optional
        The regularization parameter for the left factors. Default is 0.
    lambdaR : float, optional
        The regularization parameter for the right factor. Default is 0.
    type : str, optional
        The type of matrix completion problem to solve: 'wc' (worst-case) or 'pool'.
        Default is 'wc'.
    store_history : bool, optional
        If True, the history of the objective values is stored. Default is False.
    init_type : str, optional
        The type of initialization for the left factors: 'random', 'svd', or 'init'.
        Default is 'random'.
    init_values : list of arrays, optional
        The initial values for the left factors if init_type is 'init'.
    reruns: int, optional
        The number of reruns for the optimization. Default is 1.
    kwargs : dict, optional
        Additional options:
        - max_iters : int, maximum allowable number of iterations (default: 50).
        - optCond : callable, optimality condition function (default: absolute difference).
        - optTol : float, optimality tolerance (default: 1e-4).
        - solver : cvxpy solver to use (default: SCS).
        - verbose : int or bool, verbosity level (default: True).

    Returns:
    -------
    Us : list of arrays
        The two left m-by-r factors.
    V : array
        The right n-by-r factor.
    losses : list, optional
        The history of objective values if store_history is True.
    """
    max_iters = kwargs.get('max_iters', 50)
    optCond = kwargs.get('optCond', lambda x, y: np.abs(x - y))
    optTol = kwargs.get('optTol', 1e-4)
    verbose = kwargs.get('verbose', True)
    opts = kwargs.get('methodOptions', {'solver': cp.SCS, 'verbose': verbose})
    num_e = len(observed_entries)

    if isinstance(nrow, int):
        nrow = [nrow] * num_e
        Warning("nrow is not a list, assuming all environments have the same number of rows")

    min_loss = np.inf
    best_Us, best_V, best_losses, best_UV_history = None, None, None, None
    for i in range(reruns):
        if verbose:
            print(f"\nRerun {i+1} of {reruns}:")
        Us, V, loss, UV_history = get_mc_solution(
            observed_entries, 
            omega_indices,
            nrow, ncol, rank, num_e,
            type=type, 
            store_history=True,
            init_type=init_type,
            init_values=init_values,
            max_iters=max_iters,
            optCond=optCond,
            optTol=optTol,
            verbose=verbose, 
            opts=opts,
            lambdaL=lambdaL, 
            lambdaR=lambdaR,
        )
        if loss[-1] < min_loss:
            min_loss = loss[-1]
            best_Us, best_V, best_losses, best_UV_history = Us, V, loss, UV_history
            if verbose:
                print(f"Best loss so far: {min_loss:.5f}")

    if store_history:
        return best_Us, best_V, best_losses, best_UV_history
    else:
        return best_Us, best_V
    

# ************************************************************ #
#                         OOS Solvers                          #
# ************************************************************ #

def impute_X_pinv(right_factor, omega_indices, Ms_true):
    n_envs = len(Ms_true)
    nrow = Ms_true[0].shape[0]
    rank = right_factor.shape[1]
    Us_hat = [np.zeros((nrow, rank)) for _ in range(n_envs)]
    for i in range(n_envs):
        rows, cols = omega_indices[i]
        for j in range(nrow):
            obs_cols = cols[rows == j]
            R_obs = right_factor[obs_cols]       # (n_obs, rank)
            y_obs = Ms_true[i][j, obs_cols]      # (n_obs,)
            Us_hat[i][j, :] = np.linalg.lstsq(R_obs, y_obs, rcond=None)[0]
    return Us_hat


def get_Mhat_from_right_factor(right_factor, omega_indices_test, Ms_true):
    """
    Get the estimated matrices from the right factor and the test indices.
    The estimated matrices are obtained by using the impute_X_pinv function.
    """
    Us_hat = impute_X_pinv(right_factor, omega_indices_test, Ms_true)
    Ms_hat = [U @ right_factor.T for U in Us_hat]
    return Ms_hat
