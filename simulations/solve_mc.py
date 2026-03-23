import cvxpy as cp
import numpy as np
from math import sqrt

def compute_objective(Us, V, observed_entries, observed_indices, type='wc'):
    """Computes the reconstruction error based on the chosen framework."""
    num_e = len(Us)
    errors = []
    nrow_pool = 0
    
    for s in range(num_e):
        nrow_s = Us[s].shape[0]
        nrow_pool += nrow_s
        rows, cols = observed_indices[s]
        
        # Reconstruct only the observed entries to save memory/compute
        recon = (Us[s] @ V.T)[rows, cols]
        diff = recon - observed_entries[s]
        
        if type == 'wc':
            # Frobenius norm of the difference scaled by sqrt(rows)
            errors.append(np.linalg.norm(diff) / sqrt(nrow_s))
        elif type == 'pool':
            # Keep raw differences to compute the global norm later
            errors.append(diff)
            
    if type == 'wc':
        return np.max(errors)
    elif type == 'pool':
        all_diffs = np.concatenate(errors)
        return np.linalg.norm(all_diffs) / sqrt(nrow_pool)
    

def init_U_svd_simple(observed_entries, observed_indices, nrow, ncol, rank):
    """
    Standard SVD initialization.
    Fills a zero-matrix with observed entries and takes the top 'rank' left singular vectors.
    """
    num_e = len(observed_entries)
    Us = []
    
    for s in range(num_e):
        # Create a zero-filled projection matrix
        proj_matrix = np.zeros((nrow, ncol))
        
        # Populate with observed entries
        proj_matrix[observed_indices[s]] = observed_entries[s]
        
        # Compute SVD
        U, _, _ = np.linalg.svd(proj_matrix, full_matrices=False)
        
        # Keep only the top 'rank' components
        Us.append(U[:, :rank])
        
    return Us


def update_U(V, observed_entries, observed_indices, nrow, rank):
    """Universal U-update using numpy lstsq (same for pool and wc)."""
    Us = []
    num_e = len(observed_entries)
    
    for s in range(num_e):
        U = np.zeros((nrow, rank))
        rows, cols = observed_indices[s]
        values = observed_entries[s]
        
        for i in range(nrow):
            mask = (rows == i)
            item_indices = cols[mask]
            
            if len(item_indices) > 0:
                V_i = V[item_indices]  # Shape: (num_items_i, rank)
                y_i = values[mask]     # Shape: (num_items_i,)
                # Solve U_i * V_i^T = y_i
                U[i, :] = np.linalg.lstsq(V_i, y_i, rcond=None)[0]
                
        Us.append(U)
    return Us


def update_V_pool(Us, observed_entries, observed_indices, ncol, rank):
    """V-update for pooled framework using numpy lstsq."""
    V = np.zeros((ncol, rank))
    num_e = len(Us)
    
    for j in range(ncol):
        U_j_list = []
        y_j_list = []
        
        for s in range(num_e):
            rows, cols = observed_indices[s]
            mask = (cols == j)
            
            if np.any(mask):
                U_j_list.append(Us[s][rows[mask], :])
                y_j_list.append(observed_entries[s][mask])
                
        if U_j_list:
            # Stack equations across all environments for column j
            U_j = np.vstack(U_j_list)
            y_j = np.concatenate(y_j_list)
            V[j, :] = np.linalg.lstsq(U_j, y_j, rcond=None)[0]
            
    return V


def update_V_wc(Us, observed_entries, observed_indices, nrow, ncol, rank):
    """V-update for worst-case framework using cvxpy."""
    V_T = cp.Variable((rank, ncol))
    errors = []
    num_e = len(Us)
    
    for s in range(num_e):
        diff = (Us[s] @ V_T)[observed_indices[s]] - observed_entries[s]
        errors.append(cp.norm(diff, p='fro') / sqrt(nrow))
        
    # Minimax objective
    obj = cp.Minimize(cp.max(cp.hstack(errors)))
    prob = cp.Problem(obj)
    prob.solve(solver=cp.SCS)
    
    return (V_T.value).T


def solve_mcam_multienv(observed_entries, observed_indices, nrow, ncol, rank, 
                        max_iters=50, type='wc', optTol=1e-4, 
                        optCond=lambda x, y: np.abs(x - y), 
                        store_history=False):
    """
    Alternating minimization loop with early stopping and history tracking.
    """
    print("Initializing U via simple SVD...")
    Us = init_U_svd_simple(observed_entries, observed_indices, nrow, ncol, rank)
    V = None
    
    objPrevious = np.inf
    losses = [] if store_history else None

    
    
    for t in range(max_iters):
        # 1. Update V (Right step)
        if type == 'pool':
            V = update_V_pool(Us, observed_entries, observed_indices, ncol, rank)
        elif type == 'wc':
            V = update_V_wc(Us, observed_entries, observed_indices, nrow, ncol, rank)
            
        # 2. Update U (Left step)
        Us = update_U(V, observed_entries, observed_indices, nrow, rank)
        
        # 3. Compute Objective & Check Convergence
        objValue = compute_objective(Us, V, observed_entries, observed_indices, type=type)
        
        if store_history:
            losses.append(objValue)
            
        # Early stopping check
        if optCond(objValue, objPrevious) < optTol:
            print(f"\nOptimality conditions satisfied at iteration {t}.")
            print(f"Final Objective value = {objValue:5.3g}")
            break
            
        objPrevious = objValue
        print(f"Iteration {t}: Objective = {objValue:.5f}", end='\r')

    else:
        # Executes only if the loop doesn't hit the 'break' statement
        print(f"\nWarning: maximum number of iterations ({max_iters}) reached.")
        print(f"Final Objective value = {objValue:5.3g}")

    if store_history:
        return Us, V, losses
    else:
        return Us, V
    

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
#                         OOS Solvers                          #
# ************************************************************ #

def impute_X_pinv(right_factor, observed_indices, Ms_true):
    n_envs = len(Ms_true)
    nrow = Ms_true[0].shape[0]
    rank = right_factor.shape[1]
    Us_hat = [np.zeros((nrow, rank)) for _ in range(n_envs)]
    for i in range(n_envs):
        rows, cols = observed_indices[i]
        for j in range(nrow):
            obs_cols = cols[rows == j]
            R_obs = right_factor[obs_cols]       # (n_obs, rank)
            y_obs = Ms_true[i][j, obs_cols]      # (n_obs,)
            Us_hat[i][j, :] = np.linalg.lstsq(R_obs, y_obs, rcond=None)[0]
    return Us_hat


def get_Mhat_from_right_factor(right_factor, observed_indices_test, Ms_true):
    """
    Get the estimated matrices from the right factor and the test indices.
    The estimated matrices are obtained by using the impute_X_pinv function.
    """
    Us_hat = impute_X_pinv(right_factor, observed_indices_test, Ms_true)
    Ms_hat = [U @ right_factor.T for U in Us_hat]
    return Ms_hat
