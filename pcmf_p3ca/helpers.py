import cvxpy as cp

from itertools import combinations

from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist

from sklearn import mixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score, rand_score
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

from sksparse.cholmod import cholesky_AAt

try:
    from numba import jit, prange
except Exception:
    def jit(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

try:
    from pcmf_p3ca.utils.admm_utils import prox_c, group_soft_threshold
except Exception:
    print("Fallback: compiled prox_c not built; Use numba or Python fallback")
    prox_c = None
    # raise ImportError('admm_utils module not found, so not importing prox_c')

###------- PROXIMAL OPERATORS -------###

def prox_python(V, lamb, rho, w=None):
    """
    Group-lasso proximal operator (pure Python implementation).

    Applies row-wise soft-thresholding with group lasso penalty.

    Args:
        V (ndarray): Input matrix of shape (n_groups, n_features).
        lamb (float): Regularization parameter (λ) for the lasso penalty.
        rho (float): Augmented Lagrangian parameter (ρ) from ADMM.
        w (ndarray, optional): Row-specific weights of shape (n_groups,). 
            Defaults to uniform weights of 1.

    Returns:
        V_prox (ndarray): Output matrix of the same shape as V after applying
            the group-lasso proximal operator row-wise.
    """
    if w is None:
        w = np.ones(V.shape[0])
    V_prox = np.zeros_like(V)
    for i in range(V.shape[0]):
        alpha = w[i]*lamb/rho
        vec_norm = np.linalg.norm(V[i,:])
        if vec_norm > alpha:
            V_prox[i, :] = V[i,:] - alpha*V[i,:]/vec_norm
        else:
            V_prox[i, :] = 0
    return V_prox

@jit(nopython=True, parallel=True, fastmath=True)
def prox_numba(V_prox, V, lamb, rho, w=None):
    """
    Group-lasso proximal operator (Numba-accelerated).

    Performs the same operation as `prox_python` but uses Numba to speed up
    computation with JIT compilation, parallelization, and fast math optimizations.

    Args:
        V_prox (ndarray): Preallocated array of the same shape as V where results
            will be stored (required for Numba in-place computation).
        V (ndarray): Input matrix of shape (n_groups, n_features).
        lamb (float): Regularization parameter (λ) for the lasso penalty.
        rho (float): Augmented Lagrangian parameter (ρ) from ADMM.
        w (ndarray, optional): Row-specific weights of shape (n_groups,). 
            Defaults to uniform weights of 1.

    Returns:
        V_prox (ndarray): Matrix after applying the group-lasso proximal operator 
            row-wise (written in-place).
    """
    if w is None:
        w = np.ones(V.shape[0])
    for i in prange(V.shape[0]):
        if np.isinf(lamb):
            alpha = lamb/rho
        else:
            alpha = w[i]*lamb/rho
        vec_norm = np.linalg.norm(V[i,:])
        if vec_norm > alpha:
            V_prox[i, :] = V[i,:] - alpha*V[i,:]/vec_norm
        else:
            V_prox[i, :] = np.zeros_like(V[i,:])
    return V_prox


###------- CONVEX CLUSTERING SETUP -------###
def get_weights(X, gauss_coef=0.5, neighbors=None):
    """
    Compute pairwise similarity weights between samples using a Gaussian kernel,
    optionally restricted to k-nearest neighbors.

    Args:
        X (ndarray): Data matrix of shape (n_samples, n_features).
        gauss_coef (float): Gaussian kernel coefficient. Higher values result
            in narrower kernels (faster decay with distance).
        neighbors (int or None): If set, restricts weights to sample pairs that
            are within the k-nearest neighbors of each other. If None, computes
            weights for all pairs.

    Returns:
        w (ndarray): Flattened vector of pairwise weights of length n*(n-1)/2,
            ordered consistently with `scipy.spatial.distance.pdist`.
    """
    n = X.shape[0]
    dist_vec = pdist(X) / n
    w = np.exp(-1 * gauss_coef * dist_vec**2)
    if neighbors is not None:
        # Build indicator for pairs that are k-NN neighbors
        nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(X)
        _, indices = nbrs.kneighbors(X)
        comb_list = list(combinations(range(n), 2))
        neighbors_indicator = []
        for comb in comb_list:
            # If either is in other's neighbor list
            if (comb[1] in indices[comb[0]]) or (comb[0] in indices[comb[1]]):
                neighbors_indicator.append(1.0)
            else:
                neighbors_indicator.append(0.0)
        w *= np.array(neighbors_indicator)
    return w

def get_default_D(X):
    """
    Construct a default sparse difference operator for a given data matrix.

    Args:
        X (ndarray): Data matrix of shape (n_samples, n_features).

    Returns:
        D (csr_matrix): Sparse difference matrix encoding all pairwise
            differences across samples, shape (num_pairs, n_samples).
    """
    return sparse_D(X.shape[0], X.shape[1])

def sparse_D(n,p):
    """
    Construct a sparse matrix that encodes all pairwise differences
    between n coefficient vectors of length p.

    Specifically, for a concatenated coefficient vector
    b = [b_1, b_2, ..., b_n], where each b_i has length p,
    the matrix computes elementwise differences between b_i and b_j
    across all unique pairs (i, j).

    Args:
        n (int): Number of coefficient vectors (samples).
        p (int): Length of each coefficient vector (variables).

    Returns:
        D (csr_matrix): Sparse pairwise difference matrix of shape
            (num_pairs, n), where num_pairs = n choose 2.
    """
    comb_list = list(combinations(range(n),2))
    combs_arr = np.array(comb_list)
    num_combs = combs_arr.shape[0]
    data = np.ones_like(combs_arr)
    data[:,1] *= -1
    row = np.repeat(range(num_combs),2)
    col = combs_arr.flatten()
    return csr_matrix((data.flatten(), (row, col)), shape=(num_combs, n))

def compute_cholesky(n, rho, D, reg_scale=1.0):
    """
    Compute a Cholesky factorization for the ADMM update system.

    Args:
        n (int): Number of coefficient vectors.
        rho (float): Augmented Lagrangian parameter (ρ).
        D (csr_matrix or ndarray): Difference operator matrix.
        reg_scale (float): Regularization scaling factor.

    Returns:
        ndarray: Cholesky factor of (sqrt(rho) * D^T) with scaling applied,
        used in ADMM optimization steps.
    """
    return cholesky_AAt(np.sqrt(rho) * D.T, beta=float(reg_scale))

###------- CONVEX CLUSTERING SUBPROBLEMS -------###

def objective_fn_u(X, beta):
    """
    Objective function for PMD/PCMF subproblem without penalty term.

    Args:
        X (ndarray): Data matrix of shape (n_samples, n_features).
        beta (cp.Variable): Optimization variable (concatenated coefficients).

    Returns:
        cp.Expression: Objective expression = -sum(X @ beta).
    """
    mat = loss_fn(X, beta)
    return mat

def objective_fn_u2(X, beta, penalty):
    """
    Objective function for PCMF subproblem with fused lasso penalty.

    Args:
        X (ndarray): Data matrix of shape (n_samples, n_features).
        beta (cp.Variable): Optimization variable (concatenated coefficients).
        penalty (float): Penalty coefficient for the fused lasso term.

    Returns:
        cp.Expression: Objective expression = -sum(X @ beta) + penalty * ||D @ beta||_2.
    """
    D = sparse_D(X.shape[0],1)
    mat = loss_fn(X, beta) + penalty*cp.norm2(D@beta)
    return mat

def loss_fn(X, beta):
    """
    Linear loss function used in PMD/PCMF objectives.

    Args:
        X (ndarray): Data matrix of shape (n_samples, n_features).
        beta (cp.Variable): Optimization variable.

    Returns:
        cp.Expression: Objective term = -sum(X @ beta).
    """
    return -1*cp.sum(cp.matmul(X,beta))

def SVD(X, return_rank=2):
    """
    Compute truncated singular value decomposition.

    Args:
        X (ndarray): Input matrix of shape (m, n).
        return_rank (int): Number of singular components to return.

    Returns:
        U (ndarray): Left singular vectors (m x return_rank).
        s (ndarray): Singular values (length return_rank).
        Vh (ndarray): Right singular vectors (return_rank x n).
    """
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    return U[:, :return_rank], s[:return_rank], Vh[:return_rank, :]

def PCMF_solve_Xhat(L, X, D, G, Z1, U, s, Vh, Z2, rho):
    """
    Solve for X̂ in PCMF update step using precomputed Cholesky factorization.

    Args:
        L (callable): Linear solver function (e.g., from compute_cholesky).
        X (ndarray): Data matrix.
        D (ndarray): Difference operator.
        G (ndarray): Auxiliary variable.
        Z1 (ndarray): Dual variable.
        U, s, Vh (ndarray): SVD components of current iterate.
        Z2 (ndarray): Dual variable.
        rho (float): Augmented Lagrangian parameter.

    Returns:
        ndarray: Updated X̂ after solving the linear system.
    """
    # L: callable solver using Cholesky factorization (applies system matrix inverse).
    rhs = X + rho * (D.T @ (G - Z1)) + rho * ((U * s) @ Vh - Z2)
    return L(rhs)

def PMD_subproblem_u(X_list, num_var, verb=True):
    """
    Solve the u-subproblem in the PMD-like alternating pathwise clustered algorithm
    for one value of lambda, without penalty.

    Args:
        X_list (list of ndarrays): List of data blocks to concatenate horizontally.
        num_var (int): Number of variables in each block.
        verb (bool): If True, print solver output.

    Returns:
        ndarray: Optimal beta vector from the subproblem.
    """
    X = np.hstack(X_list)
    num_obs = int(len(X)/num_var)
    beta = cp.Variable(num_var*num_obs)
    problem = cp.Problem(cp.Minimize(objective_fn_u(X, beta)), [cp.norm2(beta)**2 <= 1])

    mosek_params = {}
    problem.solve(solver='MOSEK',verbose=verb, warm_start=True, mosek_params=mosek_params)
    return beta.value

def PCMF_subproblem_u(X_list, num_var, penalty, verb=True):
    """
    Solve the u-subproblem in the PCMF alternating pathwise clustered algorithm
    for one value of lambda, with penalty.

    Args:
        X_list (list of ndarrays): List of data blocks to concatenate horizontally.
        num_var (int): Number of variables in each block.
        penalty (float): Penalty parameter for fused lasso.
        verb (bool): If True, print solver output.

    Returns:
        ndarray: Optimal beta vector from the subproblem.
    """
    X = np.hstack(X_list)
    num_obs = int(len(X)/num_var)
    beta = cp.Variable(num_var*num_obs)
    problem = cp.Problem(cp.Minimize(objective_fn_u2(X, beta, penalty)), [cp.norm2(beta)**2 <= 1])

    mosek_params = {}
    problem.solve(solver='MOSEK',verbose=verb, warm_start=True, mosek_params=mosek_params)
    return beta.value

def l2_ball_proj(X):
    """
    Project each row of X onto the unit ℓ2-ball.

    Args:
        X (ndarray): Input matrix of shape (n_samples, n_features).

    Returns:
        ndarray: Matrix with each row projected to satisfy ||X_i||_2 <= 1.
    """
    def row_norm(vec): 
        norm = np.sqrt(np.sum(vec**2))
        # return np.sqrt(1./np.sum(vec**2)) * vec  # Always normalizes
        return vec if norm <= 1 else vec / norm    # Projects only if not inside ball
    return np.apply_along_axis(row_norm, 1, X)

def P3CA_update(
    mat_tilde, 
    chol_solve,    # callable for linear solve, e.g., from compute_cholesky
    D, 
    W, 
    Z, 
    rho, 
    penalty, 
    data, 
    weights=None, 
    neighbors=None, 
    gauss_coef=0.5, 
    prox_method="numba" # "numba", "python", or "c"
):
    """
    Single ADMM update step for P3CA (for U or V update).

    Parameters
    ----------
    mat_tilde : ndarray
        Current 'tilde' matrix for the update (shape: n_samples x n_features).
    chol_solve : callable
        Function for solving the linear system (e.g., from compute_cholesky).
    D : ndarray
        Difference matrix.
    W : ndarray
        Dual variable.
    Z : ndarray
        Dual variable.
    rho : float
        Augmented Lagrangian parameter.
    penalty : float
        Penalty parameter for clustering.
    data : ndarray
        Data matrix (used for weight calculation).
    weights : ndarray, None, or bool
        Precomputed weights or True/False. If None, auto-compute.
    neighbors : int or None
        Number of neighbors (if using kNN-graph weights).
    gauss_coef : float
        Gaussian kernel coefficient.
    prox_method : {"numba", "python", "c"}
        Proximal operator implementation.

    Returns
    -------
    V : ndarray
        Updated primal variable.
    W : ndarray
        Updated auxiliary variable.
    Z : ndarray
        Updated dual variable.
    """

    # Basic shape checks
    n = data.shape[0]
    if D.shape[1] != n:
        raise ValueError(f"D has {D.shape[1]} cols but data has {n} rows.")
    if W.shape != Z.shape:
        raise ValueError(f"W and Z must have same shape; got {W.shape} vs {Z.shape}.")

    # Compute weights if needed
    if isinstance(weights, np.ndarray):
        w = weights
    elif weights is False or weights is None:
        w = get_weights(data, gauss_coef=0.0)
    else:
        w = get_weights(data, gauss_coef=gauss_coef, neighbors=neighbors)

    # Linear system update for V
    V = chol_solve(mat_tilde + rho * (D.T @ (W - Z)))

    # Proximal update for W
    if prox_method == "numba":
        W = prox_numba(np.empty_like(W), D @ V + Z, penalty, rho, w)
    elif prox_method == "python":
        W = prox_python(D @ V + Z, penalty, rho, w)
    elif prox_method == "c":
        W = prox_c(D @ V + Z, penalty, rho, w)
    else:
        raise ValueError(f"prox_method must be 'numba', 'python', or 'c', not '{prox_method}'")

    # Dual update
    Z = Z + D @ V - W
    
    # Project onto ℓ2 ball
    V = l2_ball_proj(V)

    return V, W, Z

###------- Path selection functions -------###

def calculate_scores(pred_clusters, true_clusters):
    """
    Calculate clustering evaluation metrics by comparing predicted clusters
    against ground-truth cluster labels.

    Args:
        pred_clusters (ndarray): Predicted cluster assignments of shape (n_samples,).
        true_clusters (ndarray): Ground-truth cluster labels of shape (n_samples,).

    Returns:
        nmi_score (float): Normalized Mutual Information (NMI) score.
        adj_rand_score (float): Adjusted Rand Index (ARI).
        ri_score (float): Rand Index (RI).
        mse_score (float): Mean Squared Error (MSE) between predicted and true labels.
    """
    nmi_score = NMI_score(true_clusters, pred_clusters, average_method='arithmetic')
    adj_rand_score = ARI_score(true_clusters, pred_clusters)
    ri_score = rand_score(true_clusters, pred_clusters)
    mse_score = mean_squared_error(true_clusters, pred_clusters)

    return nmi_score, adj_rand_score, ri_score, mse_score

def confusion_matrix_ordered(pred, true):
    """
    Compute an ordered confusion matrix that aligns predicted clusters with
    true labels using the Hungarian assignment algorithm.

    Args:
        pred (ndarray): Predicted cluster labels of shape (n_samples,).
        true (ndarray): Ground-truth cluster labels of shape (n_samples,).

    Returns:
        conf_mat_ord (ndarray): Confusion matrix with permuted columns
            to maximize alignment between predicted and true labels.
    """
    def _make_cost_m(cm):
        """
        Convert a confusion matrix into a cost matrix for assignment.

        Args:
            cm (ndarray): Confusion matrix.

        Returns:
            ndarray: Cost matrix derived from the confusion matrix,
            where larger values in `cm` are transformed into smaller
            costs (by subtracting from the maximum value).
            Suitable for use with the Hungarian algorithm.
        """
        s = np.max(cm)
        return (- cm + s)
    conf_mat = confusion_matrix(pred,true)
    indexes = linear_assignment(_make_cost_m(conf_mat))
    js = [e for e in sorted(indexes, key=lambda x: x[0])[1]]
    conf_mat_ord = conf_mat[:, js]

    return conf_mat_ord

def fit_spectral(X, true_clusters, n_clusters):
    """
    Perform spectral clustering and evaluate results against ground truth.

    Args:
        X (ndarray): Data matrix of shape (n_samples, n_features).
        true_clusters (ndarray): Ground-truth cluster labels of shape (n_samples,).
        n_clusters (int): Number of clusters to find.

    Returns:
        labels (ndarray): Predicted cluster assignments.
        ari (float): Adjusted Rand Index score.
        nmi (float): Normalized Mutual Information score.
        acc (float): Clustering accuracy after optimal label alignment.
    """
    #
    data_in = X
    #
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, random_state=20, affinity="nearest_neighbors").fit(data_in)
    labels = spectral_clustering.labels_
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

###------- PLOTTING FUNCTIONS -------###

def path_plot(coefficient_arr, penalty_list, plot_range=[0,-1],cut_vars=False):
    """
    Plot the evolution of coefficient paths along a sequence of penalties (λ).

    Args:
        coefficient_arr (ndarray): Array of coefficients across penalties,
            shape (n_penalties, n_observations, n_variables).
        penalty_list (list or ndarray): Sequence of penalty values (λ).
        plot_range (list of int): Subset of penalties to display, specified
            as [start, end]. Default = [0, -1] (entire range).
        cut_vars (bool): If True, restrict plot to selected variables
            (first, second, and last).

    Returns:
        ax (matplotlib.axes.Axes): The matplotlib axis object containing the plot.
    """

    # Restrict to a subset of penalties (e.g., skip early initialization phase region)
    coefficient_arr = coefficient_arr[plot_range[0]:plot_range[1],:,:]

    # Define index range for plotting along the penalty path
    penalty_list = penalty_list[plot_range[0]:plot_range[1]]

    # Optionally reduce to a representative subset of variables
    if cut_vars:
        n_vars = coefficient_arr.shape[2]
        if n_vars >= 3:
            # pick first, middle, last as a reasonable default
            mid = n_vars // 2
            keep_cols = [0, mid, n_vars - 1]
        elif n_vars == 2:
            keep_cols = [0, 1]
        else:  # n_vars == 1
            keep_cols = [0]
        coefficient_arr = coefficient_arr[:, :, keep_cols]

    # Colormap
    cmap = cm.get_cmap('viridis', coefficient_arr.shape[2])
    colors = cmap(np.linspace(0.0,1.0,coefficient_arr.shape[2]))

    # Define X-axis range
    penalty_range = range(len(penalty_list))

    # Create figure and axis
    fig, ax = plt.subplots(1,1, figsize=(20,10))

    # Plot coefficient trajectories for each variable
    for i in range(coefficient_arr.shape[2]):
        x = np.round(np.array(penalty_list),8)[penalty_range]
        y = coefficient_arr[penalty_range,:,i]
        ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.15)

        # Format x-axis ticks with rotated penalty values
        ax.set_xticks(range(x.shape[0]), minor=False);
        plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right')
        ax.set_xticklabels(x,fontsize=24)

    # Hide every other x-axis tick label for readability
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    # Format y-axis tick labels
    ax.tick_params(axis='y', labelsize=24)
    # Axis labels
    plt.xlabel(r'$\lambda$',fontsize=24)
    plt.ylabel('Coefficients',fontsize=24)

    # Replace tick labels: use ∞ for infinite penalties
    ax.set_xticklabels([r'$\infty$' if np.isinf(v) else f'{v:.4f}' for v in penalty_list],
               rotation=70, ha='right')

    return ax

