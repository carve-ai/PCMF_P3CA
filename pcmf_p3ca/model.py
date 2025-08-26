from matplotlib import cm
import matplotlib.pyplot as plt

import logging
import numpy as np
from seaborn import despine
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

from .helpers import (
    P3CA_update,
    PMD_subproblem_u,
    PCMF_subproblem_u,
    compute_cholesky,
    fit_spectral,
    get_default_D,
    get_weights,
    l2_ball_proj,
    prox_numba,
    prox_python,
    path_plot,
    PCMF_solve_Xhat,
    SVD
    )

try:
    from pcmf_p3ca.utils.admm_utils import prox_c, group_soft_threshold
except Exception:
    print("Fallback: compiled prox_c not built; Use numba or Python fallback")
    prox_c = None
    # raise ImportError('admm_utils module not found, so not importing prox_c')

class PCMF:
    """
    Fully Constrained Pathwise Clustered Matrix Factorization (PCMF).

    Solves the PCMF problem for a sequence of penalties using ADMM updates,
    as described in Algorithm 1 from Buch et al., AISTATS 2024.

    Args:
        penalty_list (list or array): Sequence of penalty (lambda) values for the path.
        problem_rank (int): Matrix factorization rank.
        rho (float): Augmented Lagrangian parameter.
        admm_iters (int): Number of ADMM updates per lambda value.
        gauss_coef (float): Gaussian coefficient for weights.
        neighbors (int or None): Number of nearest neighbors for graph construction.
        weights (ndarray or None): Predefined pairwise weights; if None, they are auto-generated.
        prox_method (str): Which proximal operator to use ('numba', 'prox_c', or 'python').
        progress_mode (str): Progress reporting style ('auto', 'bar', 'file', or 'none').
        output_file (str or None): File to log progress when progress_mode='file'.

    Attributes:
        Xhat_list (list): Reconstructed estimates along the path.
        G_list (list): ADMM dual variable G along the path.
        U_list (list): Left singular vectors across the path.
        S_list (list): Singular values across the path.
        V_list (list): Right singular vectors across the path.
    """

    def __init__(
        self,
        penalty_list,
        problem_rank=1,
        rho=1.0,
        admm_iters=5,
        gauss_coef=2.0,
        neighbors=None,
        weights=None,
        prox_method="numba",
        progress_mode="auto",
        output_file=None,
    ):
        self.penalty_list = penalty_list
        self.problem_rank = problem_rank
        self.rho = rho
        self.admm_iters = admm_iters
        self.gauss_coef = gauss_coef
        self.neighbors = neighbors
        self.weights = weights
        self.prox_method = prox_method
        self.progress_mode = progress_mode
        self.output_file = output_file

        self.Xhat_list = []
        self.G_list = []
        self.U_list = []
        self.S_list = []
        self.V_list = []
        print('initializing PCMF model')

    def _generate_weights(self, X):
        """
        Generate pairwise weights for clustering using a Gaussian kernel.

        Args:
            X (ndarray): Input data matrix of shape (n_samples, n_features).

        Returns:
            ndarray: Flattened pairwise weights in condensed pdist order.
        """
        return get_weights(X, gauss_coef=self.gauss_coef, neighbors=self.neighbors)

    def _progress_iterator(self, iterable, desc="Lambda path"):
        """
        Wrap an iterable with a progress indicator (bar, file logging, or none).

        Args:
            iterable (iterable): Items to iterate over.
            desc (str): Description label for the progress reporting.

        Returns:
            iterator: Wrapped iterator with progress reporting applied.
        """
        mode = self.progress_mode
        if mode == "auto":
            mode = "bar" if self.output_file is None else "file"
        if mode == "bar":
            return tqdm(iterable, desc=desc)
        elif mode == "file":
            def iterator():
                total = len(iterable)
                for i, item in enumerate(iterable):
                    try:
                        with open(self.output_file, "a") as f:
                            f.write(f"{desc} {i+1}/{total}\n")
                    except Exception as e:
                        print(f"[progress_mode=file] Failed to write to {self.output_file!r}: {e}")
                        print("→ Switching progress reporting to 'off'.")
                        # fallback to no progress reporting
                        for item2 in iterable[i:]:
                            yield item2
                        return
                    yield item
            return iterator()
        else:
            return iterable

    def fit(self, X, true_clusters, weights=None):
        """
        Algorithm 1: Fit the PCMF model along a penalty path.

        Args:
            X (ndarray): Input data matrix of shape (n_samples, n_features).
            true_clusters (ndarray): Ground-truth cluster labels of shape (n_samples,).
            weights (ndarray or None, optional): Precomputed weights; if None, generated internally.

        Returns:
            dict: Results with keys {'Xhat','G','U','S','V'}, each a list across the penalty path.
        """
        # --- Input validation ---
        assert X.ndim == 2, "X must be a 2D matrix"
        assert len(self.penalty_list) > 0, "Penalty list must be non-empty"

        # --- Weights setup ---
        if weights is not None:
            self.weights = weights
            logging.info("Using weights provided to fit().")
        elif self.weights is not None:
            logging.info("Using weights set at initialization.")
        else:
            logging.info("Generating weights internally using _generate_weights().")
            self.weights = self._generate_weights(X)

        self.true_clusters = true_clusters

        # --- Initialization (Algorithm 1, lines 1-2) ---
        self.Xhat_list.clear()
        self.G_list.clear()
        self.U_list.clear()
        self.S_list.clear()
        self.V_list.clear()

        print('running PCMF model')

        D = get_default_D(X)
        G = D @ X
        Z1 = D @ X
        Xhat = X.copy()
        Z2 = X.copy()
        U, s, Vh = SVD(X, self.problem_rank)

        N, p = X.shape
        L = compute_cholesky(N, self.rho, D, reg_scale=1.0+self.rho)

        # --- Pathwise solution (Algorithm 1, lines 2-12) ---
        for lambda_ in self._progress_iterator(self.penalty_list, desc="Lambda path"):
            for k in range(self.admm_iters):
                # Step 4: Xhat update
                Xhat = PCMF_solve_Xhat(L, X, D, G, Z1, U, s, Vh, Z2, self.rho)
                # Step 5: G update
                if self.prox_method == "numba":
                    G = prox_numba(np.zeros_like(G), D @ Xhat + Z1, lambda_, self.rho, self.weights)
                elif self.prox_method == "c":
                    G = prox_c(D @ Xhat + Z1, lambda_, self.rho, self.weights)
                else:
                    G = prox_python(D @ Xhat + Z1, lambda_, self.rho, self.weights)

                # Step 6: SVD update
                U, s, Vh = SVD(Xhat + Z2, self.problem_rank)

                # Step 7-8: Dual updates
                Z1 = Z1 + self.rho * (D @ Xhat - G)
                Z2 = Z2 + self.rho * (Xhat - (U * s) @ Vh)

            # Store results
            self.Xhat_list.append(Xhat.copy())
            self.G_list.append(G.copy())
            self.U_list.append(U.copy())
            self.S_list.append(s.copy())
            self.V_list.append(Vh.copy())

        return {
            'Xhat': self.Xhat_list,
            'G': self.G_list,
            'U': self.U_list,
            'S': self.S_list,
            'V': self.V_list
        }

    def return_fit(self, lam=None):
        """
        Return fitted variables for a given penalty index or for the whole path.

        Args:
            lam (int or str or None): If int, return results at that index.
                If "best", return results at the selected best lambda.
                If None, return the entire path as a dict.

        Returns:
            tuple or dict: (Xhat, G, U, S, V) for a specific lambda, or a dict of lists for all lambdas.
        """
        if isinstance(lam, int) and lam in range(len(self.penalty_list)):
            return (
                self.Xhat_list[lam],
                self.G_list[lam],
                self.U_list[lam],
                self.S_list[lam],
                self.V_list[lam]
            )
        elif lam == "best":
            if hasattr(self, "best_lam"):
                return (
                    self.Xhat_list[self.best_lam],
                    self.G_list[self.best_lam],
                    self.U_list[self.best_lam],
                    self.S_list[self.best_lam],
                    self.V_list[self.best_lam]
                )
            else:
                raise ValueError("Run .select() first to compute and set best lambda.")
        else:
            # Return the entire path as dictionary.
            return {
                'Xhat': self.Xhat_list,
                'G': self.G_list,
                'U': self.U_list,
                'S': self.S_list,
                'V': self.V_list
            }

    def select(self, labels_true=None, trim_init_lambdas=2):
        """
        Estimate clusters along the path using spectral clustering on differences
        derived from dual variables of PCMF model fit.

        Args:
            labels_true (ndarray or None, optional): Ground-truth labels; if None, uses self.true_clusters.
            trim_init_lambdas (int, optional): Number of initial penalties to skip (initialization phase).

        Returns:
            list: Predicted cluster labels for each penalty value along the path.
        """
        print('Estimating clusters along PCMF path.')
        if labels_true is None:
            labels_true = self.true_clusters

        ari_list=[]
        nmi_list=[]
        acc_list=[]
        labels_list = []
        for p in np.arange(0,trim_init_lambdas): # 
            labels_list.append(None)
            ari_list.append(0)
            nmi_list.append(0)
            acc_list.append(0)
        for p in np.arange(trim_init_lambdas,len(self.penalty_list)): # 
            labels, ari, nmi, acc = fit_spectral(np.array(self.Xhat_list)[p,:,:], labels_true, len(np.unique(labels_true)))
            labels_list.append(labels)
            ari_list.append(ari)
            nmi_list.append(nmi)
            acc_list.append(acc)

        self.path_labels_list = labels_list
        self.path_ari_list = ari_list
        self.path_nmi_list = nmi_list
        self.path_acc_list = acc_list
        self.best_lam = int(np.argmax(np.array(ari_list)))

        print(
            f"Best Penalty IDX: {int(np.argmax(np.array(ari_list)))}, "
            f"ARI: {float(np.max(np.array(ari_list)))}, "
            f"NMI: {float(np.max(np.array(nmi_list)))}"
        )

        return labels_list

    def return_labels(self, lam=None):
        """
        Return predicted labels at a given penalty index, the best index, or all.

        Args:
            lam (int or str or None): If int, labels at that index; if "best", labels at best lambda;
                if None, labels for all lambdas.

        Returns:
            ndarray or list: Labels for the specified lambda or a list across the path.
        """
        if lam is not None and isinstance(lam, int) and lam in range(len(self.penalty_list)):
            return self.path_labels_list[lam]
        elif lam == "best":
            return self.path_labels_list[self.best_lam]
        else:
            return self.path_labels_list

    def plot_path(self, plot_range=None):
        """
        Plot reconstructed estimates along the penalty path.

        Args:
            plot_range (tuple[int, int], optional):
                A (start, end) index slice over penalty values.
                Use -1 for the end index to indicate "until the final penalty".
                Defaults to (0, -1).

        Returns:
            None
        """
        if plot_range is None:
            plot_range = (0, -1)

        # Plot estimates along path for first three variables
        ax = path_plot(np.asarray(self.Xhat_list)[:,:,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel(r"$\hat{X}$ estimates")
        plt.locator_params(nbins=25)

        # Plot estimates along path separated by true cluster label for first three variables
        path_plot(np.asarray(self.Xhat_list)[:, self.true_clusters==0,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel(r"$\hat{X}$ estimates")
        plt.locator_params(nbins=25)

        path_plot(np.asarray(self.Xhat_list)[:, self.true_clusters==1,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel(r"$\hat{X}$ estimates")
        plt.locator_params(nbins=25)


class LLPCMF:
    """
    Locally Linear Pathwise Clustered Matrix Factorization (LL-PCMF).

    Alternates convex clustering (via MOSEK) for `u` updates with ADMM for `V`
    updates. Provides locally linear constraints for pathwise clustering.
    As described in Algorithm from Buch et al., AISTATS 2024.

    Args:
        penalty_list (list or array): Sequence of penalty (lambda) values for the path.
        rho (float): Augmented Lagrangian parameter.
        admm_iters (int): Number of ADMM updates per lambda value.
        gauss_coef (float): Gaussian coefficient for weights.
        neighbors (int or None): Number of nearest neighbors for graph construction.
        weights (ndarray or None): Predefined pairwise weights; if None, they are auto-generated.
        non_negative (bool): If True, enforce non-negativity on factor matrices.
        prox_method (str): Which proximal operator to use ('numba', 'prox_c', or 'python').
        progress_mode (str): Progress reporting style ('auto', 'bar', 'file', or 'none').
        output_file (str or None): File to log progress when progress_mode='file'.

    Attributes:
        u_list (list): Learned `u` vectors along the path.
        s_list (list): Component scaling values along the path.
        V_list (list): Learned `V` matrices along the path.
    """

    def __init__(
        self,
        penalty_list,
        rho=1.0,
        admm_iters=5,
        gauss_coef=0.5,
        neighbors=None,
        weights=None,
        non_negative=False,
        prox_method="numba",
        progress_mode="auto",
        output_file=None,
    ):
        self.penalty_list = penalty_list
        self.rho = rho
        self.admm_iters = admm_iters
        self.gauss_coef = gauss_coef
        self.neighbors = neighbors
        self.weights = weights
        self.non_negative = non_negative
        self.prox_method = prox_method
        self.progress_mode = progress_mode
        self.output_file = output_file

        self.u_list = []
        self.s_list = []
        self.V_list = []

    def _generate_weights(self, X):
        """
        Generate pairwise weights for clustering using a Gaussian kernel.

        Args:
            X (ndarray): Input data matrix of shape (n_samples, n_features).

        Returns:
            ndarray: Flattened pairwise weights in condensed pdist order.
        """
        return get_weights(X, gauss_coef=self.gauss_coef, neighbors=self.neighbors)

    def _progress_iterator(self, iterable, desc="Lambda path"):
        """
        Wrap an iterable with a progress indicator (bar, file logging, or none).

        Args:
            iterable (iterable): Items to iterate over.
            desc (str): Description label for the progress reporting.

        Returns:
            iterator: Wrapped iterator with progress reporting applied.
        """
        mode = self.progress_mode
        if mode == "auto":
            mode = "bar" if self.output_file is None else "file"
        if mode == "bar":
            return tqdm(iterable, desc=desc)
        elif mode == "file":
            def iterator():
                total = len(iterable)
                for i, item in enumerate(iterable):
                    try:
                        with open(self.output_file, "a") as f:
                            f.write(f"{desc} {i+1}/{total}\n")
                    except Exception as e:
                        print(f"[progress_mode=file] Failed to write to {self.output_file!r}: {e}")
                        print("→ Switching progress reporting to 'off'.")
                        # fallback to no progress reporting
                        for item2 in iterable[i:]:
                            yield item2
                        return
                    yield item
            return iterator()
        else:
            return iterable

    def fit(self, X, true_clusters, weights=None, verb=False):
        """
        Fit the LL-PCMF model along a penalty path (convex clustering for u, ADMM for V).

        Args:
            X (ndarray): Input data matrix of shape (n_samples, n_features).
            true_clusters (ndarray): Ground-truth cluster labels of shape (n_samples,).
            weights (ndarray or None, optional): Precomputed weights; if None, generated internally.
            verb (bool, optional): Whether to print MOSEK solver output.

        Returns:
            dict: Results with keys {'u','s','V'}, each a list across the penalty path.
        """
        assert X.ndim == 2, "X must be a 2D matrix"
        assert len(self.penalty_list) > 0, "Penalty list must be non-empty"

        if weights is not None:
            self.weights = weights
        elif self.weights is not None:
            pass
        else:
            self.weights = self._generate_weights(X)

        self.true_clusters = true_clusters

        self.V_list.clear()
        self.u_list.clear()
        self.s_list.clear()

        N, p = X.shape
        D = get_default_D(X)
        chol_factor = compute_cholesky(N, self.rho, D, reg_scale=1.0)

        # --- Initial V (mean rows), no duals yet
        V = np.tile(np.mean(X, axis=0), (N, 1))

        # --- Initial u (PMD convex clustering on <x,v>)
        Xu_tildes = np.sum(X * V, axis=1)
        u = PMD_subproblem_u([Xu_tildes], 1, verb=verb)
        u = u.reshape((N, 1))

        for lam in self._progress_iterator(self.penalty_list, desc="Lambda path"):
            # --- V update via ADMM (with duals)
            W = D @ V
            Z = D @ V
            Xv_tildes = (u * X) if u.ndim == 2 else (u[:, None] * X)
            for _ in range(self.admm_iters):
                rhs = Xv_tildes + self.rho * (D.T @ (W - Z))
                V = chol_factor(rhs)
                prox_arg = D @ V + Z
                if self.prox_method == "numba":
                    W = prox_numba(np.zeros_like(W), prox_arg, lam, self.rho, self.weights)
                elif self.prox_method == "c":
                    W = prox_c(prox_arg, lam, self.rho, self.weights)
                else:
                    W = prox_python(prox_arg, lam, self.rho, self.weights)
                Z = Z + D @ V - W
            V = l2_ball_proj(V)
            if self.non_negative:
                V[V < 0] = 0
            self.V_list.append(V.copy())

            # --- u update: convex clustering subproblem on <x,v>
            Xu_tildes = np.sum(X * V, axis=1)
            try:
                u = PCMF_subproblem_u([Xu_tildes], 1, lam, verb=verb)
            except Exception:
                u = PMD_subproblem_u([Xu_tildes], 1, verb=verb)
            u = u.reshape((N, 1))
            self.u_list.append(u.copy())

            # --- s update (component scaling)
            s = u.flatten() * np.sum(X * V, axis=1)
            self.s_list.append(s.copy())

        return {'u': self.u_list, 's': self.s_list, 'V': self.V_list}

    def return_fit(self, lam=None):
        """
        Return fitted variables for a given penalty index or for the whole path.

        Args:
            lam (int or str or None): If int, return results at that index.
                If "best", return results at the selected best lambda.
                If None, return the entire path.

        Returns:
            tuple or tuple[list]: (u, s, V) for a specific lambda, or the full path as lists.
        """
        if isinstance(lam, int) and lam in range(len(self.penalty_list)):
            return (
                self.u_list[lam],
                self.s_list[lam],
                self.V_list[lam]
            )
        elif lam == "best":
            if hasattr(self, "best_lam"):
                return (
                    self.u_list[self.best_lam],
                    self.s_list[self.best_lam],
                    self.V_list[self.best_lam]
                )
            else:
                raise ValueError("Run .select() first to compute and set best lambda.")
        else:
            # Return the entire path as tuple of lists.
            return (
                self.u_list,
                self.s_list,
                self.V_list
            )

    def select(self, labels_true=None, trim_init_lambdas=2):
        """
        Estimate clusters along the path using spectral clustering on reconstructed estimates.

        Args:
            labels_true (ndarray or None, optional): Ground-truth labels; if None, uses self.true_clusters.
            trim_init_lambdas (int, optional): Number of initial penalties to skip (initialization phase).

        Returns:
            list: Predicted cluster labels for each penalty value along the path.
        """
        print('Estimating clusters along LL-PCMF path.')
        if labels_true is None:
            labels_true = self.true_clusters

        # Reconstruct Xhat estimates along path
        Xhat_list = []
        for i in range(len(self.penalty_list)):
            Xhat_list.append(self.u_list[i] * self.V_list[i])

        self.Xhat_list = Xhat_list

        ari_list=[]
        nmi_list=[]
        acc_list=[]
        labels_list = []
        for p in np.arange(0,trim_init_lambdas): # 
            labels_list.append(None)
            ari_list.append(0)
            nmi_list.append(0)
            acc_list.append(0)
        for p in np.arange(trim_init_lambdas,len(self.penalty_list)): # 
            labels, ari, nmi, acc = fit_spectral(np.array(self.Xhat_list)[p,:,:], labels_true, len(np.unique(labels_true)))
            labels_list.append(labels)
            ari_list.append(ari)
            nmi_list.append(nmi)
            acc_list.append(acc)

        self.path_labels_list = labels_list
        self.path_ari_list = ari_list
        self.path_nmi_list = nmi_list
        self.path_acc_list = acc_list
        self.best_lam = int(np.argmax(np.array(ari_list)))

        print(
            f"Best Penalty IDX: {int(np.argmax(np.array(ari_list)))}, "
            f"ARI: {float(np.max(np.array(ari_list)))}, "
            f"NMI: {float(np.max(np.array(nmi_list)))}"
        )

        return labels_list

    def return_labels(self, lam=None):
        """
        Return predicted labels at a given penalty index, the best index, or all.

        Args:
            lam (int or str or None): If int, labels at that index; if "best", labels at best lambda;
                if None, labels for all lambdas.

        Returns:
            ndarray or list: Labels for the specified lambda or a list across the path.
        """
        if lam is not None and isinstance(lam, int) and lam in range(len(self.penalty_list)):
            return self.path_labels_list[lam]
        elif lam == "best":
            return self.path_labels_list[self.best_lam]
        else:
            return self.path_labels_list

    def plot_path(self, plot_range=None):
        """
        Plot reconstructed estimates along the penalty path.

        Args:
            plot_range (tuple[int, int], optional):
                A (start, end) index slice over penalty values.
                Use -1 for the end index to indicate "until the final penalty".
                Defaults to (0, -1).

        Returns:
            None
        """
        if plot_range is None: 
            plot_range = (0, -1)

        # Plot estimates along path for first three variables
        ax = path_plot(np.asarray(self.Xhat_list)[:,:,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel(r"$\hat{X}$ estimates")
        plt.locator_params(nbins=25)

        # Plot estimates along path separated by true cluster label for first three variables
        path_plot(np.asarray(self.Xhat_list)[:, self.true_clusters==0,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel(r"$\hat{X}$ estimates")
        plt.locator_params(nbins=25)

        path_plot(np.asarray(self.Xhat_list)[:, self.true_clusters==1,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel(r"$\hat{X}$ estimates")
        plt.locator_params(nbins=25)



class ConsensusPCMF:
    """
    Consensus Pathwise Clustered Matrix Factorization (Consensus PCMF).

    Implements Appendix Algorithm 2 from Buch et al., AISTATS 2024.
    Solves PCMF across multiple data batches with consensus constraints.

    Args:
        penalty_list (list or array): Sequence of penalty (lambda) values for the path.
        problem_rank (int): Matrix factorization rank (r).
        rho (float): Augmented Lagrangian parameter.
        admm_iters (int): Number of ADMM updates per lambda value.
        gauss_coef (float): Gaussian coefficient for weights.
        neighbors (int or None): Number of nearest neighbors for graph construction.
        weights (ndarray or None): Predefined pairwise weights; if None, auto-generated.
        numba (bool): If True, use numba-accelerated proximal update.
        prox_method (str): Which proximal operator to use ('numba', 'prox_c', or 'python').
        progress_mode (str): Progress reporting style ('auto', 'bar', 'file', or 'none').
        output_file (str or None): File to log progress when progress_mode='file'.
        split_size (int): Number of samples per batch (controls data splitting).

    Attributes:
        Xhat_list (list): Reconstructed estimates along the path.
        G_list (list): ADMM dual variable G along the path.
        U_list (list): Left singular vectors across the path.
        S_list (list): Singular values across the path.
        V_list (list): Right singular vectors across the path.
    """

    def __init__(
        self,
        penalty_list,
        problem_rank=1,
        rho=1.0,
        admm_iters=5,
        gauss_coef=2.0,
        neighbors=None,
        weights=None,
        numba=True,
        prox_method='numba',
        progress_mode="auto",
        output_file=None,
        split_size=10,
    ):
        self.penalty_list = penalty_list
        self.problem_rank = problem_rank
        self.rho = rho
        self.admm_iters = admm_iters
        self.gauss_coef = gauss_coef
        self.neighbors = neighbors
        self.weights = weights
        self.numba = numba
        self.prox_method = prox_method
        self.progress_mode = progress_mode
        self.output_file = output_file
        self.split_size = split_size

        self.Xhat_list, self.G_list, self.U_list, self.S_list, self.V_list = [], [], [], [], []

    def _generate_weights(self, X):
        """
        Generate pairwise weights for clustering using a Gaussian kernel.

        Args:
            X (ndarray): Input data matrix of shape (n_samples, n_features).

        Returns:
            ndarray: Flattened pairwise weights in condensed pdist order.
        """
        return get_weights(X, gauss_coef=self.gauss_coef, neighbors=self.neighbors)

    def _progress_iterator(self, iterable, desc="Lambda path"):
        """
        Wrap an iterable with a progress indicator (bar, file logging, or none).

        Args:
            iterable (iterable): Items to iterate over.
            desc (str): Description label for the progress reporting.

        Returns:
            iterator: Wrapped iterator with progress reporting applied.
        """
        mode = self.progress_mode
        if mode == "auto":
            mode = "bar" if self.output_file is None else "file"
        if mode == "bar":
            return tqdm(iterable, desc=desc)
        elif mode == "file":
            def iterator():
                total = len(iterable)
                for i, item in enumerate(iterable):
                    try:
                        with open(self.output_file, "a") as f:
                            f.write(f"{desc} {i+1}/{total}\n")
                    except Exception as e:
                        print(f"[progress_mode=file] Failed to write to {self.output_file!r}: {e}")
                        print("→ Switching progress reporting to 'off'.")
                        # fallback to no progress reporting
                        for item2 in iterable[i:]:
                            yield item2
                        return
                    yield item
            return iterator()
        else:
            return iterable

    def fit(self, X, true_clusters, weights=None):
        """
        Fit the Consensus PCMF model along a penalty path (batched ADMM).

        Args:
            X (ndarray): Input data matrix of shape (n_samples, n_features).
            true_clusters (ndarray): Ground-truth cluster labels of shape (n_samples,).
            weights (ndarray or None, optional): Precomputed weights; if None, generated per batch.

        Returns:
            dict: Results with keys {'Xhat','G','U','S','V'}, each a list across the penalty path.
        """
        self.true_clusters = true_clusters

        # Algorithm 2 (lines 1-2): Split X into B batches and demean
        X_batches = np.array_split(X, max(1, int(X.shape[0] / self.split_size)), axis=0)
        B = len(X_batches)
        X_means = [Xb.mean(axis=0) for Xb in X_batches]
        X_batches = [Xb - mu for Xb, mu in zip(X_batches, X_means)]
        X_mean = np.vstack([np.tile(mu, (Xb.shape[0], 1)) for Xb, mu in zip(X_batches, X_means)])

        # Line 3: Cholesky decomposition
        N, p = X_batches[0].shape
        D = get_default_D(X_batches[0])
        chol_factor = compute_cholesky(N, self.rho, D, reg_scale=1.0+self.rho)

        # Lines 4-8: Initialization
        Gb = [D @ Xb for Xb in X_batches]
        Z1b = [G.copy() for G in Gb]
        Xhatb = [Xb.copy() for Xb in X_batches]
        Z2b = [Xb.copy() for Xb in X_batches]

        U, S, Vh = randomized_svd(np.vstack(Xhatb), n_components=self.problem_rank)

        # Progress iterator over penalty list
        for lambda_ in self._progress_iterator(self.penalty_list, desc="Lambda path"):
            for k in range(self.admm_iters):  # Algorithm 2: line 10
                # Batch ADMM update (lines 11-14)
                for b in range(B):
                    Xb, Z1, Z2, G = X_batches[b], Z1b[b], Z2b[b], Gb[b]
                    Ub = U[b*Xb.shape[0]:(b+1)*Xb.shape[0]]
                    # Line 12: Xhat update
                    Xhatb[b] = chol_factor(
                        Xb + self.rho * D.T @ (G - Z1)
                        + self.rho * (Ub @ np.diag(S) @ Vh - Z2)
                    )
                    # Line 13: G update with selected prox method
                    prox_arg = D @ Xhatb[b] + Z1
                    if self.prox_method == 'numba':
                        Gb[b] = prox_numba(np.zeros_like(G), prox_arg, lambda_, self.rho, weights or self._generate_weights(Xb))
                    elif self.prox_method == 'prox_c':
                        Gb[b] = prox_c(prox_arg, lambda_, self.rho, weights or self._generate_weights(Xb))
                    else:
                        Gb[b] = prox_python(prox_arg, lambda_, self.rho, weights or self._generate_weights(Xb))

                # Lines 15-19: Aggregate and update U, S, Vh
                Xhat_concat = np.vstack(Xhatb)
                Z2_concat = np.vstack(Z2b)
                (U, S, Vh) = randomized_svd(Xhat_concat + Z2_concat, n_components=self.problem_rank)
                # Line 17: Update U using means
                U = (Xhat_concat + Z2_concat + X_mean) @ Vh.T / S
                U /= np.linalg.norm(U, axis=0)

                # Lines 20-23: Dual updates
                Z2_update = self.rho * (Xhat_concat - U @ np.diag(S) @ Vh)
                for b in range(B):
                    idx = slice(b*X_batches[b].shape[0], (b+1)*X_batches[b].shape[0])
                    Z1b[b] += self.rho * (D @ Xhatb[b] - Gb[b])
                    Z2b[b] += Z2_update[idx]

            # Lines 24-30: Store path results
            self.Xhat_list.append(Xhat_concat.copy())
            self.G_list.append(np.vstack(Gb).copy())
            self.U_list.append(U.copy())
            self.S_list.append(S.copy())
            self.V_list.append(Vh.copy())

        # Line 33: Return results
        return {
            'Xhat': self.Xhat_list,
            'G': self.G_list,
            'U': self.U_list,
            'S': self.S_list,
            'V': self.V_list
        }

    def return_fit(self, lam=None):
        """
        Return fitted variables for a given penalty index or for the whole path.

        Args:
            lam (int or str or None): If int, return results at that index.
                If "best", return results at the selected best lambda.
                If None, return the entire path as a dict.

        Returns:
            tuple or dict: (Xhat, G, U, S, V) for a specific lambda, or a dict of lists for all lambdas.
        """
        if isinstance(lam, int) and lam in range(len(self.penalty_list)):
            return (
                self.Xhat_list[lam],
                self.G_list[lam],
                self.U_list[lam],
                self.S_list[lam],
                self.V_list[lam]
            )
        elif lam == "best":
            if hasattr(self, "best_lam"):
                return (
                    self.Xhat_list[self.best_lam],
                    self.G_list[self.best_lam],
                    self.U_list[self.best_lam],
                    self.S_list[self.best_lam],
                    self.V_list[self.best_lam]
                )
            else:
                raise ValueError("Run .select() first to compute and set best lambda.")
        else:
            # Return the entire path as dictionary.
            return {
                'Xhat': self.Xhat_list,
                'G': self.G_list,
                'U': self.U_list,
                'S': self.S_list,
                'V': self.V_list
            }

    def select(self, labels_true=None, trim_init_lambdas=2):
        """
        Estimate clusters along the path using spectral clustering on reconstructed estimates.

        Args:
            labels_true (ndarray or None, optional): Ground-truth labels; if None, uses self.true_clusters.
            trim_init_lambdas (int, optional): Number of initial penalties to skip (initialization phase).

        Returns:
            list: Predicted cluster labels for each penalty value along the path.
        """
        print('Estimating clusters along Consensus PCMF path.')
        if labels_true is None:
            labels_true = self.true_clusters

        ari_list=[]
        nmi_list=[]
        acc_list=[]
        labels_list = []
        for p in np.arange(0,trim_init_lambdas): # 
            labels_list.append(None)
            ari_list.append(0)
            nmi_list.append(0)
            acc_list.append(0)
        for p in np.arange(trim_init_lambdas,len(self.penalty_list)): # 
            labels, ari, nmi, acc = fit_spectral(np.array(self.Xhat_list)[p,:,:], labels_true, len(np.unique(labels_true)))
            labels_list.append(labels)
            ari_list.append(ari)
            nmi_list.append(nmi)
            acc_list.append(acc)

        self.path_labels_list = labels_list
        self.path_ari_list = ari_list
        self.path_nmi_list = nmi_list
        self.path_acc_list = acc_list
        self.best_lam = int(np.argmax(np.array(ari_list)))

        print(
            f"Best Penalty IDX: {int(np.argmax(np.array(ari_list)))}, "
            f"ARI: {float(np.max(np.array(ari_list)))}, "
            f"NMI: {float(np.max(np.array(nmi_list)))}"
        )

        return labels_list

    def return_labels(self, lam=None):
        """
        Return predicted labels at a given penalty index, the best index, or all.

        Args:
            lam (int or str or None): If int, labels at that index; if "best", labels at best lambda;
                if None, labels for all lambdas.

        Returns:
            ndarray or list: Labels for the specified lambda or a list across the path.
        """
        if lam is not None and isinstance(lam, int) and lam in range(len(self.penalty_list)):
            return self.path_labels_list[lam]
        elif lam == "best":
            return self.path_labels_list[self.best_lam]
        else:
            return self.path_labels_list
  
    def plot_path(self, plot_range=None):
        """
        Plot reconstructed estimates along the penalty path.

        Args:
            plot_range (tuple[int, int], optional):
                A (start, end) index slice over penalty values.
                Use -1 for the end index to indicate "until the final penalty".
                Defaults to (0, -1).

        Returns:
            None
        """      
        if plot_range is None: 
            plot_range = (0, -1)

        # Plot estimates along path for first three variables
        ax = path_plot(np.asarray(self.Xhat_list)[:,:,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel(r"$\hat{X}$ estimates")
        plt.locator_params(nbins=25)

        # Plot estimates along path separated by true cluster label for first three variables
        path_plot(np.asarray(self.Xhat_list)[:, self.true_clusters==0,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel(r"$\hat{X}$ estimates")
        plt.locator_params(nbins=25)

        path_plot(np.asarray(self.Xhat_list)[:, self.true_clusters==1,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel(r"$\hat{X}$ estimates")
        plt.locator_params(nbins=25)


class P3CA:
    """
    Pathwise Clustered Canonical Correlation Analysis (P3CA).

    Implements Appendix Algorithm 6 from Buch et al., AISTATS 2024.
    Uses ADMM updates for canonical correlation analysis with pathwise
    clustering across a sequence of penalties.

    Args:
        penalty_list (list or array): Sequence of penalty (lambda) values for the path.
        rho (float): Augmented Lagrangian parameter.
        admm_iters (int): Number of ADMM updates per iteration.
        cca_iters (int): Number of canonical correlation iterations.
        gauss_coef (float): Gaussian coefficient for weights.
        neighbors (int or None): Number of nearest neighbors for graph construction.
        weights (ndarray or None): Predefined pairwise weights; if None, auto-generated.
        non_negative (bool): If True, enforce non-negativity on factor matrices.
        prox_method (str): Which proximal operator to use ('numba', 'prox_c', or 'python').
        progress_mode (str): Progress reporting style ('auto', 'bar', 'file', or 'none').
        output_file (str or None): File to log progress when progress_mode='file'.

    Attributes:
        U_list (list): Learned canonical factor matrices for X across the path.
        V_list (list): Learned canonical factor matrices for Y across the path.
    """

    def __init__(
        self,
        penalty_list,
        rho=1.0,
        admm_iters=2,
        cca_iters=3,
        gauss_coef=0.5,
        neighbors=None,
        weights=None,
        non_negative=False,
        prox_method='numba',
        progress_mode="auto",
        output_file=None,
    ):
        self.penalty_list = penalty_list
        self.rho = rho
        self.admm_iters = admm_iters
        self.cca_iters = cca_iters
        self.gauss_coef = gauss_coef
        self.neighbors = neighbors
        self.weights = weights
        self.non_negative = non_negative
        self.prox_method = prox_method
        self.progress_mode = progress_mode
        self.output_file = output_file

        self.U_list = []
        self.V_list = []

    def _generate_weights(self, X):
        """
        Generate pairwise weights for clustering using a Gaussian kernel.

        Args:
            X (ndarray): Input data matrix of shape (n_samples, n_features).

        Returns:
            ndarray: Flattened pairwise weights in condensed pdist order.
        """
        return get_weights(X, gauss_coef=self.gauss_coef, neighbors=self.neighbors)

    def _progress_iterator(self, iterable, desc="Lambda path"):
        """
        Wrap an iterable with a progress indicator (bar, file logging, or none).

        Args:
            iterable (iterable): Items to iterate over.
            desc (str): Description label for the progress reporting.

        Returns:
            iterator: Wrapped iterator with progress reporting applied.
        """
        mode = self.progress_mode
        if mode == "auto":
            mode = "bar" if self.output_file is None else "file"
        if mode == "bar":
            return tqdm(iterable, desc=desc)
        elif mode == "file":
            def iterator():
                total = len(iterable)
                for i, item in enumerate(iterable):
                    try:
                        with open(self.output_file, "a") as f:
                            f.write(f"{desc} {i+1}/{total}\n")
                    except Exception as e:
                        print(f"[progress_mode=file] Failed to write to {self.output_file!r}: {e}")
                        print("→ Switching progress reporting to 'off'.")
                        # fallback to no progress reporting
                        for item2 in iterable[i:]:
                            yield item2
                        return
                    yield item
            return iterator()
        else:
            return iterable

    def fit(self, X, Y, true_clusters, weights=None):
        """
        Fit the P3CA model along a penalty path (ADMM-based P3CA updates).

        Args:
            X (ndarray): First view data of shape (n_samples, n_features_X).
            Y (ndarray): Second view data of shape (n_samples, n_features_Y).
            true_clusters (ndarray): Ground-truth cluster labels of shape (n_samples,).
            weights (ndarray or None, optional): Precomputed weights; if None, generated from X.

        Returns:
            dict: Results with keys {'U','V'}, each a list across the penalty path.
        """
        assert X.ndim == 2 and Y.ndim == 2, "X, Y must be 2D matrices"
        assert X.shape[0] == Y.shape[0], "X, Y must have same number of samples"
        assert len(self.penalty_list) > 0, "Penalty list must be non-empty"

        if weights is not None:
            self.weights = weights
        if self.weights is None:
            self.weights = self._generate_weights(X)

        self.U_list.clear()
        self.V_list.clear()

        self.true_clusters = true_clusters

        X = X.copy()
        Y = Y.copy()
        N, pX = X.shape
        _, pY = Y.shape

        Dx = get_default_D(X)
        Dy = get_default_D(Y)
        Lx = compute_cholesky(X.shape[0], self.rho, Dx, reg_scale=1.0)
        Ly = compute_cholesky(Y.shape[0], self.rho, Dy, reg_scale=1.0)

        Wx = Zx = Dx @ X
        Wy = Zy = Dy @ Y

        penalty0 = self.penalty_list[0]
        penaltyN = self.penalty_list[-1]

        # Initial U update
        if penalty0 > penaltyN:
            V_initial = np.tile(np.mean(Y, axis=0), (Y.shape[0], 1))
        else:
            V_initial = Y.copy()
        Xu_tildes = []
        for i in range(X.shape[0]):
            Xu_tildes.append(np.dot(np.outer(X[i, :].T, Y[i, :]), V_initial[i, :]))
        Xu = np.asarray(Xu_tildes)
        U, Wx, Zx = P3CA_update(
            Xu, Lx, Dx, Wx, Zx, self.rho, penalty0, X, 
            weights=self.weights, neighbors=self.neighbors, prox_method=self.prox_method, gauss_coef=self.gauss_coef)

        # Initial V update
        if penalty0 > penaltyN:
            U_initial = np.tile(np.mean(X, axis=0), (X.shape[0], 1))
        else:
            U_initial = X.copy()
        Yv_tildes = []
        for i in range(Y.shape[0]):
            Yv_tildes.append(np.dot(np.outer(Y[i, :].T, X[i, :]), U_initial[i, :]))
        Yv = np.asarray(Yv_tildes)
        V, Wy, Zy = P3CA_update(
            Yv, Ly, Dy, Wy, Zy, self.rho, penalty0, Y,
            weights=self.weights, neighbors=self.neighbors, prox_method=self.prox_method, gauss_coef=self.gauss_coef)

        # Penalty loop
        try:
            for idx, penalty in enumerate(self._progress_iterator(self.penalty_list, desc="Lambda path")):
                for it in range(self.cca_iters):
                    # U update
                    for _ in range(self.admm_iters):
                        Xu_tildes = []
                        for i in range(X.shape[0]):
                            Xu_tildes.append(np.dot(np.outer(X[i, :].T, Y[i, :]), V[i, :]))
                        Xu = np.asarray(Xu_tildes)
                        U, Wx, Zx = P3CA_update(
                            Xu, Lx, Dx, Wx, Zx, self.rho, penalty, X,
                            weights=self.weights, neighbors=self.neighbors, prox_method=self.prox_method, gauss_coef=self.gauss_coef)
                        if self.non_negative:
                            U[U < 0] = 0
                    # V update
                    for _ in range(self.admm_iters):
                        Yv_tildes = []
                        for i in range(Y.shape[0]):
                            Yv_tildes.append(np.dot(np.outer(Y[i, :].T, X[i, :]), U[i, :]))
                        Yv = np.asarray(Yv_tildes)
                        V, Wy, Zy = P3CA_update(
                            Yv, Ly, Dy, Wy, Zy, self.rho, penalty, Y,
                            weights=self.weights, neighbors=self.neighbors, prox_method=self.prox_method, gauss_coef=self.gauss_coef)
                        if self.non_negative:
                            V[V < 0] = 0
                self.U_list.append(U.copy())
                self.V_list.append(V.copy())

        except KeyboardInterrupt:
            print("KeyboardInterrupt has been caught.")

        return {'U': self.U_list, 'V': self.V_list}

    def return_fit(self, lam=None):
        """
        Return fitted variables for a given penalty index or for the whole path.

        Args:
            lam (int or str or None): If int, return results at that index.
                If "best", return results at the selected best lambda.
                If None, return the entire path.

        Returns:
            tuple or tuple[list]: (U, V) for a specific lambda, or the full path as lists.
        """
        if isinstance(lam, int) and lam in range(len(self.penalty_list)):
            return (
                self.U_list[lam],
                self.V_list[lam]
            )
        elif lam == "best":
            if hasattr(self, "best_lam"):
                return (
                    self.U_list[self.best_lam],
                    self.V_list[self.best_lam]
                )
            else:
                raise ValueError("Run .select() first to compute and set best lambda.")
        else:
            # Return the entire path as tuple of lists
            return (
                self.U_list,
                self.V_list
            )

    def select(self, labels_true=None, trim_init_lambdas=2):
        """
        Estimate clusters along the path using spectral clustering on concatenated (U, V).

        Args:
            labels_true (ndarray or None, optional): Ground-truth labels; if None, uses self.true_clusters.
            trim_init_lambdas (int, optional): Number of initial penalties to skip (initialization phase).

        Returns:
            list: Predicted cluster labels for each penalty value along the path.
        """
        print('Estimating clusters along P3CA path.')
        if labels_true is None:
            labels_true = self.true_clusters

        ari_list=[]
        nmi_list=[]
        acc_list=[]
        labels_list = []
        for p in np.arange(0,trim_init_lambdas): # 
            labels_list.append(None)
            ari_list.append(0)
            nmi_list.append(0)
            acc_list.append(0)
        for p in np.arange(trim_init_lambdas,len(self.penalty_list)): # 
            labels, ari, nmi, acc = fit_spectral(np.hstack((np.array(self.U_list)[p,:,:],np.array(self.V_list)[p,:,:])), labels_true, len(np.unique(labels_true)))
            labels_list.append(labels)
            ari_list.append(ari)
            nmi_list.append(nmi)
            acc_list.append(acc)

        self.path_labels_list = labels_list
        self.path_ari_list = ari_list
        self.path_nmi_list = nmi_list
        self.path_acc_list = acc_list
        self.best_lam = int(np.argmax(np.array(ari_list)))

        print(
            f"Best Penalty IDX: {int(np.argmax(np.array(ari_list)))}, "
            f"ARI: {float(np.max(np.array(ari_list)))}, "
            f"NMI: {float(np.max(np.array(nmi_list)))}"
        )

        return labels_list

    def return_labels(self, lam=None):
        """
        Return predicted labels at a given penalty index, the best index, or all.

        Args:
            lam (int or str or None): If int, labels at that index; if "best", labels at best lambda;
                if None, labels for all lambdas.

        Returns:
            ndarray or list: Labels for the specified lambda or a list across the path.
        """
        if lam is not None and isinstance(lam, int) and lam in range(len(self.penalty_list)):
            return self.path_labels_list[lam]
        elif lam == "best":
            return self.path_labels_list[self.best_lam]
        else:
            return self.path_labels_list
  
    def plot_path(self, plot_range=None):
        """
        Plot U and V estimates along the penalty path.

        Args:
            plot_range (tuple[int, int], optional):
                A (start, end) index slice over penalty values.
                Use -1 for the end index to indicate "until the final penalty".
                Defaults to (0, -1).

        Returns:
            None
        """
        if plot_range is None: 
            plot_range = (0, -1)
        # Plot estimates along path for first three variables
        ax = path_plot(np.asarray(self.U_list)[:,:,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel("U estimates")
        plt.locator_params(nbins=25)

        # Plot estimates along path for first three variables
        ax = path_plot(np.asarray(self.V_list)[:,:,0:3], self.penalty_list, plot_range=plot_range)
        despine()
        plt.ylabel("V estimates")
        plt.locator_params(nbins=25)

        # # Plot estimates along path separated by true cluster label for first three variables
        # path_plot(np.asarray(self.U_list)[:, self.true_clusters==0,0:3], self.penalty_list, plot_range=plot_range)
        # path_plot(np.asarray(self.V_list)[:, self.true_clusters==1,0:3], self.penalty_list, plot_range=plot_range)
