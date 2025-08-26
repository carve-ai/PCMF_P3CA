import matplotlib.pyplot as plt
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import StandardScaler

class Loader1D:
    """
    Loader class for a single dataset for PCMF and LL-PCMF.
    """

    @staticmethod
    def generate_two_cluster_data(m=[50, 50], means=[0, 0], n_X=200, sigma=0.075, density=1.0, 
                                 seed=1, gen_seeds=True, seeds=[], scale_data=True, intercept=False, plot=False):
        """Generates synthetic single dataset drawn from two different distributions (classes)."""
        np.random.seed(seed)
        n_clusters = 2
        X_clusters, u_true, v_true, _ = Loader1D.generate_cluster_PMD_data(
            m, n_X, sigma, density, n_clusters, means, gen_seeds=gen_seeds, seeds=seeds, intercept=False, verbose=False
        )
        X_c = np.vstack(X_clusters)
        true_clusters = np.repeat([0, 1], m)
        if scale_data:
            scaler = StandardScaler()
            X_c = scaler.fit_transform(X_c)
        if intercept:
            X_c = np.hstack((X_c, np.ones((X_c.shape[0], 1))))

        # Optional plotting
        if plot:
            plt.scatter(X_clusters[0][:, 0], X_clusters[0][:, 1], c='darkblue')
            plt.scatter(X_clusters[1][:, 0], X_clusters[1][:, 1], c='darkorange')
            plt.axis("off")
            plt.show()

            # Plot data matrix as image
            plt.figure()
            maxval = np.max(np.abs(X_c))
            plt.imshow(X_c,aspect='auto',interpolation='nearest',cmap='twilight_shifted',vmin=-1*maxval, vmax=maxval)  

        # Permute order for running consensus batches so that clustered data/labels are not ordered.
        np.random.seed(seed=1234)
        idx_perm = np.random.permutation(X_c.shape[0])
        X_c = X_c[idx_perm,:]
        true_clusters = true_clusters[idx_perm]

        return X_c, true_clusters

    @staticmethod
    def generate_cluster_PMD_data(m=[100, 100], n_X=20, sigma=0.01, density=0.2, n_clusters=2, 
                                  means=[0, 0], gen_seeds=True, seeds=[], intercept=False, verbose=False):
        """Generates synthetic dataset for n_clusters distributions/classes."""
        X_out, u_stars, v_stars = [], [], []
        for nc in range(n_clusters):
            if gen_seeds:
                seed = np.random.randint(99999)
                seeds.append(seed)
            if verbose:
                print(seeds)
            X, u_star, v_star = Loader1D.generate_PMD_data(
                m[nc], n_X, sigma, density, mean=means[nc], seed=seeds[nc], intercept=intercept
            )
            X_out.append(X)
            u_stars.append(u_star)
            v_stars.append(v_star)
        return X_out, u_stars, v_stars, seeds

    @staticmethod
    def generate_PMD_data(m=100, n_X=10, sigma=1, density=0.5, mean=0, seed=1, scale_data=False, intercept=False):
        """
        Generates a synthetic dataset with a latent low-rank signal structure.

        Args:
            m (int): Number of observations (rows in X).
            n_X (int): Number of variables (columns in X).
            sigma (float): Standard deviation of Gaussian noise.
            density (float): Proportion of variables that carry structured signal (between 0 and 1).
            mean (float): Mean of the Gaussian noise distribution.
            seed (int): Random seed for reproducibility.
            scale_data (bool): Whether to standardize X with zero mean and unit variance.

        Returns:
            X (ndarray): Generated data matrix of shape (m, n_X).
            u_star (ndarray): Latent left singular vector (true component for rows).
            v_star (ndarray): Latent right singular vector (true component for columns).
        """
        np.random.seed(seed)
        u_star = np.random.randn(m) / 3.0
        v_star = np.random.randn(n_X) / 3.0
        X = np.random.normal(mean, sigma, size=(m, n_X)) / 3.0
        X_idxs = np.random.choice(range(n_X), int(density * n_X), replace=False)
        for idx in X_idxs:
            X[:, idx] += v_star[idx] * u_star
        if scale_data:
            X = StandardScaler().fit_transform(X)
        if intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X, u_star, v_star

    @staticmethod
    def load_custom_file(filepath_X, filepath_X_labels, scale_data=True, intercept=False):
        """
        Loads feature matrix X and corresponding labels from CSV files.
        
        Args:
            filepath_X (str): Path to CSV file containing feature matrix.
            filepath_X_labels (str): Path to CSV file containing cluster labels.
            scale_data (bool): Whether to standardize features.
            intercept (bool): Whether to append intercept (column of ones).

        Returns:
            X (ndarray): Feature matrix (possibly scaled and with intercept).
            true_clusters (ndarray): Permuted cluster labels.
        """
        X = np.loadtxt(filepath_X, delimiter=',')
        true_clusters = np.loadtxt(filepath_X_labels, delimiter=',')
        
        if scale_data:
            X = StandardScaler().fit_transform(X)
        if intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Permute data and labels for randomized batch order
        np.random.seed(1234)
        idx_perm = np.random.permutation(X.shape[0])
        X = X[idx_perm, :]
        true_clusters = true_clusters[idx_perm]

        return X, true_clusters

class Loader2D:
    """
    Loader class for paired X and Y datasets for P3CA.
    """

    @staticmethod
    def load_penguins_data(scale_data=True, intercept=False):
        """
        Loads the Palmer Penguins dataset as paired X and Y arrays with species labels.

        Args:
            scale_data (bool): Whether to standardize X and Y with zero mean and unit variance.
            intercept (bool): Whether to append an intercept (column of ones) to X and Y.

        Returns:
            X (ndarray): Feature array of shape (n_samples, 2), containing bill length and bill depth.
            Y (ndarray): Feature array of shape (n_samples, 2), containing flipper length and body mass.
            species_labels (ndarray): Integer-encoded labels for penguin species.
            species (ndarray): Original species names as strings.
            species_names (ndarray): Array of unique species names in label order.
        """        
        data, species = load_penguins(return_X_y=True)
        data_arr = data.to_numpy()
        keep = ~np.isnan(data_arr).any(axis=1)
        data_arr = data_arr[keep]
        species = species[keep]
        X = data_arr[:, 0:2]
        Y = data_arr[:, 2:4]
        if scale_data:
            scalerX = StandardScaler()
            scalerY = StandardScaler()
            X = scalerX.fit_transform(X)
            Y = scalerY.fit_transform(Y)

        if intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            Y = np.hstack((Y, np.ones((Y.shape[0], 1))))

        species_names, species_labels = np.unique(species, return_inverse=True)

        return X, Y, species_labels,species, species_names

    @staticmethod
    def load_syntheticXY(m=100, n_X=10, n_Y=8, sigma=0.075, density=0.5, mean=0, seed=1, scale=False, intercept=False):
        """
        Generates a synthetic paired X-Y dataset with structured low-rank signal for simulation or benchmarking.

        Args:
            m (int): Number of observations.
            n_X (int): Number of variables in X.
            n_Y (int): Number of variables in Y.
            sigma (float): Standard deviation of Gaussian noise.
            density (float): Proportion of variables carrying structured signal.
            mean (float): Mean of the noise distribution.
            seed (int): Random seed for reproducibility.
            scale (bool): Whether to standardize X and Y with zero mean and unit variance.
            intercept (bool): Whether to append an intercept (column of ones) to X and Y.

        Returns:
            X (ndarray): Generated data matrix for X of shape (m, n_X [+1] if intercept).
            Y (ndarray): Generated data matrix for Y of shape (m, n_Y [+1] if intercept).
        """
        X, _, _ = Loader2D.generate_PMD_data(m=m, n_X=n_X, sigma=sigma, density=density, mean=mean, seed=seed, scale_data=scale)
        Y, _, _ = Loader2D.generate_PMD_data(m=m, n_X=n_Y, sigma=sigma, density=density, mean=mean, seed=seed, scale_data=scale)
        if scale:
            scalerX, scalerY = StandardScaler(), StandardScaler()
            X, Y = scalerX.fit_transform(X), scalerY.fit_transform(Y)

        if intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            Y = np.hstack((Y, np.ones((Y.shape[0], 1))))

        return X, Y

    @staticmethod
    def generate_PMD_data(m=100, n_X=10, sigma=1, density=0.5, mean=0, seed=1, scale_data=False):
        """
        Generates a synthetic dataset with a latent low-rank signal structure.

        Args:
            m (int): Number of observations (rows in X).
            n_X (int): Number of variables (columns in X).
            sigma (float): Standard deviation of Gaussian noise.
            density (float): Proportion of variables that carry structured signal (between 0 and 1).
            mean (float): Mean of the Gaussian noise distribution.
            seed (int): Random seed for reproducibility.
            scale_data (bool): Whether to standardize X with zero mean and unit variance.

        Returns:
            X (ndarray): Generated data matrix of shape (m, n_X).
            u_star (ndarray): Latent left singular vector (true component for rows).
            v_star (ndarray): Latent right singular vector (true component for columns).
        """
        np.random.seed(seed)
        u_star = np.random.randn(m) / 3.0
        v_star = np.random.randn(n_X) / 3.0
        X = np.random.normal(mean, sigma, size=(m, n_X)) / 3.0
        X_idxs = np.random.choice(range(n_X), int(density * n_X), replace=False)
        for idx in X_idxs:
            X[:, idx] += v_star[idx] * u_star
        if scale_data:
            X = StandardScaler().fit_transform(X)
        return X, u_star, v_star

    @staticmethod
    def load_custom_paired_files(filepath_X, filepath_Y, filepath_XY_labels, scale=True, intercept=False):
        """
        Loads paired feature matrices X and Y and their corresponding labels.

        Args:
            filepath_X (str): Path to CSV file with X data.
            filepath_Y (str): Path to CSV file with Y data.
            filepath_XY_labels (str): Path to CSV file with true cluster labels.
            scale (bool): Whether to standardize X and Y.
            intercept (bool): Whether to append an intercept (column of ones) to X and Y.

        Returns:
            X (ndarray): Processed feature matrix X.
            Y (ndarray): Processed feature matrix Y.
            true_clusters (ndarray): Cluster labels associated with rows in X and Y.
        """
        X = np.loadtxt(filepath_X, delimiter=',')
        Y = np.loadtxt(filepath_Y, delimiter=',')
        true_clusters = np.loadtxt(filepath_XY_labels, delimiter=',')

        if scale:
            X = StandardScaler().fit_transform(X)
            Y = StandardScaler().fit_transform(Y)

        if intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            Y = np.hstack((Y, np.ones((Y.shape[0], 1))))

        # Randomize order for consensus runs
        np.random.seed(1234)
        idx_perm = np.random.permutation(X.shape[0])
        X = X[idx_perm, :]
        Y = Y[idx_perm, :]
        true_clusters = true_clusters[idx_perm]

        return X, Y, true_clusters
