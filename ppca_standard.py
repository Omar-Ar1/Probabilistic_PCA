import numpy as np 
from numpy import linalg

class PPCA:
    """
    Probabilistic Principal Component Analysis (PPCA) class.

    Attributes:
        d (int): Dimensionality of the observed data.
        q (int): Dimensionality of the latent variables.
        W (np.ndarray): Weight matrix initialized randomly.
        mu_ (np.ndarray): Mean of the observed data.
        sigma_ (float): Initial noise variance (scalar).
        log_likelihoods (list): List to store log-likelihood values for plotting.
    """
    def __init__(self, d, q):
        """
        Initialize the PPCA model with given dimensions.

        Args:
            d (int): Dimensionality of the observed data.
            q (int): Dimensionality of the latent variables.
        """
        self.d = d
        self.q = q
        self.W = np.random.randn(d, q)  # Initialize W randomly
        self.mu_ = None  # Mean of observed data
        self.sigma_ = 0.2 # Initial noise variance (scalar)
        self.log_likelihoods = []  # Store log-likelihood values for plotting

    def fit(self, t, max_iter=100, tol=1e-6):
        """
        Fit the PPCA model to the observed data.

        Args:
            t (np.ndarray): Observed data matrix of shape (d, N).
            max_iter (int): Maximum number of iterations for the EM algorithm.
            tol (float): Tolerance for convergence.

        Returns:
            None
        """      
        N = t.shape[1]  # Number of samples
        self.mu_ = np.mean(t, axis=1)  # Mean of observed data (d,)
        centered_t = t - self.mu_[:, None]
        likelihood_ = -np.inf
        converged = False
        i = 0
        while not converged and i < max_iter:
            M = self.W.T @ self.W + self.sigma_**2 * np.eye(self.q)
            M_inv = linalg.inv(M)
            x = M_inv @ self.W.T @ centered_t
            ExxT = self.sigma_**2 * M_inv + (x @ x.T) / N

            S = centered_t @ centered_t.T / N
            self.W = S @ self.W @ linalg.inv(self.W.T @ self.W + self.sigma_**2 * np.eye(self.q))

            sigma_squared_update = (1 / self.d) * np.trace(S - self.W @ self.W.T)
            self.sigma_ = np.sqrt(max(sigma_squared_update, 1e-6))

            C = self.W @ self.W.T + self.sigma_**2 * np.eye(self.d)
            try:
                likelihood_new = -0.5 * N * (self.d * np.log(2 * np.pi) + np.log(np.linalg.det(C)) + np.trace(linalg.inv(C) @ S))
            except np.linalg.LinAlgError:
                print(f"Singular matrix encountered at iteration {i + 1}")
                break

            self.log_likelihoods.append(likelihood_new)
            if np.abs(likelihood_new - likelihood_) < tol:
                converged = True
            i += 1
            likelihood_ = likelihood_new
    def transform(self, t):
        """
        Transform the observed data to the latent space.

        Args:
            t (np.ndarray): Observed data matrix of shape (d, N).

        Returns:
            np.ndarray: Transformed data matrix of shape (q, N).
        """
        centered_t = t - self.mu_[:, None]
        M = self.W.T @ self.W + self.sigma_**2 * np.eye(self.q)
        M_inv = linalg.inv(M)
        x = M_inv @ self.W.T @ centered_t
        return x

    def predict(self, t):
        """
        Reconstruct the observed data from the latent space.

        Args:
            t (np.ndarray): Observed data matrix of shape (d, N).

        Returns:
            np.ndarray: Reconstructed data matrix of shape (d, N).
        """
        x = self.transform(t)
        t_reconstructed = self.W @ x + self.mu_[:, None]
        return t_reconstructed