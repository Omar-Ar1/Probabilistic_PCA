import torch
class PPCA:
    """
    Probabilistic Principal Component Analysis (PPCA) class optimized for GPU.

    Attributes:
        n_components (int): Number of principal components.
        max_iter (int): Maximum number of iterations for the EM algorithm.
        tol (float): Tolerance for convergence.
        verbose (bool): Whether to print convergence messages.
        device (str): Device to run the computations on (e.g., 'cuda' or 'cpu').
        stochastic_em (bool): Whether to use stochastic EM.
        fp16 (bool): Whether to use mixed precision (float16).
    """
    def __init__(self, n_components, max_iter=6000, tol=1e-4, stochastic_em=False, 
                 verbose=False, device="cuda", fp16=False):
        """
        Initialize the PPCA_GPU_Optimized model with given parameters.

        Args:
            n_components (int): Number of principal components.
            max_iter (int): Maximum number of iterations for the EM algorithm.
            tol (float): Tolerance for convergence.
            stochastic_em (bool): Whether to use stochastic EM.
            verbose (bool): Whether to print convergence messages.
            device (str): Device to run the computations on (e.g., 'cuda' or 'cpu').
            fp16 (bool): Whether to use mixed precision (float16).
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.device = device
        self.stochastic_em = stochastic_em
        self.fp16 = fp16  # Mixed precision support

    def fit(self, X):
        """
        Fit the PPCA model to the observed data.

        Args:
            X (torch.Tensor): Observed data matrix of shape (n_samples, n_features).

        Returns:
            None
        """
        dtype = torch.float16 if self.fp16 else torch.float32
        X = X.to(self.device, dtype=dtype)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.mean_ = torch.nanmean(X, dim=0)
        X_centered = torch.nan_to_num(X - self.mean_, nan=0.0)
        
        # SVD initialization
        valid_mask = ~torch.isnan(X)
        valid_rows = valid_mask.all(dim=1)
        if valid_rows.sum() > self.n_components:
            U, S, Vt = torch.linalg.svd(X_centered[valid_rows], full_matrices=False)
            self.W_ = (Vt[:self.n_components].T * (S[:self.n_components] / torch.sqrt(torch.tensor(n_samples, device=self.device))))
        else:
            self.W_ = torch.randn(n_features, self.n_components, device=self.device, dtype=dtype)
            
        self.sigma2_ = torch.nanmean(X_centered**2) * 0.1
        log_likelihood_old = -torch.inf
        
        # EM loop
        for iter in range(self.max_iter):
            Ez, Ezz = self.e_step_(X_centered, valid_mask)
            self.m_step_(X_centered, valid_mask, Ez, Ezz)
            
            # Convergence check
            log_likelihood = self._log_likelihood(X_centered, valid_mask)
            if torch.abs(log_likelihood - log_likelihood_old) < self.tol:
                if self.verbose: print("Converged.")
                break
            log_likelihood_old = log_likelihood
    def transform(self, X):
        """
        Project data into latent space.

        Args:
            X (torch.Tensor): Observed data matrix of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Transformed data matrix of shape (n_samples, n_components).
        """
        X = X.to(self.device)
        X_centered = torch.nan_to_num(X - self.mean_, nan=0.0)
        
        # Compute latent representation
        M = self.W_.T @ self.W_ + self.sigma2_ * torch.eye(self.n_components, device=self.device)
        M_inv = torch.linalg.inv(M)
        Z = (X_centered @ self.W_) @ M_inv
        return Z
    def reconstruct(self, X):
        """
        Impute missing values using the learned model.

        Args:
            X (torch.Tensor): Observed data matrix of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Reconstructed data matrix of shape (n_samples, n_features).
        """
        Z = self.transform(X)
        X_recon = Z @ self.W_.T + self.mean_
        
        # Preserve observed values
        observed_mask = ~torch.isnan(X)
        X_recon[observed_mask] = X[observed_mask]
        return X_recon

    def e_step_(self, X_centered, mask):
        """
        Perform the E-step of the EM algorithm.

        Args:
            X_centered (torch.Tensor): Centered observed data matrix.
            mask (torch.Tensor): Mask indicating observed values.

        Returns:
            tuple: Expected values of the latent variables and their second moments.
        """
        observed = mask.float()
        Wo = self.W_.unsqueeze(0) * observed.unsqueeze(-1)  # (n_samples, n_features, n_components)
        Sn_inv = (Wo.transpose(1,2) @ Wo) / self.sigma2_ + torch.eye(self.n_components, device=self.device)
        Sn = torch.linalg.inv(Sn_inv)
        
        diff = (X_centered * observed).unsqueeze(-1)
        Mn = Sn @ (Wo.transpose(1,2) @ diff) / self.sigma2_
        Mn = Mn.squeeze(-1)
        
        Ezz = Sn + Mn.unsqueeze(-1) @ Mn.unsqueeze(1)
        return Mn, Ezz

    def m_step_(self, X_centered, mask, Ez, Ezz):
        """
        Perform the M-step of the EM algorithm.

        Args:
            X_centered (torch.Tensor): Centered observed data matrix.
            mask (torch.Tensor): Mask indicating observed values.
            Ez (torch.Tensor): Expected values of the latent variables.
            Ezz (torch.Tensor): Expected second moments of the latent variables.

        Returns:
            None
        """
        observed = mask.float()
        
        # Update W
        numerator = (X_centered.unsqueeze(-1) * observed.unsqueeze(-1)) @ Ez.unsqueeze(1)
        denominator = Ezz.sum(dim=0) + 1e-6
        self.W_ = numerator.sum(dim=0) @ torch.linalg.inv(denominator)
        
        # Update sigma2
        recon_error = (X_centered * observed - Ez @ self.W_.T).pow(2).sum()
        trace_term = torch.einsum('nij,ji->', Ezz, self.W_.T @ self.W_)
        self.sigma2_ = (recon_error + trace_term) / observed.sum()

    def _log_likelihood(self, X_centered, mask):
        """
        Compute the log-likelihood of the observed data given the current model parameters.

        Args:
            X_centered (torch.Tensor): The centered observed data matrix of shape (n_samples, n_features).
            mask (torch.Tensor): A boolean mask indicating the observed (True) and missing (False) values in the data.

        Returns:
            torch.Tensor: The log-likelihood of the observed data.
        """
        observed = mask.float()
        C = self.W_ @ self.W_.T + self.sigma2_ * torch.eye(self.W_.shape[0], device=self.device)
        logdet_C = torch.logdet(C)
        diff = (X_centered * observed) @ torch.linalg.inv(C)
        mahal = (diff * (X_centered * observed)).sum(dim=1)
        
        return -0.5 * (logdet_C + mahal + observed.sum(dim=1) * torch.log(torch.tensor(2 * torch.pi))).sum()