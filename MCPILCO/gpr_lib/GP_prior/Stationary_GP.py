# Copyright (C) 2023 Alberto Dalla Libera
#
# SPDX-License-Identifier: MIT
"""
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""

import numpy as np
import torch

from . import GP_prior


class Stationary_GP(GP_prior.GP_prior):
    """
    Superclass of the stationary GP:
    Define common initializations and provide a function that
    computes the squared distances weighted by the lengthscales
    """

    def __init__(
        self,
        active_dims,
        lengthscales_init=None,
        flg_train_lengthscales=True,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        sigma_n_num=None,
        device=None,
    ):
        """
        Initialize the module and set the lengthscales parameters
        In order to constrain the parameters to be positive we considered
        the log of the lengthscales
        """
        super(Stationary_GP, self).__init__(
            active_dims,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            sigma_n_num=sigma_n_num,
            device=device,
        )
        # get the number of features
        if active_dims is None:
            raise RuntimeError("Stationary_GP obj require active_dims")
        else:
            self.num_features = active_dims.size
        # check the ARD flag and set the length scalesinitial value
        flg_ARD = True
        if lengthscales_init.size == 1:
            flg_ARD = False
        if lengthscales_init is None:
            lengthscales_init = np.ones(self.num_features)
        self.flg_ARD = flg_ARD
        # get the lengthscale
        self.log_lengthscales_par = torch.nn.Parameter(
            torch.tensor(np.log(lengthscales_init), dtype=self.dtype, device=self.device),
            requires_grad=flg_train_lengthscales,
        )

    def get_weigted_distances(self, X1, X2):
        """
        Computes (X1-X2)^T*sigma^-2*(X1-X2),
        where Sigma = diag(lengthscales)
        """
        if self.flg_ARD:
            lengthscales = torch.exp(self.log_lengthscales_par)
        else:
            lengthscales = torch.exp(
                self.log_lengthscales_par * torch.ones(self.num_features, dtype=self.dtype, device=self.device)
            )
        # get dimensions and if X1=X2
        N1, D1 = X1.size()
        if X2 is None:
            flg_single_input = True
            N2 = N1
            D2 = D1
        else:
            flg_single_input = False
            N2, D2 = X2.size()
        # slice the inputs and get the weighted distances
        X1_sliced = X1[:, self.active_dims] / lengthscales
        X1_squared = torch.sum(X1_sliced.mul(X1_sliced), dim=1, keepdim=True)
        if flg_single_input:
            dist = (
                X1_squared
                + X1_squared.transpose(dim0=0, dim1=1)
                - 2 * torch.matmul(X1_sliced, X1_sliced.transpose(dim0=0, dim1=1))
            )
        else:
            X2_sliced = X2[:, self.active_dims] / lengthscales
            X2_squared = torch.sum(X2_sliced.mul(X2_sliced), dim=1, keepdim=True)
            dist = (
                X1_squared
                + X2_squared.transpose(dim0=0, dim1=1)
                - 2 * torch.matmul(X1_sliced, X2_sliced.transpose(dim0=0, dim1=1))
            )

        # print(X1_squared)
        # print(X1_squared.transpose(dim0=0, dim1=1))
        # print(X1_squared + X1_squared.transpose(dim0=0, dim1=1))
        # print(2*torch.matmul(X1_sliced,X1_sliced.transpose(dim0=0, dim1=1)))
        # print(dist)

        return dist


class RBF(Stationary_GP):
    """Implementation of the standard RBF GP with constant mean"""

    def __init__(
        self,
        active_dims,
        lengthscales_init=None,
        flg_train_lengthscales=True,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        lambda_init=None,
        flg_train_lambda=True,
        mean_init=None,
        flg_train_mean=False,
        name="",
        dtype=torch.float64,
        sigma_n_num=None,
        device=None,
    ):
        super(RBF, self).__init__(
            active_dims,
            lengthscales_init=lengthscales_init,
            flg_train_lengthscales=flg_train_lengthscales,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            sigma_n_num=sigma_n_num,
            device=device,
        )
        # set the scale parameter
        if lambda_init is None:
            lambda_init = np.ones(1)
        if lambda_init.size != 1:
            raise RuntimeError("Lambda must be a np array qith dimension 1")
        self.log_lambda_par = torch.nn.Parameter(
            torch.tensor(np.log(lambda_init), dtype=self.dtype, device=self.device), requires_grad=flg_train_lambda
        )
        # set the mean parameters
        if mean_init is None:
            mean_init = np.zeros(1)
        self.mean_par = torch.nn.Parameter(
            torch.tensor(mean_init, dtype=self.dtype, device=self.device), requires_grad=flg_train_mean
        )

    def get_mean(self, X):
        """Return constant mean"""
        N = X.size()[0]
        return self.mean_par.repeat(N, 1)

    def get_covariance(self, X1, X2=None, flg_noise=False):
        """Compute the exponential of the negative squared weighted distance"""
        if flg_noise & self.GP_with_noise:
            N = X1.size()[0]
            return torch.exp(self.log_lambda_par) * torch.exp(
                -self.get_weigted_distances(X1, X2)
            ) + self.get_sigma_n_2() * torch.eye(N, dtype=self.dtype, device=self.device)
        else:
            return torch.exp(self.log_lambda_par) * torch.exp(-self.get_weigted_distances(X1, X2))

    def get_diag_covariance(self, X, flg_noise=False):
        """Returns the vector containing the element along the diagonal of the covariance matrix"""
        N = X.size()[0]
        if flg_noise:
            return (
                torch.exp(self.log_lambda_par) * torch.ones(N, dtype=self.dtype, device=self.device)
                + self.get_sigma_n_2()
            )
        else:
            return torch.exp(self.log_lambda_par) * torch.ones(N, dtype=self.dtype, device=self.device)



# =====================================================================================
# APPEND THIS CLASS TO:  gpr_lib/GP_prior/Stationary_GP.py
# (numpy as np and torch are already imported at the top of that file)
# =====================================================================================


class Matern(Stationary_GP):
    """RVGP connection-Laplacian ('Matern on a manifold') kernel as an MC-PILCO GP prior.

    This is the degenerate / finite-rank kernel of Peach et al., ICLR 2024 (Eq. 15).
    It is NOT a function of ||x - x'||: the GP *input* handed to get_covariance is the
    precomputed positional encoding phi(x) (connection-Laplacian eigenvectors), of shape
    (N, k). The kernel is linear in phi with diagonal spectral weights S(lambda):

        S(lambda) = sigma_f * (lambda + 2*nu/kappa^2)^(-nu)          (matern)
        S(lambda) = sigma_f * exp(-0.5 * kappa^2 * lambda)           (se)
        k(x, x')  = phi(x) diag(S) phi(x')^T

    All manifold machinery (proximity graph, local-PCA singular values, connection
    Laplacian, eigendecomposition, Nystrom out-of-sample) is done upstream; here we only
    reweight fixed eigen-features. Inherits Stationary_GP purely for bookkeeping; the
    lengthscale machinery is unused and frozen.

    IMPORTANT: build this with sigma_n_init != None. The kernel block phi diag(S) phi^T is
    rank <= k (low-rank), so GP_prior.forward()'s Cholesky only succeeds because it adds
    sigma_n^2 * I via get_covariance(..., flg_noise=True).
    """

    def __init__(
        self,
        active_dims,          # np.arange(k): the k columns of the positional encoding
        eigenvalues,          # (k,) connection-Laplacian eigenvalues Lambda_c (frozen)
        num_vertices,         # scalar RVGP normalisation (absorbed by sigma_f in training)
        nu_init=1.5,
        kappa_init=5.0,
        sigma_f_init=1.0,
        typ="matern",
        flg_train_nu=False,
        flg_train_kappa=True,
        flg_train_sigma_f=True,
        mean_init=None,
        flg_train_mean=False,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        sigma_n_num=None,
        device=None,
    ):
        # Stationary_GP requires a (non-None) lengthscales_init; pass a dummy and freeze it.
        super(Matern, self).__init__(
            active_dims,
            lengthscales_init=np.ones(1),
            flg_train_lengthscales=False,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            sigma_n_num=sigma_n_num,
            device=device,
        )
        if typ not in ("matern", "se"):
            raise ValueError("typ must be 'matern' or 'se'")
        self.typ = typ

        # frozen spectrum / normalisation (registered so .to(device) moves them)
        self.register_buffer(
            "eigenvalues",
            torch.tensor(np.asarray(eigenvalues).reshape(-1), dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "num_vertices",
            torch.tensor(float(num_vertices), dtype=self.dtype, device=self.device),
        )

        # trainable positive hyperparameters (log-space)
        self.log_kappa_par = torch.nn.Parameter(
            torch.tensor(np.log(kappa_init), dtype=self.dtype, device=self.device),
            requires_grad=flg_train_kappa,
        )
        self.log_sigma_f_par = torch.nn.Parameter(
            torch.tensor(np.log(sigma_f_init), dtype=self.dtype, device=self.device),
            requires_grad=flg_train_sigma_f,
        )
        if typ == "matern":
            self.log_nu_par = torch.nn.Parameter(
                torch.tensor(np.log(nu_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_nu,
            )

        # constant mean (as in RBF)
        if mean_init is None:
            mean_init = np.zeros(1)
        self.mean_par = torch.nn.Parameter(
            torch.tensor(mean_init, dtype=self.dtype, device=self.device),
            requires_grad=flg_train_mean,
        )

    def eval_S(self):
        """RVGP spectral density S(lambda) (torch port of ManifoldKernel.eval_S)."""
        kappa = torch.exp(self.log_kappa_par)
        sigma_f = torch.exp(self.log_sigma_f_par)
        if self.typ == "matern":
            nu = torch.exp(self.log_nu_par)
            S = torch.pow(self.eigenvalues + 2.0 * nu / (kappa ** 2), -nu)
        else:  # 'se'
            S = torch.exp(-0.5 * self.eigenvalues * kappa ** 2)
        S = S * (self.num_vertices / torch.sum(S))
        S = S * sigma_f
        return S  # (k,)

    def get_mean(self, X):
        """Constant prior mean."""
        N = X.size()[0]
        return self.mean_par.repeat(N, 1)

    def get_covariance(self, X1, X2=None, flg_noise=False):
        """RVGP Eq. 15 block:  (phi1 * S) @ phi2^T   (+ sigma_n^2 I if requested)."""
        Xf1 = X1[:, self.active_dims]
        S = self.eval_S()
        if X2 is None:
            K = (Xf1 * S) @ Xf1.transpose(0, 1)
        else:
            Xf2 = X2[:, self.active_dims]
            K = (Xf1 * S) @ Xf2.transpose(0, 1)
        if flg_noise & self.GP_with_noise:
            N = X1.size()[0]
            K = K + self.get_sigma_n_2() * torch.eye(N, dtype=self.dtype, device=self.device)
        return K

    def get_diag_covariance(self, X, flg_noise=False):
        """Diagonal of the Eq. 15 block:  sum_l S_l * phi[:,l]^2."""
        Xf = X[:, self.active_dims]
        S = self.eval_S()
        diag = torch.sum((Xf * S) * Xf, dim=1)
        if flg_noise & self.GP_with_noise:
            diag = diag + self.get_sigma_n_2()
        return diag
