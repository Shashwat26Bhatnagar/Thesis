#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================================
# NEW FILE:  model_learning/rvgp_manifold.py
#
# Thin abstraction that hides ALL RVGP geometry behind two calls:
#   train_features()  -> exact positional encodings for the training points
#   encode(X_new)     -> Nystrom positional encodings for new points
#
# Requires the RVGP repo importable (git clone) together with our nystrom.py:
#   from RVGP.dataclass import data          # builds graph -> local PCA -> Lc -> spectrum
#   from RVGP.nystrom   import nystrom_extend # out-of-sample extension (added earlier)
#
# ARCHER2 (module-based) setup:
#   module load PrgEnv-cray cpe-clang python
#   python -c "import RVGP.dataclass; print('ok')"  # verify ptu_dijkstra is built
# =====================================================================================

import sys
import numpy as np
import torch

# ---- Archer2 workaround: add outer RVGP repo to path if not installed ----
_rvgp_path = '/work/m25oc/m25oc/s2892016/project/Thesis/MC-PILCO/RVGP'  # adjust to your clone location
if _rvgp_path not in sys.path:
    sys.path.insert(0, _rvgp_path)


class RVGPManifold:
    """Build the connection-Laplacian manifold ONCE from training inputs, then expose
    positional encodings. All singular values / graph / Laplacian complexity stays here.

    The manifold ambient space is the full GP input (state+action), dim = p. The vector
    field we regress is the D-dimensional state-delta, so we keep only the first D rows
    (the state coordinates) of each p x k encoding. Encodings are stacked POINT-MAJOR:
    row index = point*D + component, matching a delta target reshaped with .reshape(-1, 1).
    """

    def __init__(
        self,
        X,                       # (N, p) training GP inputs (state,input), numpy
        n_out_dims,              # D: number of state (output) components to keep
        n_neighbors=10,
        frac_geodesic_neighbours=1.5,
        explained_variance=0.8,
        n_eigenpairs=50,
        nystrom_neighbors=None,  # neighbours for out-of-sample encode (default n_neighbors)
        nystrom_kernel="knn",
        nystrom_normalization=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        from RVGP.dataclass import data as RVGPData
        from RVGP.nystrom import nystrom_extend

        self._nystrom_extend = nystrom_extend
        self.dtype = dtype
        self.device = device
        self.D = int(n_out_dims)
        self.nystrom_neighbors = nystrom_neighbors or n_neighbors
        self.nystrom_kernel = nystrom_kernel
        self.nystrom_normalization = nystrom_normalization

        X = np.asarray(X, dtype=np.float64)
        self.p = X.shape[1]  # ambient dim = state + input
        if self.D > self.p:
            raise ValueError("n_out_dims (D) cannot exceed input dimension p")

        # ---- the ONE heavy call: graph -> tangent frames -> Lc -> eigendecomposition ----
        self.data = RVGPData(
            vertices=X,
            n_neighbors=n_neighbors,
            frac_geodesic_neighbours=frac_geodesic_neighbours,
            explained_variance=explained_variance,
            n_eigenpairs=n_eigenpairs,
        )

        self.k = int(np.asarray(self.data.evals_Lc).reshape(-1).shape[0])
        self.N = self.data.n

        # frozen kernel ingredients handed to the Matern GP
        self.eigenvalues = np.asarray(self.data.evals_Lc).reshape(-1)          # (k,)
        # normalisation constant (absorbed by trainable sigma_f; use #feature rows)
        self.num_vertices = float(self.N * self.D)

        # exact training encodings, sliced to the D state components, point-major
        P = np.asarray(self.data.evecs_Lc).reshape(self.N, self.p, self.k)      # (N, p, k)
        self._P_train = P[:, : self.D, :].reshape(self.N * self.D, self.k)      # (N*D, k)

    # ------------------------------------------------------------------ features
    def train_features(self):
        """Exact positional encodings of the training points: (N*D, k) torch tensor."""
        return torch.tensor(self._P_train, dtype=self.dtype, device=self.device)

    def encode(self, X_new):
        """Nystrom positional encodings of new points X_new (M, p) -> (M*D, k) torch."""
        if torch.is_tensor(X_new):
            X_np = X_new.detach().cpu().numpy()
        else:
            X_np = np.asarray(X_new, dtype=np.float64)
        P = self._nystrom_extend(
            self.data,
            X_np,
            n_neighbors=self.nystrom_neighbors,
            kernel=self.nystrom_kernel,
            normalization=self.nystrom_normalization,
        )                                                   # (M, p, k)
        M = P.shape[0]
        P = P[:, : self.D, :].reshape(M * self.D, self.k)   # (M*D, k), point-major
        return torch.tensor(P, dtype=self.dtype, device=self.device)
