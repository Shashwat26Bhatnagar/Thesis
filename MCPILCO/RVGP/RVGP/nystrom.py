#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Out-of-sample extension of connection-Laplacian eigenvectors (the RVGP positional
encoding U_c) to new points, via the Nystrom scheme of

    Singer & Wu, "Vector Diffusion Maps and the Connection Laplacian",
    Comm. Pure Appl. Math. 2012, Section 7.

For a training point the eigenvector of the (random-walk) connection-Laplacian obeys

    v_l(i) = (1/mu_l) * [ sum_j K(||x_i - x_j||) O_ij v_l(j) ] / [ sum_j K(||x_i - x_j||) ]   (Eq. 7.3)

The Nystrom extension keeps the training eigenvectors {v_l} and eigenvalues fixed and
evaluates the same right-hand side at a NEW point y, giving v_l(y). Concatenating
v_1(y),...,v_k(y) column-wise yields (U_c)_y, which mapped through the local frame T_y
is the positional encoding P_y = T_y (U_c)_y that the trained GP consumes.

Verified property (see module test): if the neighbour set / weights of y match the
graph, the extension reproduces the training encoding to machine precision, and the
ambient result is invariant to the (arbitrary) choice of local frame T_y in O(m).
"""

import numpy as np
from scipy.spatial import cKDTree


def _local_pca_frame(y, nbr_pos, m, weights=None):
    """Local-PCA tangent frame at y: top-m left singular vectors of the (weighted)
    matrix of neighbour edge vectors (x_j - y). Returns T_y of shape (p, m).

    Note: the sign/rotation of this frame is arbitrary; the final ambient encoding
    P_y = T_y (U_c)_y is invariant to it, so it need not match ptu_dijkstra's frame."""
    E = np.asarray(nbr_pos, float) - np.asarray(y, float)      # (nnbr, p)
    if weights is not None:
        E = E * np.sqrt(np.asarray(weights, float))[:, None]
    U, _, _ = np.linalg.svd(E.T, full_matrices=False)          # columns of U span T_yM
    return U[:, :m]


def _align(Ti, Tj):
    """Closest orthogonal matrix to Ti^T Tj (Kabsch / polar), i.e. the O(m) parallel
    transport O_{i<-j} : T_jM -> T_iM.  (Singer-Wu Eq. 2.4)."""
    U, _, Vt = np.linalg.svd(Ti.T @ Tj)
    return U @ Vt


def nystrom_extend(data, X_new,
                   n_neighbors=None,
                   kernel='knn',
                   sigma=None,
                   normalization=None,
                   eps=1e-8,
                   return_tangent=False):
    """Extend the connection-Laplacian eigenvectors of a fitted RVGP ``data`` object
    to new ambient points ``X_new`` via Nystrom.

    Parameters
    ----------
    data : RVGP data object
        Must expose ``vertices`` (n,p), ``gauges`` (n,p,m), ``evecs_Lc`` (n*p,k),
        ``evals_Lc`` (k,).  These are exactly what ``RVGP.create_data_object`` sets.
    X_new : array (n_new, p)
        New ambient coordinates to extend to.
    n_neighbors : int, optional
        Neighbours used both for the local-PCA frame and the Nystrom average.
        Should match the graph used at fit time (``manifold_graph`` n_neighbors).
        Defaults to the mean graph degree of the training object.
    kernel : {'knn','gaussian'}
        Edge weights K(||y - x_j||). 'knn' = binary (matches manifold_graph's default
        connectivity graph); 'gaussian' = exp(-d^2/2 sigma^2) (matches 'affinity' mode).
    sigma : float, optional
        Gaussian bandwidth; defaults to the median neighbour distance per point.
    normalization : {None,'rw'}
        Must match how the connection Laplacian was built in ``compute_connection_laplacian``.
        None  -> unnormalised  L_c = D - S,   denominator (deg(y) - lambda_l).
        'rw'  -> random-walk    L_c = I - D^{-1}S, denominator deg(y)*(1 - lambda_l).
    eps : float
        Floor on |denominator| for numerical safety.
    return_tangent : bool
        If True also return the tangent-coordinate blocks (U_c)_y of shape (n_new,m,k).

    Returns
    -------
    evecs_Lc_new : array (n_new, p, k)
        Ambient positional encoding P_y = T_y (U_c)_y, in the SAME layout as
        ``data.evecs_Lc.reshape(n, p, k)``. Reshape to (n_new*p, k) to feed a GP.
    """
    X_new = np.atleast_2d(np.asarray(X_new, float))
    Xtr = np.asarray(data.vertices, float)
    gauges = np.asarray(data.gauges, float)                 # (n, p, m)
    n, p, m = gauges.shape
    k = int(np.asarray(data.evals_Lc).reshape(-1).shape[0])
    lam = np.asarray(data.evals_Lc, float).reshape(-1)      # (k,)

    # Recover the raw tangent-coordinate eigenvector blocks (U_c)_i = T_i^T P_i.
    # (data.evecs_Lc stores the ambient P_i = T_i (U_c)_i; T_i has orthonormal columns.)
    P = np.asarray(data.evecs_Lc, float).reshape(n, p, k)
    Uc_raw = np.einsum('npm,npk->nmk', gauges, P)           # (n, m, k)

    if n_neighbors is None:
        degs = np.array([d for _, d in data.G.degree()]) if hasattr(data, 'G') else None
        n_neighbors = int(round(degs.mean())) if degs is not None else 2 * m + 1

    tree = cKDTree(Xtr)
    out = np.zeros((len(X_new), p, k))
    out_tan = np.zeros((len(X_new), m, k))

    for a, y in enumerate(X_new):
        d, nbr = tree.query(y, k=n_neighbors + 1)           # +1 in case y hits a vertex
        d = np.atleast_1d(d); nbr = np.atleast_1d(nbr)
        keep = d > 1e-12                                     # drop self if y == some x_j
        if keep.sum() >= n_neighbors:
            d, nbr = d[keep][:n_neighbors], nbr[keep][:n_neighbors]
        else:
            d, nbr = d[:n_neighbors], nbr[:n_neighbors]

        if kernel == 'knn':
            w = np.ones(len(nbr))
        elif kernel == 'gaussian':
            s = sigma if sigma is not None else max(np.median(d), 1e-12)
            w = np.exp(-d ** 2 / (2.0 * s ** 2))
        else:
            raise ValueError("kernel must be 'knn' or 'gaussian'")

        Ty = _local_pca_frame(y, Xtr[nbr], m,
                              weights=w if kernel == 'gaussian' else None)

        acc = np.zeros((m, k))
        for wj, j in zip(w, nbr):
            acc += wj * (_align(Ty, gauges[j]) @ Uc_raw[j])  # (m,m)@(m,k)

        deg = w.sum()
        denom = deg * (1.0 - lam) if normalization == 'rw' else (deg - lam)
        denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps * (denom == 0), denom)

        Uc_y = acc / denom[None, :]                          # (m, k)
        out_tan[a] = Uc_y
        out[a] = Ty @ Uc_y                                   # (p, k)

    return (out, out_tan) if return_tangent else out


def predict_out_of_sample(gp, data, X_new, **kwargs):
    """Convenience wrapper: Nystrom-extend then run a trained RVGP GP at new points.

    Returns (mean, var) each of shape (n_new, p)."""
    P_new = nystrom_extend(data, X_new, **kwargs)            # (n_new, p, k)
    n_new, p, k = P_new.shape
    Xin = P_new.reshape(n_new * p, k)                        # same layout as training input
    mean, var = gp.predict_f(Xin)
    return mean.numpy().reshape(n_new, p), var.numpy().reshape(n_new, p)
