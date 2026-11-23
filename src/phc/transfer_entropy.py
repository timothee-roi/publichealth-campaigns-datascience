"""Transfer entropy between two daily series, estimated with nearest neighbours.

Transfer entropy (Schreiber 2000, Phys. Rev. Lett. 85, 461) from a cause a to an effect y is
the conditional mutual information between the effect today and the past of the cause,
given the past of the effect:

    TE(a -> y) = I( y_t ; a_{t-1..t-m} | y_{t-1..t-m} ).

It is zero if and only if y_t is independent of the past of a given the past of y, with no
assumption on the form of the dependence. For jointly Gaussian variables it equals half the
log of the variance ratio of the restricted and unrestricted linear models, i.e. linear
Granger causality (Barnett, Barrett & Seth 2009, Phys. Rev. Lett. 103, 238701). It is
therefore the model-free counterpart of the Granger test, and is widely used to study
information flow between social, media and financial series.

Estimator: Kraskov-Stoegbauer-Grassberger (2004, Phys. Rev. E 69, 066138), algorithm 1, in
the conditional form of Frenzel & Pompe (2007, Phys. Rev. Lett. 99, 204101), with the
maximum norm:

    I = psi(k) - < psi(n_ac + 1) + psi(n_bc + 1) - psi(n_c + 1) >,

where eps_i is the distance from point i to its k-th neighbour in the joint space (a, b, c),
and n_ac, n_bc, n_c count the points strictly closer than eps_i in the subspaces (a, c),
(b, c) and c. a = y_t, b = past of the cause, c = past of the effect.

Estimates carry a bias that depends on the dimension, the sample size and k, and can be
negative. The statistic is therefore not read on its own: it is compared with its value on
season-swap surrogates (kernel_granger.py), which share the same bias.

Ties (many articles counts and search indices are repeated integers) are broken by adding
noise of standard deviation 1e-8 to the standardised series, as recommended by KSG.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

JITTER = 1e-8


class KSGTE:
    """Transfer entropy of a block ``B`` (past of the cause) towards ``y`` given ``X``.

    The trees of the subspaces that do not contain the cause (a, c) and c are built once;
    ``stat`` can then be called on the real cause and on every surrogate.
    """

    def __init__(self, y, X, k: int = 5, seed: int = 0):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.n = len(y)
        self.a = (np.asarray(y, float) + JITTER * self.rng.standard_normal(self.n))[:, None]
        self.c = np.asarray(X, float) + JITTER * self.rng.standard_normal(np.shape(X))
        self.ac = np.hstack([self.a, self.c])
        self.tree_ac = cKDTree(self.ac)
        self.tree_c = cKDTree(self.c)

    @staticmethod
    def _count(tree, pts, eps):
        # points strictly inside eps (the point itself is included in the count)
        return tree.query_ball_point(pts, np.nextafter(eps, 0), p=np.inf, return_length=True)

    def stat(self, B) -> dict:
        B = np.asarray(B, float)
        B = B + JITTER * self.rng.standard_normal(B.shape)
        bc = np.hstack([B, self.c])
        joint = np.hstack([self.a, B, self.c])
        eps = cKDTree(joint).query(joint, k=self.k + 1, p=np.inf)[0][:, -1]
        n_ac = self._count(self.tree_ac, self.ac, eps)       # includes the point itself,
        n_bc = self._count(cKDTree(bc), bc, eps)             # so "n + 1" of the formula
        n_c = self._count(self.tree_c, self.c, eps)          # equals the count returned
        te = digamma(self.k) - np.mean(digamma(n_ac) + digamma(n_bc) - digamma(n_c))
        return {"delta": float(te)}
