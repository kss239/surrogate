"""Fit feature representations on training alternatives and apply them without refitting."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import Ridge
from .data import scale_bounds


def domain_matrix(Z, physical, cfg):
    derived = []
    for operation, j, k in cfg.domain_features:
        a, b = physical[:, j], physical[:, k]
        if operation == "ratio":
            if np.any(np.abs(b) <= cfg.denominator_floor): raise ValueError("Unsafe ratio denominator")
            value = a/b
        elif operation == "difference": value = a-b
        elif operation == "product": value = a*b
        else: raise ValueError(f"Unknown domain operation: {operation}")
        derived.append(value)
    return np.column_stack([Z, *derived]) if derived else Z


def fit_features(X, y, representation, cfg, top_pairs):
    Z = scale_bounds(X, cfg)
    if representation == "quadratic":
        basis = PolynomialFeatures(2, include_bias=False).fit(Z)
        state = {"kind": representation, "basis": basis}
    elif representation == "spline":
        basis = SplineTransformer(n_knots=4, degree=3, include_bias=False).fit(Z)
        state = {"kind": representation, "basis": basis}
    elif representation == "selected" and Z.shape[1] < 2:
        state = {"kind": representation, "pairs": (), "ranking": ()}
    elif representation == "selected":
        pilot_basis = SplineTransformer(n_knots=4, degree=3, include_bias=False).fit(Z)
        additive = pilot_basis.transform(Z)
        pilot = Ridge(alpha=1.).fit(additive, y)
        residual = y-pilot.predict(additive)
        pairs = tuple(combinations(range(Z.shape[1]), 2))
        products = np.column_stack([(Z[:, j]-.5)*(Z[:, k]-.5) for j, k in pairs])
        centered = products-products.mean(axis=0)
        scores = np.abs(centered.T @ residual)/(np.linalg.norm(centered, axis=0)+1e-12)
        selected = tuple(pairs[i] for i in np.argsort(-scores, kind="stable")[:top_pairs])
        state = {"kind": representation, "pairs": selected, "ranking": tuple((pairs[i], float(scores[i])) for i in np.argsort(-scores))}
    else:
        state = {"kind": representation}
    return state


def transform_features(X, state, cfg):
    X = np.asarray(X, float)
    Z = scale_bounds(X, cfg)
    kind = state["kind"]
    if kind in ("quadratic", "spline"): return state["basis"].transform(Z)
    if kind == "selected":
        return np.column_stack([Z, *[(Z[:, j]-.5)*(Z[:, k]-.5) for j, k in state["pairs"]]])
    if kind == "domain": return domain_matrix(Z, X, cfg)
    if kind != "raw": raise ValueError(kind)
    return Z
