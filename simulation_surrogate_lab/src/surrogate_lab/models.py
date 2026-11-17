"""Fresh estimator factories and isolated training/prediction functions."""
from __future__ import annotations
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from .data import summarize_runs, training_mean_noise
from .features import fit_features, transform_features


def make_estimator(recipe, noise, cfg):
    p = dict(recipe.params)
    family = recipe.family
    if family == "ridge": return Ridge(**p)
    if family == "elastic": return ElasticNet(max_iter=10000, random_state=recipe.seed, **p)
    if family == "kernel_ridge": return KernelRidge(kernel="rbf", **p)
    if family == "gam":
        from importlib.metadata import version
        if tuple(int(x) for x in version("pygam").split('.')[:2]) < (0, 10):
            raise ImportError("pyGAM >=0.10 required for modern SciPy compatibility")
        from pygam import LinearGAM
        return LinearGAM(**p)
    if family == "ebm":
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor(n_jobs=1, random_state=recipe.seed, **p)
    if family == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_jobs=1, random_state=recipe.seed, verbosity=-1, **p)
    if family == "forest": return RandomForestRegressor(n_estimators=120, n_jobs=1, random_state=recipe.seed, **p)
    if family == "extra": return ExtraTreesRegressor(n_estimators=120, n_jobs=1, random_state=recipe.seed, **p)
    if family == "boost": return HistGradientBoostingRegressor(early_stopping=False, random_state=recipe.seed, **p)
    if family == "svr": return SVR(**p)
    if family == "mlp": return MLPRegressor(early_stopping=False, max_iter=600, random_state=recipe.seed, **p)
    if family == "gp":
        length = p.pop("length_scale", 1.)
        if p.pop("ard", False): length = np.full(len(cfg.factors), length)
        kind, nu = p.pop("kernel", "matern"), p.pop("nu", 2.5)
        if kind not in ("matern", "rbf"): raise ValueError("Unknown GP kernel")
        kernel = ConstantKernel(1., (.05, 100.)) * (RBF(length) if kind == "rbf" else Matern(length, nu=nu))
        if noise is None:
            kernel += WhiteKernel(noise_level=.1, noise_level_bounds=(1e-8, 10.))
        return GaussianProcessRegressor(kernel=kernel, alpha=1e-8 if noise is None else np.maximum(noise, 1e-8), normalize_y=False,
                                        n_restarts_optimizer=0, random_state=recipe.seed, **p)
    if family == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_jobs=1, random_state=recipe.seed, objective="reg:squarederror", **p)
    raise ValueError(family)


def fit_recipe(train, recipe, cfg):
    if recipe.training not in ("mean", "run"): raise ValueError("Unknown training grain")
    if recipe.family == "gp" and dict(recipe.params).get("ard") and recipe.representation != "raw":
        raise ValueError("ARD GP requires raw factors")
    s = summarize_runs(train, cfg)
    if recipe.family == "gp" and len(s) > cfg.max_exact_gp_alternatives:
        raise ValueError("Exact GP alternative cap exceeded")
    Xmean = s[list(cfg.factors)].to_numpy(float)
    features = fit_features(Xmean, s["mean"].to_numpy(), recipe.representation, cfg, recipe.top_pairs)
    scaler = StandardScaler().fit(transform_features(Xmean, features, cfg))
    offset, scale = float(s["mean"].mean()), max(float(s["mean"].std(ddof=0)), 1e-8)
    if recipe.training == "run":
        if recipe.family not in ("ridge", "forest", "extra", "boost", "xgb"):
            raise ValueError("This family does not support weighted-run fitting")
        X = train[list(cfg.factors)].to_numpy(float)
        y = train[cfg.response].to_numpy(float)
        weights = 1/train.groupby("group")["group"].transform("size").to_numpy()
    else:
        X, y, weights = Xmean, s["mean"].to_numpy(), None
    mean_noise = training_mean_noise(s)
    model = make_estimator(recipe, None if mean_noise is None else mean_noise/scale**2, cfg)
    noise_method = ("learned_shared_noise" if mean_noise is None else
                    "training_pooled_for_singletons" if (s.reps == 1).any() else "replicate_variances")
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = {} if weights is None else {"sample_weight": weights}
        model.fit(scaler.transform(transform_features(X, features, cfg)), (y-offset)/scale, **options)
    noise_notices = []
    if recipe.family == "gp" and noise_method != "replicate_variances":
        noise_notices = [f"GP noise method: {noise_method}; singleton local variances are not observed"]
    return {"recipe": recipe, "features": features, "scaler": scaler, "model": model,
            "offset": offset, "scale": scale, "noise_method": noise_method, "warnings": tuple(sorted(set([str(w.message) for w in caught] + noise_notices)))}


def predict_fit(fitted, X, cfg):
    Z = fitted["scaler"].transform(transform_features(X, fitted["features"], cfg))
    return fitted["offset"] + fitted["scale"]*fitted["model"].predict(Z)


def predict_gp_mean_sd(fitted, X, cfg):
    if fitted["recipe"].family != "gp": raise ValueError("Native mean SD is only available for GP")
    Z = fitted["scaler"].transform(transform_features(X, fitted["features"], cfg))
    _, std = fitted["model"].predict(Z, return_std=True)
    if fitted.get("noise_method") == "learned_shared_noise":
        # sklearn includes WhiteKernel noise in predictive variance; remove it
        # here because this helper explicitly returns latent-mean SD.
        std = np.sqrt(np.maximum(std**2-fitted["model"].kernel_.k2.noise_level, 0.))
    return fitted["scale"]*std


def fit_bundle(data, recipes, cfg):
    return tuple(fit_recipe(data, recipe, cfg) for recipe in recipes)


def predict_bundle(bundle, X, cfg):
    return np.mean([predict_fit(f, X, cfg) for f in bundle], axis=0)
