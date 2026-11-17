"""Response-space exploration, deliberately separate from predictor training."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from threadpoolctl import threadpool_limits


def response_table(raw, responses, group_columns, repetition=None):
    """One equally weighted row per configuration; no missing-value imputation.

    Supply factor columns as group_columns to merge identical input vectors, or
    alternative ID columns when inputs are unavailable. Counts may be unequal.
    """
    responses, group_columns = list(responses), list(group_columns)
    if not responses or not group_columns or set(responses)&set(group_columns):
        raise ValueError('Specify distinct response and grouping columns')
    if len(set(responses))!=len(responses) or len(set(group_columns))!=len(group_columns):
        raise ValueError('Column selections must be unique')
    required=list(dict.fromkeys(group_columns+responses+([repetition] if repetition else [])))
    if not set(required)<=set(raw): raise ValueError('Missing configured columns')
    if raw.empty or raw[required].isna().any().any(): raise ValueError('Empty table or missing values; resolve explicitly')
    values=raw[responses].to_numpy(float)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite response values')
    if repetition and raw.duplicated(group_columns+[repetition]).any():
        raise ValueError('Duplicate configuration/repetition keys')
    frame=raw[group_columns].copy()
    frame[responses]=values
    grouped=frame.groupby(group_columns,sort=True,dropna=False)
    means=grouped[responses].mean()
    counts=grouped.size().rename('repetitions')
    variances=grouped[responses].var()
    return dict(means=means, counts=counts, variances=variances,
                mean_variances=variances.div(counts,axis=0))


def explore_responses(means, max_clusters=6, seed=2026):
    """Standardized-response PCA and exploratory clustering; no natural groups assumed."""
    if len(means)<3: raise ValueError('Exploration requires at least three configurations')
    if not np.isfinite(means.to_numpy(float)).all(): raise ValueError('Finite complete means required')
    constant=means.columns[means.nunique()<=1].tolist()
    active=means.drop(columns=constant)
    if active.shape[1]<2: raise ValueError('At least two varying responses required for relationships')
    scaler=StandardScaler().fit(active)
    Z=scaler.transform(active)
    pca=PCA().fit(Z)
    components=[f'PC{i+1}' for i in range(pca.n_components_)]
    scores=pd.DataFrame(pca.transform(Z),index=means.index,columns=components)
    axes=pd.DataFrame(pca.components_.T,index=active.columns,columns=components)
    rows=[];models={};stability=[]
    with threadpool_limits(limits=1):
        for k in range(2,min(max_clusters,len(means)-1,len(np.unique(Z,axis=0)))+1):
            model=KMeans(n_clusters=k,n_init=20,random_state=seed).fit(Z)
            labels=model.labels_
            if not 1<len(np.unique(labels))<len(Z): continue
            sizes=np.bincount(labels)
            rows.append(dict(clusters=k,silhouette=silhouette_score(Z,labels),
                             smallest_cluster=int(sizes.min()),inertia=model.inertia_))
            models[k]=model
            # Initialization stability, not bootstrap evidence for real populations.
            for delta in (1,2,3):
                other=KMeans(n_clusters=k,n_init=10,random_state=seed+delta).fit_predict(Z)
                stability.append(dict(clusters=k,seed=seed+delta,adjusted_rand=adjusted_rand_score(labels,other)))
    candidates=pd.DataFrame(rows,columns=['clusters','silhouette','smallest_cluster','inertia'])
    return dict(active=active,constant_responses=constant,scaler=scaler,standardized=Z,
        pca=pca,scores=scores,axes=axes,pearson=active.corr(),spearman=active.corr(method='spearman'),
        candidates=candidates,models=models,stability=pd.DataFrame(stability))


def describe_clusters(result, k):
    """Caller explicitly selects a descriptive partition; no automatic winner."""
    model=result['models'][k]
    labels=pd.Series(model.labels_,index=result['active'].index,name='cluster')
    original=result['active'].groupby(labels).mean()
    standard=pd.DataFrame(result['standardized'],index=labels.index,columns=result['active'].columns).groupby(labels).mean()
    return dict(labels=labels,counts=labels.value_counts().sort_index(),means=original,standardized_means=standard)


def _active_frame(means):
    if not means.columns.is_unique or len(means) < 4:
        raise ValueError('Unique response columns and at least four configurations required')
    if not np.isfinite(means.to_numpy(float)).all():
        raise ValueError('Finite complete response means required')
    active = means.loc[:, means.nunique() > 1].astype(float)
    if active.shape[1] < 2:
        raise ValueError('At least two varying responses required')
    return active


def _centered_distances(x):
    d = np.abs(x[:, None] - x[None, :])
    return d - d.mean(axis=0) - d.mean(axis=1)[:, None] + d.mean()


def _distance_correlation(a, b):
    denominator = np.sqrt(np.mean(a*a) * np.mean(b*b))
    return np.sqrt(max(0., np.mean(a*b)) / denominator) if denominator > 0 else np.nan


def relationship_diagnostics(means, permutations=199, bootstrap=200, seed=2026):
    """Pairwise nonlinear dependence, BH-adjusted permutation tests and Pearson stability.

    Resampling configurations treats the empirical DOE as exchangeable. Bootstrap
    ranges describe design-composition sensitivity, not simulator mean uncertainty.
    Distance correlation is the biased, nonnegative sample estimator.
    """
    from itertools import combinations
    from scipy.stats import spearmanr
    active = _active_frame(means)
    if permutations < 1 or bootstrap < 2:
        raise ValueError('Use at least one permutation and two bootstrap samples')
    rng = np.random.default_rng(seed)
    n = len(active)
    distances = {c: _centered_distances(active[c].to_numpy()) for c in active}
    rows = []
    for left, right in combinations(active.columns, 2):
        x, y = active[left].to_numpy(), active[right].to_numpy()
        a, b = distances[left], distances[right]
        observed = _distance_correlation(a, b)
        null = []
        for _ in range(permutations):
            order = rng.permutation(n)
            null.append(_distance_correlation(a, b[np.ix_(order, order)]))
        correlations = []
        for _ in range(bootstrap):
            ix = rng.integers(n, size=n)
            if np.std(x[ix]) > 0 and np.std(y[ix]) > 0:
                correlations.append(np.corrcoef(x[ix], y[ix])[0, 1])
        lo, hi = np.quantile(correlations, [.025, .975]) if correlations else (np.nan, np.nan)
        rows.append(dict(response_a=left, response_b=right,
                         pearson=np.corrcoef(x, y)[0, 1], spearman=spearmanr(x, y).statistic,
                         distance_correlation=observed,
                         permutation_p=(1+np.sum(np.asarray(null)>=observed))/(permutations+1),
                         pearson_bootstrap_low=lo, pearson_bootstrap_high=hi,
                         valid_bootstraps=len(correlations)))
    pairs = pd.DataFrame(rows)
    order = np.argsort(pairs.permutation_p.to_numpy())
    ranked = pairs.permutation_p.to_numpy()[order] * len(pairs) / np.arange(1, len(pairs)+1)
    adjusted = np.empty(len(pairs))
    adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    pairs['bh_q'] = adjusted
    return pairs.sort_values('distance_correlation', ascending=False).reset_index(drop=True)


def partial_relationships(means):
    """Shrinkage partial correlations; regularized linear associations, not causal edges."""
    from sklearn.covariance import LedoitWolf
    active = _active_frame(means)
    z = StandardScaler().fit_transform(active)
    model = LedoitWolf().fit(z)
    precision = model.precision_
    matrix = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    np.fill_diagonal(matrix, 1.)
    return dict(partial=pd.DataFrame(matrix, index=active.columns, columns=active.columns),
                shrinkage=float(model.shrinkage_))


def within_configuration_relationships(raw, responses, group_columns, repetition=None):
    """Equal-configuration average of paired within-run covariance matrices.

    Only replicated configurations contribute. Their sample covariance is averaged
    before conversion to correlation; this differs from pooling all residual rows.
    """
    table = response_table(raw, responses, group_columns, repetition)
    covariances = [g[list(responses)].astype(float).cov().to_numpy()
                   for _, g in raw.groupby(list(group_columns), sort=True, dropna=False)
                   if len(g) >= 2]
    if not covariances:
        return dict(correlation=pd.DataFrame(np.nan, index=responses, columns=responses),
                    replicated_configurations=0)
    covariance = np.mean(covariances, axis=0)
    denominator = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    corr = np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator>0)
    return dict(correlation=pd.DataFrame(corr, index=responses, columns=responses),
                replicated_configurations=len(covariances))


def compare_response_structures(means, k=3, seed=2026, repeats=12, fraction=.8,
                                dbscan_eps=.8, min_samples=5, gamma=None):
    """Compare full-space partitions and subsampling stability; RBF embedding for display.

    All estimators/scalers are refitted on each subset. ARI compares subset labels
    against the corresponding full-data labels. DBSCAN noise remains labeled -1.
    """
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import KernelPCA
    from sklearn.metrics import pairwise_distances
    active = _active_frame(means)
    n = len(active)
    size = int(np.floor(n*fraction))
    if not 2 <= k < min(n, size) or not 0 < fraction < 1 or repeats < 1:
        raise ValueError('Require 2 <= k < subsample size < sample size and positive repeats')
    if dbscan_eps <= 0 or min_samples < 2:
        raise ValueError('DBSCAN eps must be positive and min_samples at least two')
    z = StandardScaler().fit_transform(active)
    def factories():
        return {'kmeans': KMeans(n_clusters=k, n_init=20, random_state=seed),
                'ward': AgglomerativeClustering(n_clusters=k, linkage='ward'),
                'gmm': GaussianMixture(n_components=k, n_init=3, reg_covar=1e-5, random_state=seed),
                'dbscan': DBSCAN(eps=dbscan_eps, min_samples=min_samples)}
    labels, rows, stability = {}, [], []
    rng = np.random.default_rng(seed)
    subsets = [rng.choice(n, size=size, replace=False) for _ in range(repeats)]
    with threadpool_limits(limits=1):
        for name, model in factories().items():
            lab = model.fit_predict(z)
            labels[name] = lab
            included = lab >= 0
            groups = np.unique(lab[included])
            score = silhouette_score(z[included], lab[included]) if 1 < len(groups) < included.sum() else np.nan
            rows.append(dict(method=name, groups=len(groups), noise_fraction=float((~included).mean()),
                             silhouette_excluding_noise=score,
                             smallest_group=int(pd.Series(lab[included]).value_counts().min()) if included.any() else 0,
                             bic=model.bic(z) if name=='gmm' else np.nan))
            for trial, ix in enumerate(subsets):
                local = StandardScaler().fit_transform(active.iloc[ix])
                sublabels = factories()[name].fit_predict(local)
                valid = len(np.unique(lab[ix][lab[ix]>=0])) >= 2 and len(np.unique(sublabels[sublabels>=0])) >= 2
                stability.append(dict(method=name, trial=trial,
                    adjusted_rand=adjusted_rand_score(lab[ix], sublabels) if valid else np.nan))
        embedding = KernelPCA(n_components=2, kernel='rbf', gamma=gamma or 1/active.shape[1]).fit_transform(z)
    # Distance rank agreement is only a display distortion diagnostic.
    from scipy.stats import spearmanr
    upper = np.triu_indices(n, 1)
    agreement = spearmanr(pairwise_distances(z)[upper], pairwise_distances(embedding)[upper]).statistic
    return dict(comparison=pd.DataFrame(rows), labels=pd.DataFrame(labels, index=active.index),
                stability=pd.DataFrame(stability),
                embedding=pd.DataFrame(embedding, index=active.index, columns=['RBF1','RBF2']),
                distance_rank_agreement=float(agreement))


def pareto_profiles(means, directions):
    """Observed nondominance only. Equal profiles do not dominate one another."""
    if not directions or not set(directions) <= set(means.columns):
        raise ValueError('Supply objective columns present in means')
    if any(v not in ('min', 'max') for v in directions.values()):
        raise ValueError('Directions must be min or max')
    values = means[list(directions)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Finite objectives required')
    signed = values * np.array([1 if v=='max' else -1 for v in directions.values()])
    dominators = np.array([np.sum(np.all(signed >= row, axis=1) & np.any(signed > row, axis=1)) for row in signed])
    return pd.DataFrame({'dominated_by':dominators, 'nondominated':dominators==0}, index=means.index)
