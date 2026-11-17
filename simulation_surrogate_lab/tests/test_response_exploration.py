import numpy as np
import pandas as pd
import pytest
from surrogate_lab.response_exploration import response_table, explore_responses, describe_clusters


def test_unbalanced_means_and_unknown_variance():
    raw=pd.DataFrame({'id':[1,1,2,3,3,3],'run':[1,2,1,1,2,3],
                      'a':[1,3,5,7,8,9],'b':[2,6,10,14,16,18]})
    table=response_table(raw,['a','b'],['id'],'run')
    np.testing.assert_allclose(table['means'].a,[2,5,8])
    assert np.isnan(table['variances'].loc[2,'a'])
    result=explore_responses(table['means'])
    assert result['pearson'].loc['a','b']==pytest.approx(1)
    assert result['pca'].explained_variance_ratio_[0]==pytest.approx(1)
    assert describe_clusters(result,2)['counts'].sum()==3


def test_scale_invariance_and_constants():
    rng=np.random.default_rng(2)
    means=pd.DataFrame(rng.normal(size=(20,3)),columns=['a','b','c']).assign(constant=2)
    a=explore_responses(means)
    b=explore_responses(means.assign(a=means.a*1000))
    np.testing.assert_allclose(a['standardized'],b['standardized'])
    assert a['constant_responses']==['constant']
    assert a['models'][2].labels_.tolist()==b['models'][2].labels_.tolist()


def test_missing_and_degenerate_data():
    with pytest.raises(ValueError,match='missing'):
        response_table(pd.DataFrame({'id':[1],'a':[np.nan]}),['a'],['id'])
    with pytest.raises(ValueError,match='varying'):
        explore_responses(pd.DataFrame({'a':[1,1,1],'b':[2,2,2]}))
    with pytest.raises(ValueError,match='Duplicate'):
        response_table(pd.DataFrame({'id':[1,1],'run':[1,1],'a':[2,3]}),['a'],['id'],'run')

from surrogate_lab.response_exploration import (
    relationship_diagnostics, partial_relationships, within_configuration_relationships,
    compare_response_structures, pareto_profiles,
)


def test_nonlinear_dependence_and_reproducibility():
    x = np.linspace(-2, 2, 70)
    frame = pd.DataFrame({'x': x, 'square': x*x, 'constant': 1})
    result = relationship_diagnostics(frame, permutations=39, bootstrap=30, seed=10)
    assert len(result) == 1
    row = result.iloc[0]
    assert abs(row.pearson) < 1e-12
    assert row.distance_correlation > .45
    assert row.permutation_p <= .05
    assert row.bh_q >= row.permutation_p
    pd.testing.assert_frame_equal(result, relationship_diagnostics(frame, 39, 30, 10))


def test_partial_collinear_and_scale_invariance():
    x = np.arange(20.)
    frame = pd.DataFrame({'a': x, 'b': x*2, 'c': np.sin(x)})
    result = partial_relationships(frame)
    assert np.isfinite(result['partial']).all().all()
    np.testing.assert_allclose(np.diag(result['partial']), 1)
    np.testing.assert_allclose(result['partial'], partial_relationships(frame.assign(b=frame.b*100))['partial'])


def test_noise_relationship_differs_from_mean_relationship():
    raw = pd.DataFrame({'id': np.repeat(np.arange(10), 2), 'run': [0, 1]*10})
    raw['a'] = np.repeat(np.arange(10)*10, 2) + np.tile([-1., 1.], 10)
    raw['b'] = np.repeat(np.arange(10)*10, 2) + np.tile([1., -1.], 10)
    within = within_configuration_relationships(raw, ['a', 'b'], ['id'], 'run')
    assert within['replicated_configurations'] == 10
    assert within['correlation'].loc['a','b'] == pytest.approx(-1)
    assert response_table(raw, ['a','b'], ['id'])['means'].corr().loc['a','b'] == pytest.approx(1)
    single = within_configuration_relationships(raw.iloc[::2], ['a','b'], ['id'])
    assert single['replicated_configurations'] == 0
    assert single['correlation'].isna().all().all()


def test_cluster_comparison_and_noise_degeneracy():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(np.vstack([rng.normal(-4,.1,(15,3)), rng.normal(4,.1,(15,3))]), columns=list('abc'))
    result = compare_response_structures(frame, k=2, repeats=3, dbscan_eps=1e-9)
    assert result['labels'].index.equals(frame.index)
    assert result['embedding'].shape == (30, 2)
    assert set(result['labels']) == {'kmeans','ward','gmm','dbscan'}
    db = result['comparison'].set_index('method').loc['dbscan']
    assert db.noise_fraction == 1
    assert np.isnan(db.silhouette_excluding_noise)
    assert result['stability'].query("method == 'dbscan'").adjusted_rand.isna().all()
    assert result['stability'].query("method == 'kmeans'").adjusted_rand.min() > .9


def test_pareto_directions_and_duplicates():
    frame = pd.DataFrame({'benefit':[3, 3, 2, 4], 'cost':[1, 1, 2, 3]}, index=list('abcd'))
    result = pareto_profiles(frame, {'benefit':'max', 'cost':'min'})
    assert result.nondominated.tolist() == [True, True, False, True]
    assert result.loc['c','dominated_by'] == 2
    with pytest.raises(ValueError, match='Directions'):
        pareto_profiles(frame, {'benefit':'higher'})
