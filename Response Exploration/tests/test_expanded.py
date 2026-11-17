from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from surrogate_lab import Config, Recipe
from surrogate_lab.data import synthetic_runs, summarize_runs, subset
from surrogate_lab.workflow import prepare_study, box_feasible
from surrogate_lab.models import fit_recipe, predict_fit
from surrogate_lab.search import optional_model_status, make_recipes
from surrogate_lab.validation import distributed_splits, evaluate_recipe
from surrogate_lab.diagnostics import response_noise_report, allocation_learning_curves
from surrogate_lab.postdiagnostics import postfit_report

@pytest.fixture
def example():
    cfg=Config()
    raw=synthetic_runs(cfg,60,4)
    return cfg,raw,prepare_study(raw,cfg)

@pytest.mark.parametrize('family,flag',[('gam','enable_gam'),('ebm','enable_ebm'),('xgb','enable_xgboost'),('lgbm','enable_lightgbm')])
def test_optional_models(example,family,flag):
    cfg,_,study=example
    cfg=replace(cfg,**{flag:True})
    status=optional_model_status(cfg).set_index('family').loc[family]
    if not status.available: pytest.skip(status.reason)
    params={'gam':(('n_splines',5),), 'ebm':(('max_rounds',20),('outer_bags',2)),
            'xgb':(('n_estimators',10),), 'lgbm':(('n_estimators',10),)}[family]
    fitted=fit_recipe(study['development'],Recipe(family,'raw',params=params),cfg)
    assert np.isfinite(predict_fit(fitted,study['test'][list(cfg.factors)].to_numpy(),cfg)).all()


def test_unavailable_requested_model_visible(example,monkeypatch):
    import surrogate_lab.search as search
    cfg,_,_=example
    def missing(name): raise OSError('native library unavailable')
    monkeypatch.setattr(search,'import_module',missing)
    cfg=replace(cfg,enable_lightgbm=True)
    row=optional_model_status(cfg).set_index('family').loc['lgbm']
    assert not row.available and 'native library unavailable' in row.reason
    assert 'lgbm' not in {r.family for r in make_recipes(cfg,60)}

@pytest.mark.parametrize('kernel',['rbf','matern'])
def test_ard_and_noise_scaling(example,kernel):
    cfg,_,study=example
    data=study['development']
    fitted=fit_recipe(data,Recipe('gp','raw',params=(('kernel',kernel),('ard',True))),cfg)
    np.testing.assert_allclose(fitted['model'].alpha,
        np.maximum(summarize_runs(data,cfg).mean_variance/fitted['scale']**2,1e-8))
    assert len(fitted['model'].kernel_.k2.length_scale)==len(cfg.factors)


def test_response_specific_noise(example):
    cfg,raw,_=example
    raw['scaled']=raw[cfg.response]*3
    result=response_noise_report(raw,[cfg.response,'scaled'],cfg,box_feasible)
    assert result.iloc[1].run_noise_variance==pytest.approx(9*result.iloc[0].run_noise_variance)
    assert result.iloc[1].estimated_run_signal_fraction==pytest.approx(result.iloc[0].estimated_run_signal_fraction)


def test_crossed_curves_cost_and_disjointness(example):
    cfg,_,study=example
    data=study['runs'];split=distributed_splits(data,cfg)[0]
    curves=allocation_learning_curves(data,(Recipe('ridge','raw'),),(split,),cfg,
                                      fractions=(.5,1),repetitions=(2,4),seeds=(0,1))
    assert len(curves)==16
    assert (curves.simulation_runs==curves.alternatives*curves.repetitions).all()
    assert set(curves.grain)=={'alternative_mean','individual_run'}
    with pytest.raises(ValueError,match='Overlapping'):
        allocation_learning_curves(data,(Recipe('ridge','raw'),),(('bad',split[1],split[1]),),cfg)


def test_post_phase_and_cv_grains(example):
    cfg,_,study=example
    train,held=study['development'],study['test']
    recipe=Recipe('ridge','quadratic')
    report=postfit_report((fit_recipe(train,recipe,cfg),),train,held,cfg)
    assert len(report['coverage']['pairwise'])==66
    assert report['coverage']['summary'].iloc[0].quadratic_coefficients==91
    assert set(report['residual_patterns'].variable)>={'predicted','distance',cfg.factors[0]}
    lo=np.quantile(summarize_runs(train,cfg)['mean'],.1)
    assert (report['tails'].lower_threshold==lo).all()
    trial=evaluate_recipe(recipe,train,distributed_splits(train,cfg),cfg)
    assert np.isfinite(trial['run_metrics']['rmse'])


def test_validation_changes_do_not_change_fold_predictions(example):
    cfg,_,study=example
    data=study['runs'];split=distributed_splits(data,cfg)[0]
    changed=data.copy();changed.loc[changed.group.isin(split[2]),cfg.response]+=10000
    for recipe in (Recipe('elastic','quadratic'),Recipe('kernel_ridge','raw'),Recipe('gp','raw')):
        first=evaluate_recipe(recipe,data,(split,),cfg)
        second=evaluate_recipe(recipe,changed,(split,),cfg)
        np.testing.assert_allclose([p['predicted'] for p in first['predictions']],
                                   [p['predicted'] for p in second['predictions']])
