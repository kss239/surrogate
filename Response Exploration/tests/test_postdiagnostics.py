from dataclasses import replace
import numpy as np
import pytest
from surrogate_lab import Config,Recipe
from surrogate_lab.data import synthetic_runs,summarize_runs
from surrogate_lab.workflow import prepare_study
from surrogate_lab.models import fit_bundle
from surrogate_lab.postdiagnostics import prediction_statistics,postfit_report


def test_horizontal_band_detected_metrics():
    y=np.linspace(0,100,101); p=np.full_like(y,50)
    s=prediction_statistics(y,p,50,1)
    assert s['sd_ratio']==0 and s['skill_vs_training_mean']==0
    assert s['predicted_on_observed_slope']==0
    assert np.isnan(s['correlation'])


def test_perfect_predictions_and_constant_observations():
    y=np.arange(10.)
    s=prediction_statistics(y,y,4.5,1)
    assert s['sd_ratio']==1 and s['skill_vs_training_mean']==1
    s=prediction_statistics(np.ones(5),np.ones(5),1,1)
    assert np.isnan(s['sd_ratio']) and np.isnan(s['r2']) and np.isnan(s['skill_vs_training_mean'])


def fixture():
    cfg=Config()
    study=prepare_study(synthetic_runs(cfg,70,5),cfg)
    bundle=fit_bundle(study['development'],(Recipe('ridge','raw'),),cfg)
    return cfg,study,bundle


def test_keyed_alignment_and_flat_findings():
    cfg,study,bundle=fixture()
    pred=summarize_runs(study['test'],cfg).assign(predicted_mean=summarize_runs(study['development'],cfg)['mean'].mean())
    a=postfit_report(bundle,study['development'],study['test'],cfg,observed_predictions=pred)
    b=postfit_report(bundle,study['development'],study['test'],cfg,observed_predictions=pred.sample(frac=1,random_state=1))
    assert 'Near-flat held-out predictions' in a['findings'].finding.tolist()
    np.testing.assert_allclose(a['scores'].rmse,b['scores'].rmse)
    bad=pred.copy(); bad.loc[bad.index[0],cfg.factors[0]]+=.1
    with pytest.raises(ValueError):postfit_report(bundle,study['development'],study['test'],cfg,observed_predictions=bad)
    with pytest.raises(ValueError):postfit_report(bundle,study['development'],study['development'],cfg)
