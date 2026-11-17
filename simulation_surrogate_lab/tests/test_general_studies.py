from dataclasses import replace
import numpy as np
import pytest
from surrogate_lab import config_from_bounds, Recipe
from surrogate_lab.data import synthetic_runs, audit_runs, summarize_runs, training_mean_noise
from surrogate_lab.workflow import prepare_study, develop, box_feasible
from surrogate_lab.models import fit_recipe, predict_fit, predict_gp_mean_sd
from surrogate_lab.validation import distributed_splits, challenge_splits
from surrogate_lab.diagnostics import allocation_learning_curves, interaction_screen, response_noise_report
from surrogate_lab.postdiagnostics import postfit_report
from surrogate_lab.uncertainty import fit_uncertainty, calibrate_runs, final_evaluation
from surrogate_lab.design import propose_batch

@pytest.mark.parametrize('dimensions',[1,2,5,17])
def test_arbitrary_dimensions_unequal_reps(dimensions,tmp_path):
    cfg=config_from_bounds({f'x{i}':(-2,3) for i in range(dimensions)},
        checkpoint_dir=str(tmp_path),bootstrap_members=2)
    counts=np.resize([1,2,3,7,4],31)
    raw=synthetic_runs(cfg,31,counts)
    study=prepare_study(raw,cfg)
    inv=study['inventory'].iloc[0]
    assert inv.unique_configurations==31 and inv.total_runs==counts.sum()
    assert inv.factors==dimensions and not inv.balanced_repetitions
    s=summarize_runs(study['runs'],cfg)
    assert s.loc[s.reps==1,'variance'].isna().all()
    assert s.loc[s.reps==1,'mean_variance'].isna().all()
    recipe=Recipe('ridge','selected')
    fitted=fit_recipe(study['development'],recipe,cfg)
    assert np.isfinite(predict_fit(fitted,s[list(cfg.factors)],cfg)).all()
    report=postfit_report((fitted,),study['development'],study['test'],cfg)
    assert len(report['coverage']['marginal'])==dimensions
    assert len(report['coverage']['pairwise'])==dimensions*(dimensions-1)//2
    if dimensions==1: assert interaction_screen(study['development'],cfg).empty
    result=develop(study['development'],cfg,recipes=(recipe,))
    assert len(result['finalists']['oof'])==study['development'].group.nunique()

@pytest.mark.parametrize('counts',[1,np.resize([1,2,5],24)])
def test_uncertainty_and_proposals_with_singletons(counts,tmp_path):
    cfg=config_from_bounds({'temperature':(0,1),'pressure':(0,1)},
        checkpoint_dir=str(tmp_path),bootstrap_members=2,candidate_count=32,
        new_alternatives=2,repeat_alternatives=2)
    study=prepare_study(synthetic_runs(cfg,24,counts),cfg)
    dev=study['development'];recipe=Recipe('ridge','raw')
    result=develop(dev,cfg,recipes=(recipe,))
    state=fit_uncertainty(dev,(recipe,),cfg)
    calibration=calibrate_runs(state,study['calibration'],cfg)
    final=final_evaluation(state,calibration,study['test'],cfg,box_feasible)
    assert np.isfinite(final['mean_metrics']['rmse'])
    if np.isscalar(counts):
        assert state['noise_status']=='unavailable_no_replication'
        assert final['means'].estimated_run_sd.isna().all()
        assert (final['means'].run_scale==1).all()
    else:
        assert state['noise_status']=='replicated_subset_model'
        assert np.isfinite(final['means'].estimated_run_sd).all()
    reserved=study['summary'].loc[~study['summary'].group.isin(dev.group),list(cfg.factors)].to_numpy()
    batch=propose_batch(state,dev,result['finalists']['oof'],cfg,box_feasible,reserved)
    repeats=batch.loc[batch.action=='additional_repetitions']
    assert len(batch)==4
    assert repeats.expected_local_mean_variance_reduction.isna().all()
    assert repeats.reason.str.contains('unknown').all()


def test_singleton_gp_noise_is_explicit():
    cfg=config_from_bounds({'x':(0,1)})
    runs=audit_runs(synthetic_runs(cfg,12,1),cfg,box_feasible)
    recipe=Recipe('gp','raw')
    model=fit_recipe(runs,recipe,cfg)
    assert model['noise_method']=='learned_shared_noise'
    assert training_mean_noise(summarize_runs(runs,cfg)) is None
    X=runs[list(cfg.factors)].to_numpy()
    assert np.isfinite(predict_fit(model,X,cfg)).all()
    assert np.isfinite(predict_gp_mean_sd(model,X,cfg)).all()
    assert model['model'].kernel_.k2.noise_level>0
    mixed=audit_runs(synthetic_runs(cfg,12,np.resize([1,3],12)),cfg,box_feasible)
    model=fit_recipe(mixed,recipe,cfg)
    assert model['noise_method']=='training_pooled_for_singletons'
    assert np.isfinite(model['model'].alpha).all()


def test_noise_unknown_is_not_zero():
    cfg=config_from_bounds({'x':(0,1)})
    for counts in [1,np.resize([1,4],12)]:
        raw=synthetic_runs(cfg,12,counts)
        row=response_noise_report(raw,[cfg.response],cfg,box_feasible).iloc[0]
        assert np.isnan(row.signal_variance) and np.isnan(row.run_noise_variance)
        assert row.singleton_alternatives>0


def test_curves_keep_unbalanced_observations():
    cfg=config_from_bounds({'x':(0,1)})
    data=audit_runs(synthetic_runs(cfg,15,np.resize([1,3,7],15)),cfg,box_feasible)
    split=distributed_splits(data,cfg)[0]
    curves=allocation_learning_curves(data,(Recipe('ridge','raw'),),(split,),cfg,
        fractions=(1.,),repetitions=(1,3,7),seeds=(0,))
    available=data.loc[data.group.isin(split[1])].group.value_counts()
    for row in curves.itertuples():
        assert row.simulation_runs==available.clip(upper=row.repetitions).sum()
        assert row.min_repetitions==1
    assert curves.simulation_runs.max()==available.sum()


def test_small_studies_and_operation_specific_limits(tmp_path):
    cfg=config_from_bounds({'x':(0,1)},checkpoint_dir=str(tmp_path))
    audited=audit_runs(synthetic_runs(cfg,2,1),cfg,box_feasible)
    assert len(audited)==2
    with pytest.raises(ValueError,match='Too few alternatives'):
        prepare_study(synthetic_runs(cfg,2,1),cfg)
    study=prepare_study(synthetic_runs(cfg,5,1),cfg)
    assert study['development'].group.nunique()==3
    for _,tr,va in challenge_splits(study['development'],cfg):
        assert len(tr)>=2 and len(va)>=1 and not set(tr)&set(va)
    result=develop(study['development'],cfg,recipes=(Recipe('ridge','raw'),))
    assert result['board'].shape[0]==1
    report=postfit_report((fit_recipe(study['development'],Recipe('ridge','raw'),cfg),),
                         study['development'],study['test'],cfg)
    assert len(report['scores'])==4


def test_schema_requires_explicit_aligned_inputs():
    with pytest.raises(ValueError,match='at least one'):
        config_from_bounds({})
    with pytest.raises(ValueError,match='distinct'):
        config_from_bounds({'response_1':(0,1)})
