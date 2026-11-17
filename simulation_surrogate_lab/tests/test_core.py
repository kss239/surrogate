from dataclasses import replace, FrozenInstanceError
import numpy as np
import pandas as pd
import pytest
from surrogate_lab import Config, Recipe
from surrogate_lab.data import synthetic_runs,audit_runs,summarize_runs,subset,scale_bounds
from surrogate_lab.workflow import prepare_study,box_feasible,develop,nested_evaluate
from surrogate_lab.models import fit_recipe,predict_fit
from surrogate_lab.features import domain_matrix
from surrogate_lab.validation import distributed_splits,metrics
from surrogate_lab.search import run_search,make_recipes
from surrogate_lab.storage import package_fingerprint,data_hash,export_study
from surrogate_lab.uncertainty import conformal_quantile,fit_uncertainty,calibrate_runs,predict_new,final_evaluation
from surrogate_lab.design import propose_batch

@pytest.fixture
def cfg(tmp_path):
    return Config(checkpoint_dir=str(tmp_path/'cache'),bootstrap_members=3,candidate_count=64,new_alternatives=3,repeat_alternatives=2)

@pytest.fixture
def study(cfg):
    return prepare_study(synthetic_runs(cfg,72,4),cfg)


def test_configuration_immutable(cfg):
    with pytest.raises(FrozenInstanceError): cfg.seed=7

@pytest.mark.parametrize('kwargs',[{'workers':0},{'folds':1},{'bootstrap_members':1},{'profile':'unknown'},{'new_alternatives':0}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError): Config(**kwargs)


def test_partition_whole_configurations(study,cfg):
    parts=study['partitions']
    assert not set(parts['development']) & set(parts['test'])
    assert not set(parts['development']) & set(parts['calibration'])
    assert sum(map(len,parts.values()))==72
    for _,tr,va in distributed_splits(study['development'],cfg): assert not set(tr)&set(va)


def test_duplicate_vectors_share_group(cfg):
    raw=synthetic_runs(cfg,50,3)
    extra=raw.iloc[:3].assign(alternative='another_id')
    audited=audit_runs(pd.concat([raw,extra]),cfg,box_feasible)
    assert audited.group.nunique()==50
    assert audited.loc[audited.alternative=='another_id','group'].iloc[0]==audited.iloc[0].group

@pytest.mark.parametrize('kind',['missing','infinite','bounds','key','varying'])
def test_audit_rejects_bad_data(cfg,kind):
    raw=synthetic_runs(cfg,50,3)
    if kind=='missing': raw.loc[0,cfg.response]=np.nan
    if kind=='infinite': raw.loc[0,cfg.response]=np.inf
    if kind=='bounds': raw.loc[0,cfg.factors[0]]=2
    if kind=='key': raw=pd.concat([raw,raw.iloc[[0]]])
    if kind=='varying': raw.loc[0,cfg.factors[0]]=.123
    with pytest.raises(ValueError): audit_runs(raw,cfg,box_feasible)


def test_zeros_retained(cfg):
    raw=synthetic_runs(cfg,50,3); raw.loc[0,cfg.response]=0
    assert len(audit_runs(raw,cfg,box_feasible))==len(raw)


def test_input_immutable_and_leakage(study,cfg):
    data=study['development']; before=data_hash(data)
    _,tr,va=distributed_splits(data,cfg)[0]
    changed=data.assign(**{cfg.response:np.where(data.group.isin(va),1e12,data[cfg.response])})
    recipe=Recipe('ridge','selected',params=(('alpha',1.),))
    a=fit_recipe(subset(data,tr),recipe,cfg); b=fit_recipe(subset(changed,tr),recipe,cfg)
    X=summarize_runs(data,cfg)[list(cfg.factors)].to_numpy()
    assert a['features']['pairs']==b['features']['pairs']
    np.testing.assert_allclose(predict_fit(a,X,cfg),predict_fit(b,X,cfg))
    assert data_hash(data)==before


def test_weighted_run_ridge_equals_means(study,cfg):
    data=study['development']
    # Unequal repetitions must not change equal-alternative squared-error fitting.
    data=pd.concat([data,data.iloc[:7]])
    mean=Recipe('ridge','quadratic',params=(('alpha',2.),))
    run=replace(mean,training='run')
    X=summarize_runs(data,cfg)[list(cfg.factors)].to_numpy()
    np.testing.assert_allclose(predict_fit(fit_recipe(data,mean,cfg),X,cfg),
                               predict_fit(fit_recipe(data,run,cfg),X,cfg),rtol=1e-7,atol=1e-7)

@pytest.mark.parametrize('family',['ridge','elastic','kernel_ridge','forest','extra','boost','svr','mlp','gp'])
def test_family_predicts_finite(family,study,cfg):
    fitted=fit_recipe(study['development'],Recipe(family,'raw'),cfg)
    X=summarize_runs(study['test'],cfg)[list(cfg.factors)].to_numpy()
    assert np.isfinite(predict_fit(fitted,X,cfg)).all()


def test_conformal_finite_rank():
    assert conformal_quantile(np.arange(1,10),.2)==8
    assert np.isinf(conformal_quantile([1,2],.1))


def test_ratio_rejects_zero(cfg):
    c=replace(cfg,domain_features=(('ratio',0,1),))
    with pytest.raises(ValueError): domain_matrix(np.zeros((2,12)),np.zeros((2,12)),c)


def test_checkpoints_resume_and_source_invalidation(study,cfg):
    recipes=(Recipe('ridge','raw'),)
    splits=distributed_splits(study['development'],cfg)
    a=run_search(recipes,study['development'],splits,cfg,'source1')
    b=run_search(recipes,study['development'],splits,cfg,'source1')
    assert a['records']==b['records']
    c=run_search(recipes,study['development'],splits,cfg,'source2')
    assert a['directory']!=c['directory']


def test_failure_recorded(study,cfg):
    search=run_search((Recipe('nonexistent','raw'),),study['development'],distributed_splits(study['development'],cfg),cfg,'a')
    assert search['records'][0]['status']=='failed'


def test_breadth(study,cfg):
    recipes=make_recipes(cfg,72)
    assert set(r.family for r in recipes)=={'ridge','elastic','kernel_ridge','forest','extra','boost','svr','mlp','gp'}
    assert {'mean','run'}<=set(r.training for r in recipes)


def test_end_to_end_and_export(study,cfg,tmp_path):
    dev=study['development']
    result=develop(dev,cfg,recipes=(Recipe('ridge','raw'),Recipe('ridge','quadratic')))
    selected=result['finalists']['recipes']
    u=fit_uncertainty(dev,selected,cfg)
    cal=calibrate_runs(u,study['calibration'],cfg)
    final=final_evaluation(u,cal,study['test'],cfg,box_feasible)
    assert np.isfinite(final['mean_metrics']['rmse'])
    reserved=study['summary'].loc[~study['summary'].group.isin(study['partitions']['development']),list(cfg.factors)].to_numpy()
    batch=propose_batch(u,dev,result['finalists']['oof'],cfg,box_feasible,reserved)
    assert len(batch)==5
    assert set(batch.loc[batch.action=='additional_repetitions','group'])<=set(dev.group)
    b=np.asarray(cfg.bounds); X=batch[list(cfg.factors)].to_numpy()
    assert ((X>=b[:,0])&(X<=b[:,1])).all()
    with pytest.raises(ValueError): predict_new(u,cal,np.full((1,12),2.),cfg,box_feasible)
    out=export_study(tmp_path/'export',cfg,study['runs'],study['partitions'],result['records'],result['board'],result['finalists'],u,cal,batch,final)
    assert (tmp_path/'export'/'manifest.json').exists()


def test_nested_selection_runs(study,cfg):
    result=nested_evaluate(study['development'],cfg,outer_folds=2,recipes=(Recipe('ridge','raw'),))
    assert len(result['predictions'])==study['development'].group.nunique()
    assert np.isfinite(result['metrics']['rmse'])


def test_manifest_cannot_reassign_new_data(study,cfg):
    manifest={"data_hash":data_hash(study["runs"]),"partitions":study["partitions"]}
    raw=synthetic_runs(cfg,72,4)
    same=prepare_study(raw,cfg,manifest=manifest)
    assert same["partitions"]["test"].tolist()==study["partitions"]["test"].tolist()
    changed=raw.copy(); changed.loc[0,cfg.response]+=1
    with pytest.raises(ValueError): prepare_study(changed,cfg,manifest=manifest)
