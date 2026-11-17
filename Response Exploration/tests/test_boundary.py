from dataclasses import replace
import numpy as np
import pytest
from surrogate_lab import config_from_bounds,Recipe
from surrogate_lab.data import synthetic_runs
from surrogate_lab.workflow import prepare_study
from surrogate_lab.validation import distributed_splits,evaluate_recipe

@pytest.mark.parametrize('d',[1,3,17])
def test_boundary(d):
 c=config_from_bounds({f'x{i}':(0,1) for i in range(d)},split_strategy='boundary',boundary_budget=6)
 raw=synthetic_runs(c,50,np.resize([1,3,7],50))
 a=prepare_study(raw,c);b=prepare_study(raw.assign(response_1=999),c)
 ids=set(a['boundary']['selected'].group)
 assert ids==set(b['boundary']['selected'].group) and len(ids)<=6
 for _,tr,va in distributed_splits(a['development'],c):
  assert ids<=set(tr) and not set(tr)&set(va)
 assert not ids&set(a['partitions']['test'])
 assert sum(map(len,a['partitions'].values()))==50
 assert len(a['boundary']['coverage'])>=2*d


def test_small():
 c=config_from_bounds({'x':(0,1)},split_strategy='boundary',boundary_budget=100)
 a=prepare_study(synthetic_runs(c,5,1),c)
 assert a['boundary']['effective_budget']<=1
