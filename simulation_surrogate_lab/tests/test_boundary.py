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


def test_ties_dedup_budget_and_no_interior_selection():
 import pandas as pd
 from surrogate_lab.boundary import boundary_support
 c=config_from_bounds({'x':(0,10),'y':(0,10)},split_strategy='boundary',
                      boundary_budget=8,boundary_max_fraction=.4)
 data=pd.DataFrame({'group':range(20),'x':[1,1,9,9]+[5]*16,
                    'y':[2,8,2,8]+[5]*16,'response_1':range(20)})
 a=boundary_support(data,c)
 b=boundary_support(data.sample(frac=1,random_state=42),c)
 assert a['selected'].group.tolist()==b['selected'].group.tolist()==[0,2,1]
 assert not a['budget_conflict'] and not a['coverage'].gap.any()
 assert set(a['selected'].group)<={0,1,2,3}
 assert all(a['coverage'].physical_bound_gap>0)
 limited=boundary_support(data,replace(c,boundary_budget=1))
 assert limited['budget_conflict']
 assert set(limited['uncovered_extremes'].target)=={'x:max','y:max'}
 assert limited['selected'].iloc[0].extremes_covered==('x:min','y:min')
 zero=boundary_support(data,replace(c,boundary_budget=0))
 assert zero['selected'].empty and zero['coverage'].gap.all()
