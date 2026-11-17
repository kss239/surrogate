"""Response-independent, budgeted operating-region support selection."""
import numpy as np
import pandas as pd
from .data import scale_bounds


def boundary_support(data, cfg):
    s=data.groupby('group')[list(cfg.factors)].first().sort_index()
    X=scale_bounds(s.to_numpy(),cfg)
    n,d=X.shape
    budget=min(cfg.boundary_budget, int(n*cfg.boundary_max_fraction), max(0,n-5))
    # Marginal targets and joint directions; only observed feasible points qualify.
    targets=[]
    for j in range(d):
        for side in (0.,1.):
            distance=abs(X[:,j]-side)
            targets.append((f'{cfg.factors[j]}:{int(side)}',distance))
    rng=np.random.default_rng(cfg.seed)
    for k in range(min(32,4*d)):
        corner=rng.integers(0,2,d)
        targets.append((f'joint_direction_{k}',np.linalg.norm(X-corner,axis=1)/np.sqrt(d)))
    marginal, joint = targets[:2*d], targets[2*d:]
    targets=[item for pair in zip(marginal,joint) for item in pair]+marginal[len(joint):]+joint[len(marginal):]
    chosen=[]; reasons={}
    for label,dist in targets:
        if len(chosen)>=budget: break
        index=int(np.argmin(dist))
        if index not in chosen:
            chosen.append(index);reasons[index]=label
    # Fill remaining budget with points farthest from current support.
    while len(chosen)<budget:
        dist=np.min(np.linalg.norm(X[:,None]-X[chosen],axis=2),axis=1) if chosen else np.linalg.norm(X-.5,axis=1)
        dist[chosen]=-1
        index=int(np.argmax(dist));chosen.append(index);reasons[index]='joint_space_filling'
    gaps=[]
    for label,dist in targets:
        achieved=float(min(dist[chosen])) if chosen else float('inf')
        gaps.append(dict(target=label,best_available=float(min(dist)),selected_distance=achieved,
                         gap=achieved>cfg.boundary_tolerance))
    selected=s.iloc[chosen].reset_index().assign(reason=[reasons[i] for i in chosen])
    return {'selected':selected,'coverage':pd.DataFrame(gaps),'requested_budget':cfg.boundary_budget,
            'effective_budget':budget,'target':'support-conditioned interpolation; joint support not guaranteed'}
