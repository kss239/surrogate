"""Deterministic observed-extreme support; no responses or geometric expansion."""
import pandas as pd


def boundary_support(data, cfg):
    """Visit factors in configured order, minimum then maximum.

    Reuse an already nominated tied group; otherwise use the smallest group ID.
    Auditing assigns IDs from sorted factor vectors, independent of row order.
    Budget truncates this deduplicated nomination order, never adds interior points.
    """
    s=data.groupby('group')[list(cfg.factors)].first().sort_index()
    if s.empty: raise ValueError('Cannot select extremes from empty data')
    budget=min(cfg.boundary_budget, int(len(s)*cfg.boundary_max_fraction), max(0,len(s)-5))
    nominated=[]; rows=[]
    for factor, bounds in zip(cfg.factors,cfg.bounds):
        for side,value,physical in [('min',s[factor].min(),bounds[0]),('max',s[factor].max(),bounds[1])]:
            tied=s.index[s[factor]==value].tolist()
            group=next((g for g in nominated if g in tied),tied[0])
            if group not in nominated: nominated.append(group)
            rows.append(dict(target=f'{factor}:{side}',factor=factor,side=side,
                observed_value=float(value),physical_bound=float(physical),
                physical_bound_gap=abs(float(value)-physical), nominated_group=group,
                tied_groups=tuple(tied)))
    chosen=nominated[:budget]
    for row in rows:
        row['selected_groups']=tuple(g for g in chosen if g in row['tied_groups'])
        row['gap']=not bool(row['selected_groups'])
    coverage=pd.DataFrame(rows)
    selected=s.loc[chosen].reset_index()
    selected['extremes_covered']=[tuple(r['target'] for r in rows if g in r['selected_groups']) for g in chosen]
    selected['reason']=['; '.join(x) for x in selected.extremes_covered]
    return dict(selected=selected,coverage=coverage,
        uncovered_extremes=coverage.loc[coverage.gap].copy(),
        requested_budget=cfg.boundary_budget,effective_budget=budget,
        required_edge_alternatives=len(nominated),budget_conflict=len(nominated)>budget,
        budget_limits=dict(absolute=cfg.boundary_budget,fraction=int(len(s)*cfg.boundary_max_fraction),
                           reserve_for_splitting=max(0,len(s)-5)),
        target='observed-extreme-supported validation; joint support not guaranteed')
