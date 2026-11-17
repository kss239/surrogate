"""Presentation functions returning matplotlib figures; no file writes."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .models import predict_bundle
from .data import summarize_runs


def plot_diagnostics(board, oof, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    best = board.groupby("family").rmse.min().sort_values()
    axes[0,0].barh(best.index[::-1], best.values[::-1], color="#356e96")
    axes[0,0].set(xlabel="Development CV RMSE", title="Best trial per family (selection scores)")
    axes[0,1].scatter(oof["mean"], oof.predicted, s=19, alpha=.7, color="#356e96")
    bounds = [min(oof["mean"].min(), oof.predicted.min()), max(oof["mean"].max(), oof.predicted.max())]
    axes[0,1].plot(bounds, bounds, "--", color=".3")
    axes[0,1].set(xlabel="Observed repetition mean", ylabel="OOF prediction", title="Selected procedure: development predictions")
    axes[1,0].scatter(oof.predicted, oof.residual, s=19, color="#b27724")
    axes[1,0].axhline(0, color=".3", linestyle="--")
    axes[1,0].set(xlabel="OOF prediction", ylabel="Observed minus predicted", title="Residual structure")
    axes[1,1].scatter(oof.distance, abs(oof.residual), s=19, color="#356e96")
    axes[1,1].set(xlabel="Nearest training distance (normalized factors)", ylabel="Absolute error", title="Error versus support")
    return fig

def plot_learning_curves(curves):
    fig, axes = plt.subplots(1,2,figsize=(11,3.5), constrained_layout=True)
    for ax,(kind,group) in zip(axes,curves.groupby("axis",sort=False)):
        ax.plot(group.amount,group.rmse,"o-",color="#356e96")
        ax.set(xlabel=kind.replace("_"," "),ylabel="Held-out RMSE",title="Fixed development holdout")
    return fig

def plot_interactions(interactions, cfg):
    matrix = np.zeros((len(cfg.factors),len(cfg.factors)))
    for row in interactions.itertuples():
        j,k=cfg.factors.index(row.factor_a),cfg.factors.index(row.factor_b)
        matrix[j,k]=matrix[k,j]=row.screen_score
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
    im=ax.imshow(matrix,cmap="Blues")
    ax.set_xticks(range(len(cfg.factors)),cfg.factors,rotation=60,ha="right")
    ax.set_yticks(range(len(cfg.factors)),cfg.factors)
    ax.set_title("Training-only interaction screening (not SHAP)")
    fig.colorbar(im,ax=ax,label="Residual-product screen score")
    return fig

def plot_surface(bundle, reference, pair, cfg, feasibility, points=35):
    """Conditional slice, not marginal effect: other factors fixed at development medians."""
    j,k=pair
    X=reference[list(cfg.factors)].to_numpy()
    a=np.linspace(*cfg.bounds[j],points); b=np.linspace(*cfg.bounds[k],points)
    xx,yy=np.meshgrid(a,b)
    query=np.tile(np.median(X,axis=0),(points*points,1))
    query[:,j],query[:,k]=xx.ravel(),yy.ravel()
    mask=np.asarray(feasibility(query),bool)
    z=np.full(len(query),np.nan)
    z[mask]=predict_bundle(bundle,query[mask],cfg)
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    im=ax.contourf(xx,yy,z.reshape(xx.shape),levels=16,cmap="viridis")
    ax.scatter(X[:,j],X[:,k],s=7,facecolors="none",edgecolors="white",alpha=.65)
    ax.set(xlabel=cfg.factors[j],ylabel=cfg.factors[k],title="Conditional prediction slice: other factors at medians")
    fig.colorbar(im,ax=ax,label=cfg.response)
    return fig

def plot_final(final, calibrator, cfg):
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    ordered=final["means"].sort_values("predicted_mean").reset_index(drop=True)
    if np.isfinite(calibrator["q"]):
        axes[0].errorbar(np.arange(len(ordered)),ordered.predicted_mean,
            yerr=calibrator["q"]*ordered.run_scale,fmt=".",alpha=.7,color="#356e96",label="Future-run interval")
    else:
        axes[0].scatter(np.arange(len(ordered)),ordered.predicted_mean,color="#356e96",label="Mean; interval unbounded")
    axes[0].scatter(np.arange(len(ordered)),ordered.test_run,s=13,color="#b27724",label="One held-out run")
    axes[0].set(xlabel="Test alternatives ordered by prediction",ylabel=cfg.response,title=f"Nominal {1-cfg.alpha:.0%} run intervals")
    axes[0].legend(fontsize=8)
    curve=final["coverage_curve"]
    axes[1].plot(curve.nominal,curve.observed,"o-",color="#356e96")
    axes[1].plot([.45,1],[.45,1],"--",color=".3")
    axes[1].set(xlabel="Nominal coverage",ylabel="Observed test coverage",title="Run-interval calibration",xlim=(.45,1),ylim=(0,1.03))
    return fig


def plot_postfit(report, cfg):
    """Observed on x, predicted on y; common square axes expose horizontal compression."""
    fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    values=[]
    for part in ('training','heldout'):
        values.extend(report['runs'][part][cfg.response].tolist())
        values.extend(report['means'][part].predicted.tolist())
    lo,hi=min(values),max(values); pad=max((hi-lo)*.05,1e-6); limits=(lo-pad,hi+pad)
    for row,part in enumerate(('training','heldout')):
        for col,grain in enumerate(('individual_run','alternative_mean')):
            ax=axes[row,col]
            frame=report['runs'][part] if col==0 else report['means'][part]
            observed=frame[cfg.response] if col==0 else frame['mean']
            ax.scatter(observed,frame.predicted,s=10 if col==0 else 22,alpha=.35 if col==0 else .75,color='#356e96')
            ax.plot(limits,limits,'--',color='.3',label='Perfect prediction')
            ax.axhline(report['baseline'],linestyle=':',color='#b27724',label='Training mean baseline')
            ax.set(xlabel='Observed '+('individual run' if col==0 else 'alternative mean'),
                   ylabel='Predicted mean',title=f'{part.title()} • {grain.replace("_"," ")}',xlim=limits,ylim=limits)
            ax.set_aspect('equal',adjustable='box')
    axes[0,0].legend(fontsize=8)
    return fig
