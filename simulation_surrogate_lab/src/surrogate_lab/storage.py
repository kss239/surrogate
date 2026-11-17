"""Explicit checkpoint and export boundary. Only load trusted local model files."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import hashlib, json, platform
import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from . import __version__ as ENGINE_VERSION


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def data_hash(data):
    return hashlib.sha256(pd.util.hash_pandas_object(data.drop(columns=["boundary_support"], errors="ignore"), index=True).values.tobytes()).hexdigest()


def export_study(directory, cfg, runs, partitions, records, board, finalists, uncertainty, calibrator, proposal, final):
    path=Path(directory); path.mkdir(parents=True, exist_ok=True)
    board.to_csv(path/"leaderboard.csv",index=False)
    finalists["oof"].to_csv(path/"development_oof_predictions.csv",index=False)
    proposal.to_csv(path/"next_simulation_batch.csv",index=False)
    final["means"].to_csv(path/"final_test_predictions.csv",index=False)
    manifest={"engine":ENGINE_VERSION,"data_hash":data_hash(runs),"config":asdict(cfg),
              "partitions":{k:v.tolist() for k,v in partitions.items()},
              "configuration_groups": runs.groupby("group")[list(cfg.factors)].first().reset_index().to_dict("records"),
              "source_hash": package_fingerprint(),
              "selected_recipes":[asdict(r) for r in finalists["recipes"]],
              "environment":{"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__},
              "final_metrics":final["mean_metrics"],"interval_target":calibrator["target"]}
    (path/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str))
    (path/"experiment_ledger.json").write_text(json.dumps(records,default=str))
    dump({"uncertainty":uncertainty,"calibration":calibrator,"config":cfg},path/"trusted_model_bundle.joblib")
    return str(path.resolve())

def package_fingerprint():
    """Hash all package Python sources, stable across interpreter sessions."""
    files = sorted(Path(__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in files)).hexdigest()[:20]
