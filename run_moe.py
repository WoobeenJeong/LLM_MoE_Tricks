#!/usr/bin/env python
"""Run a generic multi-omics MoE from a YAML config.

  python run_moe.py configs/scratch.yaml            # train everything at once
  python run_moe.py configs/pretrained_gate.yaml    # frozen experts, gate only
  python run_moe.py cfg.yaml --set train.epochs=50 --set model.gate.type=attention
  python run_moe.py --selftest                      # end-to-end on synthetic data

The two modes are the SAME code path; what separates them is the config:
`model.experts.<view>.pretrained` points at a .pth and `unfreeze_last_n` says
how deep that expert is allowed to keep learning (0 = frozen, n = last n
parameter-owning sub-modules, -1 = everything).

`optimize.space` turns any dotted config path into an optuna search dimension,
so the architecture (mlp / cnn1d / transformer), the gate structure (sigmoid /
softmax / gumbel / attention), the unfreeze depth and every loss weight can be
searched without touching the code.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from moe_core import build_moe, set_seed
from moe_data import aggregate_rows, load_views
from moe_train import (
    append_tsv,
    binary_metrics,
    cross_val,
    get_device,
    predict,
    save_bundle,
    score_oof,
    train_model,
)

DEFAULTS = {
    "run": {"name": "moe", "out_dir": "result", "seed": 42, "objective": "auc",
            "gpu": None, "threads": 8, "log_every": 25},
    "model": {"num_classes": 1, "gate": {"type": "softmax", "init_temperature": 1.0, "hidden": 64}},
    "loss": {"task": {"name": "bce", "weight": 1.0},
             "gate": {"auxiliary": 0.0, "importance": 0.03, "entropy": 0.01},
             "representation": {"name": "ntxent", "weight": 0.0, "temperature": 0.05}},
    "train": {"epochs": 100, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4,
              "clip_grad": 1.0, "balanced_sampler": True, "expert_lr_scale": 1.0},
    "optimize": {"enabled": False, "n_trials": 30, "space": {}},
}


# ------------------------------------------------------------ config plumbing
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def get_in(cfg: dict, path: str):
    node = cfg
    for part in path.split("."):
        node = node[part]
    return node


def set_in(cfg: dict, path: str, value):
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_literal(text: str):
    return yaml.safe_load(text)


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = deep_merge(DEFAULTS, yaml.safe_load(f))
    for item in overrides or []:
        key, _, value = item.partition("=")
        set_in(cfg, key.strip(), parse_literal(value))
    cfg["run"]["out_dir"] = os.path.abspath(cfg["run"]["out_dir"])
    return cfg


def suggest_cfg(trial, cfg: dict) -> dict:
    """Every entry of optimize.space is a dotted config path."""
    out = copy.deepcopy(cfg)
    for path, spec in (cfg["optimize"].get("space") or {}).items():
        kind = spec["type"]
        if kind == "float":
            v = trial.suggest_float(path, spec["low"], spec["high"],
                                    log=spec.get("log", False), step=spec.get("step"))
        elif kind == "int":
            v = trial.suggest_int(path, spec["low"], spec["high"], step=spec.get("step", 1))
        elif kind == "categorical":
            v = trial.suggest_categorical(path, spec["choices"])
        else:
            raise ValueError(f"unknown search type {kind!r} for {path}")
        set_in(out, path, v)
    return out


# ------------------------------------------------------------------- the run
def describe(cfg: dict, dims: dict):
    print(f"[config] gate={cfg['model']['gate']['type']} "
          f"| task={cfg['loss']['task']['name']} "
          f"| gate losses={ {k: v for k, v in cfg['loss']['gate'].items() if v} } "
          f"| representation={cfg['loss']['representation'].get('name')}"
          f"@{cfg['loss']['representation'].get('weight')}")
    for k, e in cfg["model"]["experts"].items():
        src = os.path.basename(e["pretrained"]) if e.get("pretrained") else "scratch"
        print(f"  {k}: dim={dims[k]} arch={e.get('arch', 'mlp')} init={src} "
              f"unfreeze_last_n={e.get('unfreeze_last_n', -1 if not e.get('pretrained') else 0)} "
              f"lr={e.get('lr', cfg['train']['lr']):.1e}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", help="YAML config")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="path.to.key=value", help="override a config entry")
    ap.add_argument("--selftest", action="store_true",
                    help="run the whole pipeline on synthetic data and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.config:
        ap.error("a config is required (or --selftest)")

    cfg = load_config(args.config, args.overrides)
    run_cfg = cfg["run"]
    os.makedirs(run_cfg["out_dir"], exist_ok=True)
    set_seed(run_cfg["seed"])
    device = get_device(run_cfg.get("gpu"), threads=run_cfg.get("threads") or None)

    views = load_views(cfg["data"])
    dev = views.subset((views.rows["split"] == "development").to_numpy())
    ext_mask = (views.rows["split"] == "external").to_numpy()
    ext = views.subset(ext_mask) if ext_mask.any() else None
    dims = dev.input_dims
    if set(dims) != set(cfg["model"]["experts"]):
        raise KeyError(f"views {sorted(dims)} do not match the experts "
                       f"{sorted(cfg['model']['experts'])}")
    describe(cfg, dims)

    objective_name = run_cfg["objective"]
    metrics_tsv = os.path.join(run_cfg["out_dir"], "trials_metrics.tsv")

    def run_cv(trial_cfg, trial=None):
        t0 = time.time()
        model_fn = lambda fold: build_moe(dims, trial_cfg["model"]["experts"], trial_cfg["model"])
        report = None
        if trial is not None:
            def report(step, running):
                trial.report(running, step)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
        oof, gates, pair = cross_val(dev, trial_cfg, device, model_fn, seed=run_cfg["seed"],
                                     report_every=1 if trial is not None else 0,
                                     report_fn=report)
        score, m = score_oof(dev.y.numpy(), oof, pair, objective_name)
        gate_mean = np.nanmean(gates, axis=0)
        row = {"run": run_cfg["name"], "trial": -1 if trial is None else trial.number,
               "score": score, **m, "sec": round(time.time() - t0, 1),
               **{f"gate_{k}": float(g) for k, g in zip(dev.x, gate_mean)}}
        append_tsv(metrics_tsv, row)
        print(f"  [trial {row['trial']:3d}] {objective_name}={score:.3f} "
              f"CV AUC={m['cv_auc']:.3f} | gate "
              + " ".join(f"{k} {g:.2f}" for k, g in zip(dev.x, gate_mean))
              + f" ({row['sec']}s)", flush=True)
        return score, m, oof, gates, pair

    if cfg["optimize"].get("enabled") and cfg["optimize"].get("space"):
        import optuna

        best = {"score": -np.inf}

        def objective(trial):
            trial_cfg = suggest_cfg(trial, cfg)
            score, m, oof, gates, pair = run_cv(trial_cfg, trial)
            if score > best["score"]:
                best.update({"score": score, "cfg": trial_cfg, "metrics": m,
                             "oof": oof, "gates": gates, "trial": trial.number})
            return score

        storage = cfg["optimize"].get("storage") or \
            f"sqlite:///{os.path.join(run_cfg['out_dir'], 'optuna.db')}"
        study = optuna.create_study(
            study_name=cfg["optimize"].get("study_name", run_cfg["name"]),
            storage=storage, direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=run_cfg["seed"], multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3))
        print(f"\n[optimize] {cfg['optimize']['n_trials']} trials over "
              f"{list(cfg['optimize']['space'])}", flush=True)
        study.optimize(objective, n_trials=int(cfg["optimize"]["n_trials"]),
                       gc_after_trial=True)
        if not np.isfinite(best["score"]):
            raise RuntimeError("no trial finished")
        print(f"[optimize] best trial {best['trial']}: {best['score']:.4f}")
    else:
        score, m, oof, gates, pair = run_cv(cfg)
        best = {"score": score, "cfg": cfg, "metrics": m, "oof": oof,
                "gates": gates, "trial": -1}

    # ---- one final model, refit on the whole development set
    final_cfg = best["cfg"]
    set_seed(run_cfg["seed"])
    model = build_moe(dims, final_cfg["model"]["experts"], final_cfg["model"])
    print(f"\n[refit] best score {best['score']:.4f} -> training on all development rows")
    model = train_model(model, dev.x, dev.y, final_cfg, device,
                        pair_id=dev.rows["pair_id"].to_numpy(),
                        log_every=run_cfg["log_every"], keep_history=True)

    p_dev, g_dev = predict(model, dev.x, device)
    extra = {"cv_auc": best["metrics"]["cv_auc"]}
    arrays = {"dev_sample": dev.rows["sample"].to_numpy(),
              "dev_pair_id": dev.rows["pair_id"].to_numpy(),
              "dev_label": dev.y.numpy(), "dev_oof": best["oof"],
              "dev_gate_oof": best["gates"], "dev_fit": p_dev, "dev_gate": g_dev}
    if ext is not None:
        p_ext, g_ext = predict(model, ext.x, device)
        pat, p_pat = aggregate_rows(ext.rows["pair_id"], p_ext)
        _, y_pat = aggregate_rows(ext.rows["pair_id"], ext.y.numpy())
        m_ext = binary_metrics(y_pat, p_pat, prefix="ext_")
        extra["ext_auc"] = m_ext["ext_auc"]
        print(f"[external] AUC={m_ext['ext_auc']:.3f} AP={m_ext['ext_ap']:.3f} "
              f"(n={m_ext['ext_n']} subjects)")
        arrays |= {"ext_sample": ext.rows["sample"].to_numpy(),
                   "ext_pair_id": ext.rows["pair_id"].to_numpy(),
                   "ext_label": ext.y.numpy(), "ext_prob": p_ext, "ext_gate": g_ext}

    entry_dir = os.path.join(run_cfg["out_dir"], run_cfg["name"])
    save_bundle(model, entry_dir, run_cfg["name"], final_cfg, best["score"],
                example_inputs=dev.x)
    np.savez(os.path.join(entry_dir, "pred.npz"), **arrays)
    append_tsv(os.path.join(run_cfg["out_dir"], "runs.tsv"),
               {"run": run_cfg["name"], "score": best["score"],
                "best_trial": best["trial"], **extra, "dir": entry_dir})
    print(f"[saved] {entry_dir}")
    return 0


def _selftest():
    """Both modes end to end on synthetic data: train from scratch, then re-use
    the saved experts as `pretrained` with only the last layer unfrozen."""
    import tempfile
    from moe_data import write_synthetic

    with tempfile.TemporaryDirectory() as d:
        data_cfg = write_synthetic(d, n_samples=36, dims=(8, 6), n_folds=3)
        base = {
            "run": {"name": "scratch", "out_dir": os.path.join(d, "result"),
                    "threads": 2, "log_every": 0, "seed": 0},
            "data": data_cfg,
            "model": {"gate": {"type": "attention"}, "experts": {
                "OMICS_01": {"arch": "mlp", "hidden_dims": [8, 4]},
                "OMICS_02": {"arch": "cnn1d", "token_dim": 2, "channels": [4, 4]}}},
            "loss": {"task": {"name": "bce"},
                     "representation": {"name": "vicreg", "weight": 0.1}},
            "train": {"epochs": 3, "batch_size": 8, "lr": 1e-3},
        }
        p = os.path.join(d, "scratch.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(base, f)
        assert main([p]) == 0
        trained = os.path.join(d, "result", "scratch")
        assert os.path.exists(os.path.join(trained, "expert_OMICS_01.pth"))

        # mode 2: the experts of run 1 are now the pretrained .pth
        gate_only = copy.deepcopy(base)
        gate_only["run"]["name"] = "gate_only"
        for k, e in gate_only["model"]["experts"].items():
            e["pretrained"] = os.path.join(trained, f"expert_{k}.pth")
            e["unfreeze_last_n"] = 1
        gate_only["optimize"] = {"enabled": True, "n_trials": 2, "space": {
            "train.lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
            "model.experts.OMICS_01.unfreeze_last_n": {"type": "categorical",
                                                       "choices": [0, 1]},
            "loss.gate.entropy": {"type": "categorical", "choices": [0.0, 0.03]}}}
        p2 = os.path.join(d, "gate_only.yaml")
        with open(p2, "w") as f:
            yaml.safe_dump(gate_only, f)
        assert main([p2]) == 0
        out = os.path.join(d, "result", "gate_only")
        assert os.path.exists(os.path.join(out, "model_state.pth"))
        assert os.path.exists(os.path.join(out, "pred.npz"))

        # --set reaches a nested key
        cfg = load_config(p, ["train.epochs=7", "model.gate.type=softmax"])
        assert cfg["train"]["epochs"] == 7 and cfg["model"]["gate"]["type"] == "softmax"
        assert cfg["loss"]["gate"]["importance"] == 0.03, "defaults must survive the merge"
    print("run_moe selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
