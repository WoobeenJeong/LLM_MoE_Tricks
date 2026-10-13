"""Training, cross-validation and checkpointing for the generic MoE.

The config dict is the single source of truth: `train` holds the optimisation
settings, `loss` the weighted mix of task / gate / representation terms, and
each expert may carry its own `lr` (a single shared rate rarely serves views of
very different size).  An expert whose effective lr is 0 - or whose parameters
were frozen by `unfreeze_last_n` - simply contributes no parameter group.

Self-check:  python moe_train.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from moe_core import (
    MoEScriptWrapper,
    build_moe,
    build_representation_loss,
    build_task_loss,
    gate_losses,
    set_seed,
)
from moe_data import aggregate_rows


def get_device(gpu_id=None, threads: int | None = None) -> torch.device:
    """CPU threads are capped by default: these models are small, so BLAS
    over-subscription costs more than it buys on a shared machine."""
    if threads:
        torch.set_num_threads(threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
    return torch.device("cpu")


# ------------------------------------------------------------------- metrics
def binary_metrics(y_true, y_score, prefix: str = "") -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    out = {}
    if len(np.unique(y_true)) < 2:
        out[f"{prefix}auc"] = float("nan")
        out[f"{prefix}ap"] = float("nan")
    else:
        out[f"{prefix}auc"] = float(roc_auc_score(y_true, y_score))
        out[f"{prefix}ap"] = float(average_precision_score(y_true, y_score))
    best_j, best_thr, sens95 = -1.0, 0.5, float("nan")
    for t in np.unique(np.concatenate([[0.0, 1.0], y_score])):
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        if sens + spec - 1 > best_j:
            best_j, best_thr = sens + spec - 1, float(t)
        if spec >= 0.95 and (np.isnan(sens95) or sens > sens95):
            sens95 = sens
    out[f"{prefix}youden_j"] = float(best_j)
    out[f"{prefix}youden_thr"] = best_thr
    out[f"{prefix}sens_at_spec95"] = float(sens95)
    out[f"{prefix}n"] = int(len(y_true))
    out[f"{prefix}n_pos"] = int(y_true.sum())
    return out


def objective_from_oof(y_true, y_score, mode: str = "auc") -> float:
    m = binary_metrics(y_true, y_score)
    if mode == "auc":
        v = m["auc"]
    elif mode == "auc_ap":
        v = 0.5 * (m["auc"] + m["ap"])
    elif mode == "auc_youden":
        v = 0.5 * (m["auc"] + 0.5 * (m["youden_j"] + 1))
    else:
        raise ValueError(f"unknown objective: {mode}")
    return -1.0 if np.isnan(v) else float(v)


# ---------------------------------------------------------------- train loop
def pair_batches(pair_id, labels, n_per_batch: int, rng, balanced: bool = True):
    """Batch at the *subject* level so that the replicates of a subject always
    land in the same batch - the representation losses need the positive present.
    Class balancing therefore samples subjects, not rows."""
    groups, group_label = [], []
    for pid in np.unique(pair_id):
        idx = np.where(pair_id == pid)[0]
        groups.append(idx)
        group_label.append(int(labels[idx[0]]))
    group_label = np.asarray(group_label)
    n_groups = len(groups)
    per_batch = max(1, min(n_per_batch, n_groups))

    if balanced:
        cnt = np.bincount(group_label, minlength=2).astype(float)
        w = (1.0 / np.clip(cnt, 1, None))[group_label]
        order = rng.choice(n_groups, size=n_groups, replace=True, p=w / w.sum())
    else:
        order = rng.permutation(n_groups)

    batches = []
    for s in range(0, n_groups, per_batch):
        rows = np.concatenate([groups[g] for g in order[s:s + per_batch]])
        if len(rows) > 1:          # BatchNorm cannot run on a single row
            batches.append(rows)
    return batches


def _optimizer(model, cfg) -> torch.optim.Optimizer:
    """One parameter group per expert, anchored on that expert's own lr.

    `train.expert_lr_scale` scales every expert rate at once; 0 freezes them all,
    which is the "pretrained experts, train the gate only" setting.
    """
    train_cfg = cfg["train"]
    scale = float(train_cfg.get("expert_lr_scale", 1.0))
    base_lr = float(train_cfg["lr"])
    expert_cfgs = cfg["model"]["experts"]

    groups, expert_ids = [], set()
    for i, key in enumerate(model.expert_keys):
        params = [p for p in model.experts[i].parameters() if p.requires_grad]
        expert_ids |= {id(p) for p in model.experts[i].parameters()}
        lr = float(expert_cfgs[key].get("lr", base_lr)) * scale
        if params and lr > 0:
            groups.append({"params": params, "lr": lr})
        else:
            for p in model.experts[i].parameters():
                p.requires_grad_(False)
    rest = [p for p in model.parameters() if id(p) not in expert_ids and p.requires_grad]
    groups.append({"params": rest, "lr": base_lr})
    return torch.optim.AdamW(groups, weight_decay=float(train_cfg.get("weight_decay", 0.0)))


@torch.no_grad()
def train_auc(model, xs, ys) -> float:
    """AUC on the training rows themselves - a fit monitor, not a score."""
    was = model.training
    model.eval()
    logit, _ = model(xs)
    model.train(was)
    prob = torch.sigmoid(logit.view(-1)).cpu().numpy()
    y = ys.view(-1).cpu().numpy()
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, prob))


def train_model(model, x_tr: dict, y_tr: torch.Tensor, cfg: dict, device,
                pair_id=None, log_every: int = 0, keep_history: bool = False):
    """Train for a fixed number of epochs (no inner validation split: on the
    small cohorts this is built for, the epoch count is a hyperparameter the
    outer CV objective tunes)."""
    train_cfg, loss_cfg = cfg["train"], cfg["loss"]
    model.to(device)
    xs = {k: v.to(device) for k, v in x_tr.items()}
    ys = y_tr.to(device).float().view(-1)
    model.fit_scaler(xs)

    n = len(y_tr)
    labels_np = ys.cpu().numpy().astype(int)
    pair_id = np.arange(n) if pair_id is None else np.asarray(pair_id)
    pair_t = torch.tensor(pd.factorize(pair_id)[0], dtype=torch.long, device=device)
    rng = np.random.default_rng(train_cfg.get("batch_seed", 0))

    task_fn = build_task_loss(loss_cfg.get("task", {"name": "bce"}), y_train=ys)
    task_w = float(loss_cfg.get("task", {}).get("weight", 1.0))
    rep_cfg = loss_cfg.get("representation", {}) or {}
    rep_fn = build_representation_loss(rep_cfg)
    rep_w = float(rep_cfg.get("weight", 0.0))
    gate_w = {k: float(v) for k, v in (loss_cfg.get("gate") or {}).items() if float(v) != 0}

    optimizer = _optimizer(model, cfg)
    epochs = int(train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    clip = float(train_cfg.get("clip_grad", 0.0))

    # a frozen expert must stay in eval mode so its dropout / BatchNorm statistics
    # do not keep moving while only the gate learns
    frozen = [m for m in model.experts if not any(p.requires_grad for p in m.parameters())]

    def set_train_mode():
        model.train()
        for m in frozen:
            m.eval()

    if log_every:
        shapes = " | ".join(f"{k} {tuple(v.shape)}" for k, v in xs.items())
        print(f"  [input] {shapes} | y {tuple(ys.shape)} "
              f"({int(labels_np.sum())}/{n} positive rows, "
              f"{len(np.unique(pair_id))} subjects) | epochs {epochs}")

    history = [] if keep_history else None
    set_train_mode()
    for ep in range(epochs):
        acc = {"total": 0.0, "task": 0.0, "representation": 0.0,
               **{k: 0.0 for k in gate_w}}
        seen = 0
        for rows in pair_batches(pair_id, labels_np, int(train_cfg["batch_size"]), rng,
                                 balanced=train_cfg.get("balanced_sampler", True)):
            sel = torch.tensor(rows, dtype=torch.long, device=device)
            xb = {k: v[sel] for k, v in xs.items()}
            yb, pb = ys[sel], pair_t[sel]
            optimizer.zero_grad()

            logit, gate_probs, rep = model.forward_with_rep(xb)
            task = task_fn(logit, yb, rep)
            loss = task_w * task
            parts = {"task": float(task)}

            if rep_fn is not None:
                r = rep_fn(rep, pb)
                loss = loss + rep_w * r
                parts["representation"] = float(r)
            if gate_w:
                gl = gate_losses(gate_probs, list(gate_w), simplex=model.gate.simplex)
                for k, w in gate_w.items():
                    loss = loss + w * gl[k]
                    parts[k] = float(gl[k])

            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], clip)
            optimizer.step()

            nb = len(yb)
            seen += nb
            acc["total"] += float(loss) * nb
            for k, v in parts.items():
                acc[k] += v * nb
        scheduler.step()

        mean = {k: v / max(seen, 1) for k, v in acc.items()}
        if history is not None:
            history.append({"epoch": ep + 1, "lr": optimizer.param_groups[-1]["lr"], **mean})
        if log_every and ((ep + 1) % log_every == 0 or ep == 0):
            terms = "  ".join(f"{k} {v:.4f}" for k, v in mean.items())
            print(f"  epoch {ep+1:4d}/{epochs}  {terms}  | train AUC "
                  f"{train_auc(model, xs, ys):.3f}", flush=True)
            set_train_mode()

    if history is not None:
        model.loss_history = history
    return model


@torch.no_grad()
def predict(model, x: dict, device):
    model.to(device).eval()
    logit, gate_probs = model({k: v.to(device) for k, v in x.items()})
    return (torch.sigmoid(logit.view(-1)).cpu().numpy(), gate_probs.cpu().numpy())


# ------------------------------------------------------------ cross-validation
def cross_val(views, cfg: dict, device, model_fn, seed: int = 42,
              report_every: int = 0, report_fn=None):
    """K-fold CV over the development rows.

    The fold of a subject comes from the meta, so every replicate of a subject is
    held out at once and a subject can never be scored by a model that saw its
    replicate.  Returns (oof, gates, pair_id), all row-aligned.
    """
    y = views.y
    folds = views.rows["fold"].to_numpy()
    pair_id = views.rows["pair_id"].to_numpy()
    n = len(y)
    oof = np.full(n, np.nan)
    gates = np.full((n, len(views.x)), np.nan)

    for step, f in enumerate(np.unique(folds), start=1):
        te = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        set_seed(seed + int(f))
        model = model_fn(f)
        x_tr = {k: v[tr] for k, v in views.x.items()}
        x_te = {k: v[te] for k, v in views.x.items()}
        model = train_model(model, x_tr, y[tr],
                            {**cfg, "train": {**cfg["train"], "batch_seed": int(f)}},
                            device, pair_id=pair_id[tr])
        prob, gate = predict(model, x_te, device)
        oof[te], gates[te] = prob, gate
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if report_every and report_fn and step % report_every == 0:
            done = ~np.isnan(oof)
            _, p = aggregate_rows(pair_id[done], oof[done])
            _, t = aggregate_rows(pair_id[done], y.numpy()[done])
            report_fn(step, objective_from_oof(t, p, cfg["run"].get("objective", "auc")))

    return oof, gates, pair_id


def score_oof(y, oof, pair_id, objective: str = "auc"):
    """Subject-level score: the replicate predictions are averaged first."""
    _, p = aggregate_rows(pair_id, oof)
    _, t = aggregate_rows(pair_id, np.asarray(y, dtype=float))
    return objective_from_oof(t, p, objective), binary_metrics(t, p, prefix="cv_")


# ------------------------------------------------------- checkpoint bundling
def save_bundle(model, out_dir: str, tag: str, config: dict, score: float,
                example_inputs: dict) -> list[str]:
    """model.pth (pickled) / model_state.pth (state_dict + config) /
    expert_<view>.pth + moe_head.pth (re-usable parts) / model_scripted.pth /
    model_config.yaml / history.tsv"""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    model = model.to("cpu").eval()

    torch.save(model, os.path.join(out_dir, "model.pth"))
    torch.save({"state_dict": model.state_dict(), "config": config,
                "score": score, "tag": tag}, os.path.join(out_dir, "model_state.pth"))
    written += [os.path.join(out_dir, f) for f in ("model.pth", "model_state.pth")]

    # the parts, so a trained expert can be re-used as `pretrained:` next time
    for i, key in enumerate(model.expert_keys):
        p = os.path.join(out_dir, f"expert_{key}.pth")
        torch.save({"state_dict": model.experts[i].state_dict(),
                    "config": config["model"]["experts"][key], "part": key, "tag": tag}, p)
        written.append(p)
    head = {k: v for k, v in model.state_dict().items() if not k.startswith("experts.")}
    p = os.path.join(out_dir, "moe_head.pth")
    torch.save({"state_dict": head, "config": config, "part": "moe_head", "tag": tag}, p)
    written.append(p)

    p = os.path.join(out_dir, "model_scripted.pth")
    try:
        with torch.no_grad():
            ex = tuple(example_inputs[k][:2].to("cpu") for k in model.expert_keys)
            torch.jit.save(torch.jit.trace(MoEScriptWrapper(model).eval(), ex, strict=False), p)
        written.append(p)
    except Exception as e:                      # never lose a run over a trace quirk
        print(f"  [warn] TorchScript export failed for {tag}: {type(e).__name__}: {e}")

    p = os.path.join(out_dir, "model_config.yaml")
    with open(p, "w") as f:
        yaml.safe_dump({"tag": tag, "score": score,
                        "config": json.loads(json.dumps(config, default=str))},
                       f, sort_keys=False)
    written.append(p)

    history = getattr(model, "loss_history", None)
    if history:
        rows = [{"tag": tag, "epoch": h["epoch"], "metric_type": "losses",
                 "metric_name": k, "metric_value": v, "data_type": "train",
                 "lr": h["lr"]}
                for h in history for k, v in h.items() if k not in ("epoch", "lr")]
        p = os.path.join(out_dir, "history.tsv")
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
        written.append(p)
    return written


def append_tsv(path: str, row: dict):
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a",
                               header=not os.path.exists(path))


def _selftest():
    """End-to-end on synthetic data: CV runs, the OOF is complete, and with
    every expert frozen only the gate/classifier move."""
    import tempfile
    from moe_data import load_views, write_synthetic

    device = get_device(threads=2)
    with tempfile.TemporaryDirectory() as d:
        data_cfg = write_synthetic(d, n_samples=24, dims=(8, 6), n_folds=3)
        views = load_views(data_cfg, verbose=False)
        dev = views.subset((views.rows["split"] == "development").to_numpy())

        cfg = {
            "run": {"objective": "auc"},
            "model": {"gate": {"type": "attention"}, "experts": {
                "OMICS_01": {"arch": "mlp", "hidden_dims": [8, 4], "lr": 1e-3},
                "OMICS_02": {"arch": "transformer", "token_dim": 2, "d_model": 8,
                             "n_heads": 2, "lr": 1e-3}}},
            "loss": {"task": {"name": "bce"},
                     "gate": {"importance": 0.03, "entropy": 0.01},
                     "representation": {"name": "ntxent", "weight": 0.1}},
            "train": {"epochs": 3, "batch_size": 8, "lr": 1e-3, "weight_decay": 1e-4,
                      "clip_grad": 1.0, "balanced_sampler": True},
        }
        model_fn = lambda fold: build_moe(dev.input_dims, cfg["model"]["experts"], cfg["model"])
        oof, gates, pair = cross_val(dev, cfg, device, model_fn, seed=0)
        assert not np.isnan(oof).any(), "every development row must get an OOF prediction"
        assert np.allclose(gates.sum(axis=1), 1.0, atol=1e-4)
        score, m = score_oof(dev.y.numpy(), oof, pair, "auc")
        assert 0.0 <= score <= 1.0 and m["cv_n"] == dev.rows["pair_id"].nunique()

        # frozen experts: their weights must be bit-identical after training
        frozen_cfg = {**cfg, "model": {**cfg["model"], "experts": {
            k: {**v, "unfreeze_last_n": 0} for k, v in cfg["model"]["experts"].items()}}}
        model = build_moe(dev.input_dims, frozen_cfg["model"]["experts"], frozen_cfg["model"])
        before = [p.detach().clone() for p in model.experts.parameters()]
        gate_before = model.classifier.weight.detach().clone()
        model = train_model(model, dev.x, dev.y, frozen_cfg, device,
                            pair_id=dev.rows["pair_id"].to_numpy(), keep_history=True)
        after = [p.detach().cpu() for p in model.experts.parameters()]
        assert all(torch.equal(a, b) for a, b in zip(before, after)), "frozen expert moved"
        assert not torch.equal(gate_before, model.classifier.weight.detach().cpu())
        assert len(model.loss_history) == 3

        files = save_bundle(model, os.path.join(d, "bundle"), "selftest", frozen_cfg,
                            score, dev.x)
        assert any(f.endswith("expert_OMICS_01.pth") for f in files)
    print("moe_train selftest OK")


if __name__ == "__main__":
    _selftest()
