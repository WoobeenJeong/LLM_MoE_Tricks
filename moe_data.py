"""Config-driven loading of N omics views.

The YAML names the views (OMICS_01, OMICS_02, ...); each one points at a table
whose index column joins to the meta.  One meta row = one modelled observation,
so repeated measurements of the same subject (e.g. two flowcell libraries) are
two rows sharing a `pair_id` - a view that only exists per subject is simply
broadcast onto both rows.

Expected meta columns (names are configurable):
  id_col       one row per modelled observation (e.g. a library)
  subject_col  repeated measurements share it; defaults to id_col
  label_col    0/1
  split_col    development | external
  fold_col     cross-validation fold - the same for every row of a subject

Downstream everything uses the canonical names row_id / sample / label / split /
fold / pair_id, and a view joins on either `row_id` or `sample` (`join_on`).

Self-check:  python moe_data.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class Views:
    rows: pd.DataFrame                  # sample / label / split / fold / pair_id
    x: dict[str, torch.Tensor]          # view name -> (n_rows, n_features)
    features: dict[str, list]           # view name -> column names

    @property
    def input_dims(self) -> dict:
        return {k: int(v.shape[1]) for k, v in self.x.items()}

    def subset(self, mask) -> "Views":
        mask = np.asarray(mask)
        return Views(self.rows[mask].reset_index(drop=True),
                     {k: v[torch.tensor(mask)] for k, v in self.x.items()},
                     self.features)

    @property
    def y(self) -> torch.Tensor:
        return torch.tensor(self.rows["label"].to_numpy(dtype=np.float32))


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def load_views(data_cfg: dict, verbose: bool = True) -> Views:
    meta = read_table(data_cfg["meta"])
    id_col = data_cfg.get("id_col", "sample")
    subject_col = data_cfg.get("subject_col") or id_col
    cols = {"row_id": id_col, "sample": subject_col,
            "label": data_cfg.get("label_col", "label"),
            "split": data_cfg.get("split_col", "split"),
            "fold": data_cfg.get("fold_col", "fold")}
    missing = [c for c in cols.values() if c not in meta.columns]
    if missing:
        raise KeyError(f"{data_cfg['meta']} is missing column(s) {sorted(set(missing))}")

    rows = pd.DataFrame({
        "row_id": meta[id_col].astype(str).to_numpy(),
        "sample": meta[subject_col].astype(str).to_numpy(),
        "label": meta[cols["label"]].to_numpy().astype(np.float32),
        "split": meta[cols["split"]].astype(str).to_numpy(),
        "fold": meta[cols["fold"]].to_numpy(),
    })
    # subject == pair group: rows sharing it are the repeated measurements the
    # representation losses pair up.  With subject_col == id_col every row is its
    # own group, which turns those losses off by construction.
    rows["pair_id"] = pd.factorize(rows["sample"])[0]
    if rows.groupby("pair_id")["fold"].nunique().max() > 1:
        raise ValueError("a subject spans several folds - its replicates would be "
                         "split between train and test")
    for c in data_cfg.get("keep_meta_cols", []):
        if c in meta.columns and c not in rows.columns:
            rows[c] = meta[c].to_numpy()

    x, features, drop = {}, {}, np.zeros(len(rows), dtype=bool)
    for name, vcfg in data_cfg["omics"].items():
        tab = read_table(vcfg["path"])
        idx = vcfg.get("index_col", id_col)
        if idx not in tab.columns:
            raise KeyError(f"[{name}] {vcfg['path']} has no index column {idx!r}")
        tab = tab.set_index(tab[idx].astype(str)).drop(columns=[idx])
        tab = tab[~tab.index.duplicated(keep="first")]

        feat = vcfg.get("columns") or [c for c in tab.columns
                                       if pd.api.types.is_numeric_dtype(tab[c])]
        on = vcfg.get("join_on", "row_id")
        if on not in ("row_id", "sample"):
            raise ValueError(f"[{name}] join_on must be 'row_id' or 'sample', got {on!r}")
        # a per-subject view is simply broadcast onto every row of that subject
        join_on = rows[on].astype(str)
        aligned = tab.reindex(join_on)[feat]
        drop |= aligned.isna().any(axis=1).to_numpy()
        x[name] = aligned
        features[name] = list(feat)
        if verbose:
            print(f"[{name}] {os.path.basename(vcfg['path'])}: {len(feat)} features")

    if drop.any():
        lost = rows.loc[drop, "sample"].nunique()
        print(f"[views] dropping {int(drop.sum())} rows ({lost} samples) "
              f"without every view")
        rows = rows[~drop].reset_index(drop=True)
        x = {k: v[~drop] for k, v in x.items()}

    out = Views(rows=rows,
                x={k: torch.tensor(v.to_numpy(dtype=np.float32)) for k, v in x.items()},
                features=features)
    if verbose:
        print(f"[views] {len(rows)} rows / {rows['sample'].nunique()} samples "
              f"| dims {out.input_dims}")
        print(rows.groupby(["split"]).agg(rows=("sample", "size"),
                                          samples=("sample", "nunique"),
                                          positives=("label", "sum")).to_string())
    return out


def aggregate_rows(pair_id, values):
    """Row-level values -> one value per subject (mean over the replicates).
    Returns (unique pair ids, aggregated values)."""
    pair_id = np.asarray(pair_id)
    values = np.asarray(values, dtype=float)
    uniq = np.unique(pair_id)
    if values.ndim == 1:
        agg = np.array([np.nanmean(values[pair_id == p]) for p in uniq])
    else:
        agg = np.vstack([np.nanmean(values[pair_id == p], axis=0) for p in uniq])
    return uniq, agg


def write_synthetic(out_dir: str, n_samples: int = 60, dims=(32, 16), n_rep: int = 2,
                    n_folds: int = 3, seed: int = 0) -> dict:
    """A tiny synthetic dataset in the expected layout - used by the self-checks
    and as a template for what a real config has to point at."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    samples = [f"S{i:03d}" for i in range(n_samples)]
    label = rng.integers(0, 2, n_samples)

    meta = pd.DataFrame([
        {"sample": s, "library": f"{s}_{r}", "label": int(label[i]),
         "split": "development" if i < int(n_samples * 0.8) else "external",
         "fold": i % n_folds}
        for i, s in enumerate(samples) for r in range(n_rep)])
    meta.to_parquet(os.path.join(out_dir, "meta.parquet"))

    omics = {}
    for j, d in enumerate(dims, start=1):
        name = f"OMICS_{j:02d}"
        sig = np.repeat(label, n_rep)[:, None] * rng.normal(0.8, 0.1, size=(1, d))
        mat = rng.normal(size=(len(meta), d)) + sig
        df = pd.DataFrame(mat, columns=[f"f{k}" for k in range(d)])
        df.insert(0, "library", meta["library"].to_numpy())
        path = os.path.join(out_dir, f"{name.lower()}.parquet")
        df.to_parquet(path)
        omics[name] = {"path": path, "index_col": "library", "join_on": "row_id"}

    return {"meta": os.path.join(out_dir, "meta.parquet"),
            "id_col": "library", "subject_col": "sample", "label_col": "label",
            "split_col": "split", "fold_col": "fold", "omics": omics}


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = write_synthetic(d, n_samples=20, dims=(8, 6))
        v = load_views(cfg, verbose=False)
        assert len(v.rows) == 40 and v.input_dims == {"OMICS_01": 8, "OMICS_02": 6}
        # the two replicates of a subject share a pair_id and a fold
        assert v.rows.groupby("pair_id")["fold"].nunique().max() == 1
        assert v.rows["pair_id"].nunique() == 20
        dev = v.subset((v.rows["split"] == "development").to_numpy())
        assert len(dev.rows) == 32 and dev.x["OMICS_01"].shape == (32, 8)
        uniq, agg = aggregate_rows(v.rows["pair_id"], v.rows["label"])
        assert len(uniq) == 20 and set(np.unique(agg)) <= {0.0, 1.0}
    print("moe_data selftest OK")


if __name__ == "__main__":
    _selftest()
