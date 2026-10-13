"""Generic multi-omics Mixture-of-Experts: experts, gate, losses, checkpoints.

Nothing here knows how many omics views there are or what they are called - the
view names (OMICS_01, OMICS_02, ...) and every architecture / loss choice come
from the YAML config, so the same code serves

  * the from-scratch model  (experts + gate + classifier trained at once)
  * the pretrained model    (expert .pth given; only the last N layers of each
                             expert are unfrozen, the gate / classifier train)

Only the loss names listed in TASK_LOSSES / GATE_LOSSES / REPRESENTATION_LOSSES
are accepted; an unknown name raises instead of being silently ignored.

Self-check:  python moe_core.py
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "mish": nn.Mish,
               "silu": nn.SiLU, "gelu": nn.GELU, "selu": nn.SELU}

ARCHS = ("mlp", "cnn1d", "transformer")
TASK_LOSSES = ("bce", "wbce", "focal", "triplet")
GATE_LOSSES = ("auxiliary", "importance", "entropy")
REPRESENTATION_LOSSES = ("ntxent", "barlow_twins", "vicreg")
GATE_TYPES = ("sigmoid", "softmax", "gumbel", "attention")
GATE_ALIASES = {"mome": "attention"}   # the earlier pipelines called it 'mome'


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------- experts
class StandardScalerLayer(nn.Module):
    """Standardisation with the statistics frozen at fit time; they are buffers,
    so they travel with the state_dict."""

    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        self.mean.copy_(x.mean(dim=0))
        self.std.copy_(torch.clamp(x.std(dim=0, unbiased=False), min=1e-8))
        return self

    def forward(self, x):
        return (x - self.mean) / self.std


class _Expert(nn.Module):
    """Common contract: forward(x) -> (representation, logit)."""

    def fit_scaler(self, x: torch.Tensor):
        if getattr(self, "scaler", None) is not None:
            self.scaler.fit(x)
        return self

    def _scale(self, x):
        return self.scaler(x) if getattr(self, "scaler", None) is not None else x


class MLPExpert(_Expert):
    def __init__(self, input_dim: int, hidden_dims=(256, 64), dropout: float = 0.25,
                 activation: str = "relu", batch_norm: bool = True,
                 num_classes: int = 1, with_scaler: bool = True, **_):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = int(hidden_dims[-1])
        self.scaler = StandardScalerLayer(input_dim) if with_scaler else None

        act = ACTIVATIONS[activation]
        layers, d = [], input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.body = nn.Sequential(*layers)
        self.head = _linear_head(d, num_classes)

    def forward(self, x):
        feat = self.body(self._scale(x))
        return feat, self.head(feat)


class CNN1DExpert(_Expert):
    """1D-CNN over the feature axis.

    `token_dim` is the number of input channels: the flat vector is read as
    (n_tokens, token_dim) and transposed, so token_dim=1 means one channel over
    the whole feature axis and token_dim=16 means 16 channels of n_tokens steps.
    """

    def __init__(self, input_dim: int, token_dim: int = 1, channels=(32, 64),
                 kernel_size: int = 5, pool: int = 2, dropout: float = 0.1,
                 activation: str = "relu", num_classes: int = 1,
                 with_scaler: bool = True, **_):
        super().__init__()
        if input_dim % token_dim:
            raise ValueError(f"input_dim {input_dim} is not a multiple of token_dim {token_dim}")
        self.input_dim = input_dim
        self.token_dim = token_dim
        self.n_tokens = input_dim // token_dim
        self.output_dim = int(channels[-1])
        self.scaler = StandardScalerLayer(input_dim) if with_scaler else None

        act = ACTIVATIONS[activation]
        layers, c, length = [], token_dim, self.n_tokens
        for ch in channels:
            layers += [nn.Conv1d(c, ch, kernel_size, padding=kernel_size // 2),
                       nn.BatchNorm1d(ch), act()]
            if pool > 1 and length >= pool:   # a short view runs out of length first
                layers.append(nn.MaxPool1d(pool))
                length //= pool
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            c = ch
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.body = nn.Sequential(*layers)
        self.head = _linear_head(self.output_dim, num_classes)

    def forward(self, x):
        h = self._scale(x).view(x.shape[0], self.n_tokens, self.token_dim).transpose(1, 2)
        feat = self.body(h).flatten(1)
        return feat, self.head(feat)


class TransformerExpert(_Expert):
    """Transformer encoder over feature tokens: the flat vector is read as
    n_tokens tokens of token_dim channels (token_dim=1 is FT-Transformer style,
    one token per feature)."""

    def __init__(self, input_dim: int, token_dim: int = 1, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 1, dropout: float = 0.1,
                 ff_mult: int = 2, pool: str = "cls", num_classes: int = 1,
                 with_scaler: bool = True, **_):
        super().__init__()
        if input_dim % token_dim:
            raise ValueError(f"input_dim {input_dim} is not a multiple of token_dim {token_dim}")
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} is not a multiple of n_heads {n_heads}")
        self.input_dim = input_dim
        self.token_dim = token_dim
        self.n_tokens = input_dim // token_dim
        self.output_dim = d_model
        self.scaler = StandardScalerLayer(input_dim) if with_scaler else None

        self.embed = nn.Linear(token_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if pool == "cls" else None
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        # norm_first disables the nested-tensor fast path anyway; saying so keeps
        # torch from warning once per constructed model
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = _linear_head(d_model, num_classes)

    def forward(self, x):
        x = self._scale(x)
        h = self.embed(x.view(x.shape[0], self.n_tokens, self.token_dim)) + self.pos
        if self.cls is not None:
            h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
        h = self.encoder(h)
        feat = self.norm(h[:, 0] if self.cls is not None else h.mean(dim=1))
        return feat, self.head(feat)


def _linear_head(d: int, num_classes: int) -> nn.Linear:
    head = nn.Linear(d, num_classes)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    return head


_EXPERT_CLASSES = {"mlp": MLPExpert, "cnn1d": CNN1DExpert, "transformer": TransformerExpert}


class ScriptedExpert(nn.Module):
    """A pretrained TorchScript module made to look like an _Expert.

    Its output dim is unknown until it runs, so it is measured once with a dummy
    batch at build time.
    """

    def __init__(self, module: torch.jit.ScriptModule, input_dim: int):
        super().__init__()
        self.module = module
        self.input_dim = input_dim
        self.scaler = None
        with torch.no_grad():
            was = self.module.training
            self.module.eval()
            feat, _ = _split_expert_output(self.module(torch.zeros(2, input_dim)))
            self.module.train(was)
        self.output_dim = int(feat.shape[-1])

    def fit_scaler(self, x):  # the scripted module carries its own normalisation
        return self

    def forward(self, x):
        return _split_expert_output(self.module(x))


def _split_expert_output(out):
    """(feat, logit) out of whatever an expert returns."""
    if isinstance(out, (tuple, list)):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


def build_expert(input_dim: int, cfg: dict) -> nn.Module:
    """cfg: {arch: mlp|cnn1d|transformer, pretrained: path|null, ...arch args}."""
    arch = cfg.get("arch", "mlp")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}, pick one of {ARCHS}")
    path = cfg.get("pretrained")
    if path:
        return load_pretrained_expert(path, input_dim, cfg)
    kwargs = {k: v for k, v in cfg.items()
              if k not in ("arch", "pretrained", "unfreeze_last_n", "lr")}
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in kwargs.items()}
    return _EXPERT_CLASSES[arch](input_dim=input_dim, **kwargs)


def load_pretrained_expert(path: str, input_dim: int, cfg: dict) -> nn.Module:
    """Accepts a TorchScript file, a pickled module, or a state_dict bundle
    ({"state_dict": ...} or a bare state_dict) rebuilt into cfg["arch"]."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"pretrained expert not found: {path}")
    try:
        return ScriptedExpert(torch.jit.load(path, map_location="cpu"), input_dim)
    except (RuntimeError, ValueError):
        pass
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    arch_cfg = {**(obj.get("config", {}) if isinstance(obj, dict) else {}), **cfg}
    model = build_expert(input_dim, {**arch_cfg, "pretrained": None})
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [pretrained] {os.path.basename(path)}: "
              f"{len(missing)} missing / {len(unexpected)} unexpected keys")
    return model


# ------------------------------------------------- freezing / partial unfreeze
def param_modules(module: nn.Module):
    """Sub-modules that own parameters, in definition order."""
    return [(n, m) for n, m in module.named_modules()
            if n and sum(p.numel() for p in m.parameters(recurse=False)) > 0]


def set_trainable(module: nn.Module, unfreeze_last_n: int) -> int:
    """How deep to train a pretrained expert.

      -1  everything trainable (the from-scratch case)
       0  fully frozen - only the gate / classifier learn
       n  the last n parameter-owning sub-modules are unfrozen

    Returns the number of trainable parameters.
    """
    for p in module.parameters():
        p.requires_grad_(unfreeze_last_n < 0)
    if unfreeze_last_n > 0:
        for _, m in param_modules(module)[-unfreeze_last_n:]:
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ------------------------------------------------------------------ gate / MoE
class MoEGate(nn.Module):
    """Four gate structures over the concatenated (normalised) views:

      sigmoid    one linear layer, an INDEPENDENT weight per expert.  The
                 weights do not sum to 1, so experts can be on or off together -
                 a soft ensemble rather than a routing decision.
      softmax    one linear layer, a simplex over the experts (the usual MoE).
      gumbel     softmax with Gumbel noise while training, so the gate learns a
                 near-discrete routing; plain softmax at eval time.
      attention  per view a gated-tanh block (tanh * sigmoid, the MoME gate),
                 then one linear layer to the expert logits + softmax.  The
                 gate can weigh *parts* of a view instead of the view as a whole.

    `temperature` is learned (log-parameterised) in every case.
    """

    def __init__(self, input_dims, num_experts: int, gate_type: str = "softmax",
                 init_temperature: float = 1.0, hidden: int = 64):
        super().__init__()
        gate_type = GATE_ALIASES.get(gate_type, gate_type)
        if gate_type not in GATE_TYPES:
            raise ValueError(f"unknown gate {gate_type!r}, pick one of {GATE_TYPES}")
        self.input_dims = list(input_dims)
        self.num_experts = num_experts
        self.kind = gate_type
        # a sigmoid gate is the one structure whose weights are not a distribution
        self.simplex = gate_type != "sigmoid"

        self.gate = nn.Linear(sum(self.input_dims), num_experts)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temperature))))

        if gate_type == "attention":
            # hidden size capped per view so the gate stays small relative to the
            # (usually small) sample size
            hs = [min(hidden, d) for d in self.input_dims]
            self.attn_tanh = nn.ModuleList(
                [nn.Sequential(nn.Linear(d, h), nn.Tanh()) for d, h in zip(self.input_dims, hs)])
            self.attn_sig = nn.ModuleList(
                [nn.Sequential(nn.Linear(d, h), nn.Sigmoid()) for d, h in zip(self.input_dims, hs)])
            self.attn_lin = nn.Linear(sum(hs), num_experts)
            nn.init.xavier_uniform_(self.attn_lin.weight)
            nn.init.zeros_(self.attn_lin.bias)

    def forward(self, x_list, hard: bool = False, eps: float = 1e-8):
        temperature = F.softplus(self.log_temperature) + eps
        if self.kind == "attention":
            c = [self.attn_tanh[i](x).mul(self.attn_sig[i](x)) for i, x in enumerate(x_list)]
            logits = self.attn_lin(torch.cat(c, dim=-1))
        else:
            logits = self.gate(torch.cat(x_list, dim=-1))

        if self.kind == "sigmoid":
            return torch.sigmoid(logits / temperature)
        if self.kind == "gumbel" and self.training:
            return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        return F.softmax(logits / temperature, dim=-1)


class MoE(nn.Module):
    """N experts + gate + joint classifier.

    Which parts actually learn is decided outside (set_trainable per expert), so
    "train everything from scratch" and "frozen pretrained experts, gate only"
    are the same module with different requires_grad flags.
    """

    def __init__(self, experts: dict, gate_type: str = "softmax", num_classes: int = 1,
                 init_temperature: float = 1.0, gate_hidden: int = 64):
        super().__init__()
        self.expert_keys = list(experts.keys())
        self.experts = nn.ModuleList([experts[k] for k in self.expert_keys])
        self.expert_input_dims = [experts[k].input_dim for k in self.expert_keys]
        self.expert_output_dims = [experts[k].output_dim for k in self.expert_keys]
        self.num_experts = len(self.expert_keys)

        self.gate_norm_layers = nn.ModuleList([nn.LayerNorm(d) for d in self.expert_input_dims])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d) for d in self.expert_output_dims])
        self.gate = MoEGate(self.expert_input_dims, self.num_experts, gate_type=gate_type,
                            init_temperature=init_temperature, hidden=gate_hidden)
        self.classifier = _linear_head(sum(self.expert_output_dims), num_classes)

    def fit_scaler(self, x_dict: dict):
        for k, e in zip(self.expert_keys, self.experts):
            e.fit_scaler(x_dict[k])
        return self

    def forward(self, x_dict):
        logit, gate_probs, _ = self.forward_with_rep(x_dict)
        return logit, gate_probs

    def forward_with_rep(self, x_dict):
        """Also returns the gate-weighted joint representation (the classifier's
        input), which is what the representation losses are computed on."""
        gate_inputs = []
        for i, k in enumerate(self.expert_keys):
            x = x_dict[k]
            scaler = getattr(self.experts[i], "scaler", None)
            gate_inputs.append(self.gate_norm_layers[i](scaler(x) if scaler is not None else x))
        gate_probs = self.gate(gate_inputs)

        feats = []
        for i, k in enumerate(self.expert_keys):
            feat, _ = _split_expert_output(self.experts[i](x_dict[k]))
            feats.append(self.norm_layers[i](feat) * gate_probs[:, i].unsqueeze(1))
        rep = torch.cat(feats, dim=-1)
        return self.classifier(rep), gate_probs, rep


class MoEScriptWrapper(nn.Module):
    """Positional-argument view of a trained MoE, for torch.jit.trace (dict
    inputs are not traceable).  Call order is moe.expert_keys."""

    def __init__(self, moe: MoE):
        super().__init__()
        self.moe = moe
        self.expert_keys = list(moe.expert_keys)

    def forward(self, *views):
        return self.moe(dict(zip(self.expert_keys, views)))


def build_moe(input_dims: dict, expert_cfgs: dict, model_cfg: dict) -> MoE:
    """input_dims / expert_cfgs are keyed by view name; the key order fixes the
    expert order everywhere downstream."""
    experts = {k: build_expert(input_dims[k], expert_cfgs[k]) for k in input_dims}
    gate_cfg = model_cfg.get("gate", {})
    moe = MoE(experts=experts,
              gate_type=gate_cfg.get("type", "softmax"),
              num_classes=model_cfg.get("num_classes", 1),
              init_temperature=gate_cfg.get("init_temperature", 1.0),
              gate_hidden=gate_cfg.get("hidden", gate_cfg.get("mome_hidden", 64)))
    for i, k in enumerate(moe.expert_keys):
        cfg = expert_cfgs[k]
        # an expert built from scratch is fully trainable unless told otherwise
        default = -1 if not cfg.get("pretrained") else 0
        n = set_trainable(moe.experts[i], int(cfg.get("unfreeze_last_n", default)))
        if cfg.get("pretrained"):
            total = sum(p.numel() for p in moe.experts[i].parameters())
            print(f"  [expert {k}] pretrained, unfreeze_last_n="
                  f"{cfg.get('unfreeze_last_n', default)} -> {n:,}/{total:,} params trainable")
    return moe


# --------------------------------------------------------------------- losses
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logit, target):
        p = torch.sigmoid(logit)
        ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        pt = p * target + (1 - p) * (1 - target)
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (at * (1 - pt).pow(self.gamma) * ce).mean()


def triplet_loss(rep, labels, margin: float = 1.0):
    """Batch-hard triplet over the joint representation, using the binary label
    as the identity.  0 when the batch has only one class."""
    labels = labels.view(-1)
    d = torch.cdist(rep, rep)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=rep.device)
    pos_mask, neg_mask = same & ~eye, ~same
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid.any():
        return torch.zeros((), device=rep.device)
    hardest_pos = d.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    hardest_neg = d.masked_fill(~neg_mask, float("inf")).min(dim=1).values
    return F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()


class NtXentLoss(nn.Module):
    """NT-Xent over repeated measurements of the same subject.

    The positive is given explicitly by `pair_id` (e.g. the flowcell A/B
    libraries of one patient), so a batch that also holds unpaired rows is fine:
    those rows are skipped as anchors while still acting as negatives.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, x, pair_id):
        features = F.normalize(x, p=2, dim=1)
        sim_mat = torch.matmul(features, features.T) / self.temperature

        n = features.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=features.device)
        pair_id = pair_id.view(-1, 1)
        pos = (pair_id == pair_id.t()) & ~eye
        if pos.sum() == 0:
            return torch.zeros((), device=features.device)

        # self-similarity out of the denominator; masked_fill (not a 0 multiply)
        # so that -inf never turns into NaN
        sim_mat = sim_mat.masked_fill(eye, torch.finfo(sim_mat.dtype).min)
        log_prob = sim_mat - torch.logsumexp(sim_mat, dim=1, keepdim=True)
        pos_cnt = pos.sum(dim=1)
        anchors = pos_cnt > 0
        return (-log_prob.masked_fill(~pos, 0.0).sum(dim=1)[anchors] / pos_cnt[anchors]).mean()


def split_pair_views(x, pair_id):
    """(za, zb): the first two rows of every group that has at least two.

    Barlow Twins and VICReg are defined on two aligned batches, which for
    repeated measurements means the A and B replicate of the same subject.
    """
    pair_id = pair_id.view(-1)
    a, b = [], []
    for pid in torch.unique(pair_id):
        idx = (pair_id == pid).nonzero(as_tuple=True)[0]
        if len(idx) >= 2:
            a.append(idx[0])
            b.append(idx[1])
    if not a:
        return None, None
    return x[torch.stack(a)], x[torch.stack(b)]


def barlow_twins_loss(x, pair_id, lambd: float = 5e-3, eps: float = 1e-8):
    za, zb = split_pair_views(x, pair_id)
    if za is None or za.shape[0] < 2:
        return torch.zeros((), device=x.device)
    za = (za - za.mean(0)) / (za.std(0, unbiased=False) + eps)
    zb = (zb - zb.mean(0)) / (zb.std(0, unbiased=False) + eps)
    c = (za.T @ zb) / za.shape[0]
    on_diag = (torch.diagonal(c) - 1).pow(2).sum()
    off_diag = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag


def vicreg_loss(x, pair_id, sim_coef: float = 25.0, std_coef: float = 25.0,
                cov_coef: float = 1.0, eps: float = 1e-4):
    za, zb = split_pair_views(x, pair_id)
    if za is None or za.shape[0] < 2:
        return torch.zeros((), device=x.device)
    n, d = za.shape
    sim = F.mse_loss(za, zb)
    std = sum(F.relu(1 - torch.sqrt(z.var(dim=0) + eps)).mean() for z in (za, zb))
    cov = 0.0
    for z in (za, zb):
        z = z - z.mean(dim=0)
        c = (z.T @ z) / (n - 1)
        cov = cov + (c.pow(2).sum() - torch.diagonal(c).pow(2).sum()) / d
    return sim_coef * sim + std_coef * std + cov_coef * cov


def gate_losses(gate_probs, names, simplex: bool = True, eps: float = 1e-8) -> dict:
    """The gate regularisers, all meant to be MINIMISED.

    auxiliary  : load balancing (Switch-Transformer): N * sum_i f_i * P_i, where
                 f_i is the fraction of samples whose top expert is i and P_i the
                 mean gate probability of i.  1.0 when perfectly balanced.
    importance : squared deviation of the batch-average gate from uniform - pure
                 anti-collapse, and a weaker signal than `auxiliary`.
    entropy    : mean per-sample entropy.  Minimising it lets the gate *commit*
                 per sample while importance/auxiliary keep the global mix even.

    A sigmoid gate (simplex=False) is not a distribution: auxiliary/importance
    then read the weights normalised to one, and entropy becomes the mean
    per-expert BINARY entropy, which is what "commit" means for an on/off gate.
    """
    out = {}
    n_experts = gate_probs.shape[1]
    norm = gate_probs if simplex else \
        gate_probs / (gate_probs.sum(dim=1, keepdim=True) + eps)
    for name in names:
        if name == "auxiliary":
            top = gate_probs.argmax(dim=1)
            f = torch.bincount(top, minlength=n_experts).float() / gate_probs.shape[0]
            out[name] = n_experts * torch.sum(f * norm.mean(dim=0))
        elif name == "importance":
            out[name] = torch.sum((norm.mean(dim=0) - 1.0 / n_experts) ** 2)
        elif name == "entropy":
            if simplex:
                out[name] = -(norm * torch.log(norm + eps)).sum(dim=1).mean()
            else:
                p = gate_probs.clamp(eps, 1 - eps)
                out[name] = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
        else:
            raise ValueError(f"unknown gate loss {name!r}, pick from {GATE_LOSSES}")
    return out


def build_task_loss(cfg: dict, y_train=None):
    """cfg: {name: bce|wbce|focal|triplet, ...}.  Returns fn(logit, target, rep).

    bce   plain, or with an explicit `pos_weight`
    wbce  class-weighted BCE; `pos_weight` defaults to n_neg / n_pos of the
          TRAINING labels, so it needs y_train.  Use it instead of - not on top
          of - `train.balanced_sampler`, which already rebalances the batches.
    """
    name = cfg.get("name", "bce")
    if name in ("bce", "wbce"):
        pw = cfg.get("pos_weight")
        if pw is None and name == "wbce":
            if y_train is None:
                raise ValueError("wbce needs the training labels (or an explicit pos_weight)")
            lab = np.asarray(y_train.detach().cpu() if hasattr(y_train, "detach")
                             else y_train).ravel()
            pw = float((lab == 0).sum()) / max(float((lab == 1).sum()), 1.0)
            print(f"  [wbce] pos_weight = {pw:.2f} "
                  f"({int((lab == 0).sum())} neg / {int((lab == 1).sum())} pos)")
        crit = nn.BCEWithLogitsLoss(
            pos_weight=None if pw is None else torch.tensor([float(pw)]))
        return lambda logit, y, rep: crit.to(logit.device)(logit.view(-1), y.view(-1))
    if name == "focal":
        crit = BinaryFocalLoss(alpha=cfg.get("alpha", 0.3), gamma=cfg.get("gamma", 2.0))
        return lambda logit, y, rep: crit(logit.view(-1), y.view(-1))
    if name == "triplet":
        margin = cfg.get("margin", 1.0)
        # triplet shapes the representation, so the classifier still needs a BCE
        # term to stay calibrated; `bce_weight` keeps it
        w = float(cfg.get("bce_weight", 1.0))
        bce = nn.BCEWithLogitsLoss()
        return lambda logit, y, rep: (triplet_loss(rep, y, margin)
                                      + w * bce(logit.view(-1), y.view(-1)))
    raise ValueError(f"unknown task loss {name!r}, pick from {TASK_LOSSES}")


def build_representation_loss(cfg: dict):
    """cfg: {name: ntxent|barlow_twins|vicreg, weight: w, ...}.

    All three act on repeated measurements of the same subject (`pair_id`), on
    the gate-weighted joint representation.  Returns fn(rep, pair_id) or None.
    """
    name = cfg.get("name")
    if not name or float(cfg.get("weight", 0.0)) <= 0:
        return None
    if name == "ntxent":
        crit = NtXentLoss(temperature=cfg.get("temperature", 0.05))
        return crit
    if name == "barlow_twins":
        lambd = cfg.get("lambd", 5e-3)
        return lambda rep, pid: barlow_twins_loss(rep, pid, lambd=lambd)
    if name == "vicreg":
        kw = {k: cfg[k] for k in ("sim_coef", "std_coef", "cov_coef") if k in cfg}
        return lambda rep, pid: vicreg_loss(rep, pid, **kw)
    raise ValueError(f"unknown representation loss {name!r}, pick from {REPRESENTATION_LOSSES}")


# ----------------------------------------------------------------- self-check
def _selftest():
    set_seed(0)
    dims = {"OMICS_01": 32, "OMICS_02": 16, "OMICS_03": 12}
    cfgs = {"OMICS_01": {"arch": "mlp", "hidden_dims": [16, 8]},
            "OMICS_02": {"arch": "cnn1d", "token_dim": 2, "channels": [8, 8], "pool": 2},
            "OMICS_03": {"arch": "transformer", "token_dim": 2, "d_model": 8, "n_heads": 2}}
    x = {k: torch.randn(8, d) for k, d in dims.items()}
    y = torch.tensor([0.0, 1.0] * 4)
    pair = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    # every gate structure runs, and only the sigmoid one leaves the simplex
    for kind in GATE_TYPES + tuple(GATE_ALIASES):
        m = build_moe(dims, cfgs, {"gate": {"type": kind}})
        g = m.forward_with_rep(x)[1]
        assert g.shape == (8, 3) and (g >= 0).all(), kind
        on_simplex = bool(torch.allclose(g.sum(1), torch.ones(8), atol=1e-5))
        assert on_simplex == m.gate.simplex == (kind != "sigmoid"), kind
        for name in GATE_LOSSES:
            v = gate_losses(g, [name], simplex=m.gate.simplex)[name]
            assert torch.isfinite(v), f"{kind}/{name}"
    assert build_moe(dims, cfgs, {"gate": {"type": "mome"}}).gate.kind == "attention"

    moe = build_moe(dims, cfgs, {"gate": {"type": "attention"}})
    logit, gp, rep = moe.forward_with_rep(x)
    assert logit.shape == (8, 1)
    assert rep.shape[1] == sum(moe.expert_output_dims)
    for name in REPRESENTATION_LOSSES:
        fn = build_representation_loss({"name": name, "weight": 1.0})
        assert torch.isfinite(fn(rep, pair)), name
        # a batch with no replicate pair must be 0, not NaN
        assert float(fn(rep, torch.arange(8))) == 0.0, f"{name} without pairs"
    for name in TASK_LOSSES:
        fn = build_task_loss({"name": name}, y_train=y)
        assert torch.isfinite(fn(logit, y, rep)), name

    # wbce: the positives weigh n_neg/n_pos, so an all-negative prediction costs
    # more than under a plain BCE whenever the positives are the minority
    y_imb = torch.tensor([0.0] * 7 + [1.0])
    lg = torch.full((8, 1), -3.0)
    assert float(build_task_loss({"name": "wbce"}, y_train=y_imb)(lg, y_imb, rep)) > \
        float(build_task_loss({"name": "bce"}, y_train=y_imb)(lg, y_imb, rep))
    assert build_task_loss({"name": "wbce", "pos_weight": 2.0})  # explicit, no labels

    # partial unfreeze: 0 freezes everything, n unfreezes exactly the last n modules
    e = MLPExpert(32, [16, 8])
    assert set_trainable(e, 0) == 0
    assert set_trainable(e, -1) == sum(p.numel() for p in e.parameters())
    last = param_modules(e)[-1][1]
    assert set_trainable(e, 1) == sum(p.numel() for p in last.parameters(recurse=False))

    # gradients reach the gate even with every expert frozen
    for i in range(len(moe.experts)):
        set_trainable(moe.experts[i], 0)
    moe.zero_grad()
    build_task_loss({"name": "bce"})(*moe.forward_with_rep(x)[:1], y, rep).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in moe.gate.parameters())
    print("moe_core selftest OK")


if __name__ == "__main__":
    _selftest()
