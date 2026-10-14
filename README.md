# MoE Tricks

Trials to use Mixture of Experts on multi-omic features.

**Objective: not only SOTA, but also Explainable.**

Omics = Dataset with Modality (in Biology)

One config file decides the whole model: how many omics views there are, what
each expert looks like, which gate structure routes them, and which losses are
attached where. Nothing in the code is tied to a particular assay - the views
are called `OMICS_01`, `OMICS_02`, ... and you add a third by adding two YAML
blocks.

---

## What you can combine

![architecture](docs/architecture.png)


| axis | values | config key |
|---|---|---|
| views | any number, any width | `data.omics.*` |
| expert architecture | `mlp` · `cnn1d` · `transformer` | `model.experts.<view>.arch` |
| expert init | scratch · pretrained `.pth` | `model.experts.<view>.pretrained` |
| how much of a pretrained expert trains | `0` frozen · `n` last n modules · `-1` all | `model.experts.<view>.unfreeze_last_n` |
| gate structure | `sigmoid` · `softmax` · `gumbel` · `attention` | `model.gate.type` |
| task loss | `bce` · `wbce` · `focal` · `triplet` | `loss.task.name` |
| gate losses | `auxiliary` · `importance` · `entropy`, weighted | `loss.gate.*` |
| representation loss | `ntxent` · `barlow_twins` · `vicreg`, weighted | `loss.representation` |
| per-expert learning rate | one rate per view, or `expert_lr_scale: 0` to freeze all | `model.experts.<view>.lr` |

**Notation used below.** Sample $n$ of a batch of size $B$, view $i$ of $N$
views. $x_{ni}$ is the raw view, $h_{ni}$ the expert's representation,
$g_{ni}$ the gate weight, $p_n$ the predicted probability, $y_n$ the label.
The joint representation and the logit are

$$
z_n=\big[\,g_{n1}h_{n1}\;\Vert\;\cdots\;\Vert\;g_{nN}h_{nN}\,\big],\qquad
\text{logit}_n=w^\top z_n+b .
$$

## Experiment

![experiment](docs/experiment.png)

Any dotted path in that table can be handed to optuna instead of being fixed:

```yaml
optimize:
  enabled: true
  n_trials: 30
  space:
    model.gate.type: {type: categorical, choices: [sigmoid, softmax, gumbel, attention]}
    model.experts.OMICS_01.arch: {type: categorical, choices: [transformer, cnn1d]}
    model.experts.OMICS_01.unfreeze_last_n: {type: categorical, choices: [0, 1, 2, 3]}
    loss.representation.weight: {type: float, low: 0.0, high: 0.5, step: 0.05}
```

---

## Run

```bash
python run_moe.py configs/scratch.yaml           # experts + gate trained at once
python run_moe.py configs/pretrained_gate.yaml   # pretrained experts, gate only
python run_moe.py cfg.yaml --set train.epochs=50 --set model.gate.type=attention
python run_moe.py --selftest                     # whole pipeline on synthetic data
```

Both modes are the *same* code path - what separates them is whether
`model.experts.<view>.pretrained` is set.

| file | contents |
|---|---|
| `moe_core.py` | experts, the four gates, every loss, checkpoint loading, freeze/unfreeze |
| `moe_data.py` | reads N omics tables + meta into aligned views |
| `moe_train.py` | train loop, cross-validation, metrics, checkpoint bundles |
| `run_moe.py` | CLI: config -> CV -> (optuna) -> refit -> save |
| `configs/*.yaml` | the two starting points, every knob documented inline |

Each module runs its own self-check: `python moe_core.py`, `python moe_data.py`,
`python moe_train.py`, `python run_moe.py --selftest`.

### Data contract

One meta row = one observation. The same subject measured twice gives two rows
sharing `subject_col`; they stay together in the same split, fold and batch, and
the representation losses pair them. A view that exists only per subject is
copied onto both rows. Scores are averaged back to one value per subject.

---

## Consideration for loss as statistical approach

Total objective, every weight set in the YAML:

$$
\mathcal{L}=\mathcal{L}_{\text{task}}
+\lambda_{\text{aux}}\mathcal{L}_{\text{aux}}
+\lambda_{\text{imp}}\mathcal{L}_{\text{imp}}
+\lambda_{\text{ent}}\mathcal{L}_{\text{ent}}
+\lambda_{\text{rep}}\mathcal{L}_{\text{rep}}
$$

### [for gate]

All three gate terms are **minimised**. They are deliberately small: at weight
$\approx 1$ they are ten times the task loss and freeze every sample at
$1/N$ - a gate that looks alive and says nothing.

<img src="docs/fig_gate_entropy.png" width="900">

Write the batch-mean weight of view $i$ and its hard share as

$$
P_i=\frac{1}{B}\sum_n g_{ni},\qquad
f_i=\frac{1}{B}\sum_n \mathbf{1}\!\left[\arg\max_j g_{nj}=i\right].
$$

**auxiliary** (load balancing, Switch-Transformer):

$$
\mathcal{L}_{\text{aux}}=N\sum_{i} f_i P_i \;\;\ge 1,\quad =1 \text{ when balanced.}
$$

It multiplies the hard count by the soft probability, so it keeps pushing even
when the average already looks even.

**importance** (anti-collapse): squared distance of the mean weight from uniform,

$$
\mathcal{L}_{\text{imp}}=\sum_i\Big(P_i-\tfrac{1}{N}\Big)^2 .
$$

Smoother and weaker than `auxiliary`; the one to start with when $N$ is small.

**entropy** (per-sample decisiveness):

$$
\mathcal{L}_{\text{ent}}=-\frac{1}{B}\sum_n\sum_i g_{ni}\log g_{ni}.
$$

Minimising it pushes each sample towards a one-hot route; `importance` keeps the
cohort-level mix even. Use them **together**: entropy alone sends everyone to one
expert, importance alone pins everyone at $1/N$. An earlier version *maximised*
entropy - with importance on top, every sample sat at 50:50 (left panel above).

For the `sigmoid` gate the weights are not a distribution, so entropy becomes the
mean **binary** entropy per view and the balance terms use $g_{ni}/\sum_j g_{nj}$.

### [for feature]

**bce / wbce** - binary cross-entropy, optionally weighted:

$$
\mathcal{L}_{\text{bce}}=-\frac{1}{B}\sum_n \big[\,c\,y_n\log p_n+(1-y_n)\log(1-p_n)\big],\qquad
c=\frac{n_{\text{neg}}}{n_{\text{pos}}}\ \text{for \texttt{wbce}} .
$$

**focal** - down-weights the easy cases by $(1-p_t)^\gamma$ with
$p_t=p_n$ if $y_n=1$ else $1-p_n$:

$$
\mathcal{L}_{\text{focal}}=-\frac{1}{B}\sum_n \alpha_t(1-p_t)^{\gamma}\log p_t .
$$

**triplet** - batch-hard, using the label as the identity:

$$
\mathcal{L}_{\text{tri}}=\frac{1}{B}\sum_n\Big[\max_{y_j=y_n}\lVert z_n-z_j\rVert-\min_{y_j\neq y_n}\lVert z_n-z_j\rVert+m\Big]_+ ,\qquad m=\text{margin}.
$$

It shapes distances, not the decision value, so a BCE term rides along
(`bce_weight`) to keep the probability calibrated.

Imbalance: use **either** `balanced_sampler` (resamples subjects, replicates
stay together) **or** `wbce`. Both at once corrects twice.

### [for representation]

Same subject measured twice: anything that moves between the two measurements is
the assay, not the patient. All three act on $z_n$ and return exactly $0$ when a
batch holds no pair. Let $z^A,z^B$ be the two measurements of one subject.

<img src="docs/fig_replicate.png" width="900">

**NT-Xent** - the replicate is the positive, the rest of the batch negatives,
with cosine similarity $s$ and temperature $\tau$:

$$
\mathcal{L}_{\text{ntx}}=-\log\frac{\exp(s(z^A,z^B)/\tau)}{\sum_{m\neq A}\exp(s(z^A,z_m)/\tau)} .
$$

Needs negatives, so small batches hurt; it also pushes apart two patients who are
genuinely alike.

**Barlow Twins** - cross-correlation matrix $C$ of the two standardised
measurements driven to the identity:

$$
\mathcal{L}_{\text{bt}}=\sum_i (1-C_{ii})^2+\lambda\sum_{i\neq j}C_{ij}^2 .
$$

No negatives, so batch size matters far less - the practical choice at $n\sim100$.

**VICReg** - invariance + variance floor + decorrelation:

$$
\mathcal{L}_{\text{vic}}=\alpha\,\lVert z^A-z^B\rVert^2
+\beta\sum_{k}\big[1-\sqrt{\operatorname{Var}(z^{\cdot}_k)+\epsilon}\,\big]_+
+\gamma\sum_{k\neq l}\operatorname{Cov}(z^{\cdot})_{kl}^2 .
$$

The middle term is a hinge that *pays the model to stay spread out*, so VICReg
resists the collapse that makes two measurements agree - visible as its wide
scatter and low polarization in the figure.

Rule of thumb: Barlow Twins or VICReg when replicates are the only structure you
have, NT-Xent when the batch is big enough for negatives to mean something.

---

## Consideration for gate structure

All four read the standardised, LayerNormed views and emit one weight per view.
With $u_n=W[x_{n1}\Vert\cdots\Vert x_{nN}]+b$ and a learned temperature
$\tau=\operatorname{softplus}(\theta)$:

<img src="docs/fig_gate_structure.png" width="900">

| gate | weight | reads as |
|---|---|---|
| `sigmoid` | $g_{ni}=\sigma(u_{ni}/\tau)$ | independent switches, $\sum_i g_{ni}\neq 1$ - a soft ensemble |
| `softmax` | $g_{n}=\operatorname{softmax}(u_n/\tau)$ | a share out of 100% - the usual MoE, and the one that quotes cleanly |
| `gumbel` | $g_n=\operatorname{softmax}((u_n+\varepsilon)/\tau)$, $\varepsilon\sim$ Gumbel (train only) | a near-discrete pick; the noise also regularises a small cohort |
| `attention` | $c_{ni}=\tanh(W_ix_{ni})\odot\sigma(V_ix_{ni})$, then $g_n=\operatorname{softmax}(W_c[c_{n1}\Vert\cdots])$ | weighs *parts* of a view first - survives signal buried in thousands of features, at the cost of parameters |

Choose by what you want to claim: `softmax` if the weight itself is the result,
`gumbel` if you believe one view is right per patient, `sigmoid` if the views are
complementary rather than competing, `attention` if a view is mostly noise.

### gate for input feature vs hidden dimension

**Implemented: input-feature gating**, $g_n=\text{gate}(x_n)$. The routing is a
statement about the *data*, independent of how well each expert is trained, which
is what makes the weight quotable - and the only option for a frozen expert.

**Hidden-dimension gating**, $g_n=\text{gate}(h_n)$, is the alternative: more
informative, but gate and expert now train each other. A weak expert gets
down-weighted, trains less, stays weak. *Not implemented*: feed `feats` to the
gate in `MoE.forward_with_rep` and size `MoEGate` by `expert_output_dims`.

Baseline worth keeping: **no gate** (plain concatenation). If the gate does not
beat it, the gate is decoration.

---

### Consideration for search space difference

<img src="docs/fig_sampler_space.png" width="900">

Four ways to pick the next trial $x^{(t)}$ from the search space $\mathcal{X}$,
given the trials already scored, $\mathcal{D}=\{(x^{(s)},y^{(s)})\}_{s<t}$:

| sampler | what it assumes / samples from | in one line |
|---|---|---|
| `Random` | $x^{(t)}\sim\mathrm{Uniform}(\mathcal{X})$, independent of $\mathcal{D}$ | no model at all - covers the box evenly, never zooms in |
| `TPE` | splits trials at a quantile $y^\ast$ and fits two densities, $\ell(x)=p(x\mid y>y^\ast)$ and $g(x)=p(x\mid y\le y^\ast)$; samples where $\ell(x)/g(x)$ is large | density ratio, not a surface - cheap, and it concentrates fast |
| `GP` | $f\sim\mathcal{GP}(\mu,k)$, so the posterior gives $\mu_t(x)$ and $\sigma_t(x)$; picks $\arg\max_x \mathbb{E}\big[(f(x)-f^\ast)_+\big]$ | one smooth surface with error bars - spends trials on the corners it has not seen |
| `CmaEs` | $x^{(t)}\sim\mathcal{N}\!\left(m_t,\sigma_t^2 C_t\right)$, with $m_t,C_t$ re-estimated from the best trials | an ellipse that walks downhill - tight, and blind to anything outside it |

The density panels show exactly that: `Random` fills the box, `TPE` collapses
onto one peak, `GP` keeps probing the edges, `CmaEs` draws one elongated cloud.
Best objective was reached by `GP` on OMICS1 and by `TPE` on OMICS2 - with 42
trials the ranking is not stable enough to call a winner.

---

## Consideration for expert (+ transfer learning)

Experts are built from scratch or loaded from a `.pth` (TorchScript, pickled
module, or `state_dict`). A loaded expert can stay frozen, open partly, or
fine-tune whole.

### how many layers should be open / freeze

`unfreeze_last_n` counts **parameter-owning sub-modules from the output end**:

| value | effect | when |
|---|---|---|
| `0` | frozen; only gate + classifier learn | the expert saw far more data than this cohort; you want the gate's contribution isolated |
| `1`–`2` | head (and last block) adapt | the representation transfers, the decision boundary does not - the usual cross-site case |
| larger `n` | deeper adaptation | the input distribution itself shifted (assay, depth) |
| `-1` | everything trains | enough data, or the weights are only an initialisation |

What the code enforces so the choice means what you think:

- a frozen expert sits in **eval mode** - its dropout and BatchNorm statistics
  stop moving while "only the gate learns";
- frozen parameters get **no optimiser group** - weight decay cannot shrink them;
- each expert keeps **its own learning rate**; a tail being fine-tuned usually
  wants an order of magnitude less than a gate trained from scratch.

`train.expert_lr_scale: 0` freezes every expert at once - the cheapest "is the
gate doing anything?" ablation.

**Leakage warning.** An expert pretrained on the whole development set has
already seen the held-out fold. Either pretrain outside the cohort, or keep
fold-specific checkpoints. The code cannot detect this.

---

## Consideration for explainability

A gate is worth having only if it is readable. What one run leaves behind:

<img src="docs/fig_gate_cancer.png" width="900">

- `pred.npz` - per-row **gate weights**, out-of-fold (`dev_gate_oof`) and
  external (`ext_gate`), next to predictions and labels. Which view carried
  which patient is a table, not a guess.
- `history.tsv` - every loss component per epoch, so a gate that collapsed can
  be told from a gate that was never pushed.
- `expert_<view>.pth` / `moe_head.pth` - parts saved separately: probe an expert
  alone, or re-use it as the next run's `pretrained:`.
- `trials_metrics.tsv` - mean gate weight per view beside the score, which is how
  you catch a model that scores well *because* the gate stopped routing.
