# Reproducing the takeover experiments (RoGS-Takeover)

Step-by-step instructions to reproduce the results of *Lifecycle State
Consistency for Cross-Session Map Takeover in Online 3DGS SLAM* (The Visual
Computer, under review).

The paper makes three claims, and each has a reproduction path below:

1. **The contract holds** — every mutation path keeps the parameter tensors,
   the ledger tensors, and the optimizer statistics at the same length, and the
   runtime guard fires on any mismatch. Reproduced by the regression suite
   (§2), which needs no dataset.
2. **Takeover completes** — a session-1 prior map loads into a session-2 run and
   full mapping continues with zero guard triggers and zero counted repairs.
   Reproduced by the three-arm batch (§3).
3. **The control is discriminable at probe scale** — a map displaced by 0.6 m
   separates from self-building in an 80-frame probe. Reproduced by §4.

---

## 0. Requirements

| Item | Requirement |
|---|---|
| OS | Linux x86_64 |
| GPU | NVIDIA, ≥ 12 GB VRAM for the full-length arms |
| Toolchain | CUDA 11.8 (`nvcc`), `gcc`/`g++`, `git` |
| Conda | Miniconda or Anaconda |
| Disk | ~10 GB for the Bonn RGB-D Dynamic sequences |
| Python | 3.10, provided by `environment.yaml` |

```bash
conda env create -f environment.yaml
conda activate monogs-ours
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

The two rasterisation submodules are not vendored here. Clone them into
`submodules/` from the upstream 3D Gaussian Splatting release before installing:

```bash
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization submodules/diff-gaussian-rasterization
git clone https://github.com/camenduru/simple-knn submodules/simple-knn
```

Both are covered by the Inria / MPII non-commercial research licence reproduced
in `LICENSE`.

---

## 1. Dataset

Download the **Bonn RGB-D Dynamic** dataset from the University of Bonn:

<https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/>

The reported runs use three sequences:

| Sequence | Frames | Role |
|---|---|---|
| `rgbd_bonn_crowd2` | 895 | session 1 — builds the prior map |
| `rgbd_bonn_crowd3` | 854 | session 2 — first cross-session pair |
| `rgbd_bonn_crowd` | 928 | session 2 — second pair (reuses the same prior) |

Point the `datasets/` symlink at the downloaded tree, or edit
`Dataset.dataset_path` in `configs/rgbd/bonn/base_config.yaml`.

---

## 2. The contract tests (no GPU data needed)

This is the fastest check that the contract is intact. It runs in a few
seconds on CPU:

```bash
pytest tests/mapload/ -v
```

Expected: **13 passed**.

What is being verified:

| Test file | Claim |
|---|---|
| `test_ledger_contract.py` | Every mutation path (`append`, `prune`, `load`) preserves `len(P) == len(L) == len(O)`; the guard raises on a corrupted state; a load initialises primitives as first-class citizens (anchored keyframe, zero observations, `UNTRACKED` lineage) |
| `test_mapload_default_off.py` | With `MapLoad` disabled the system is unchanged, and the contract check fails fast on a malformed prior map |

---

## 3. Takeover on a cross-session pair

### 3.1 Build the session-1 prior map (once)

```bash
python slam.py --config configs/experiments/paper3_probe/contract_full_crowd2_make.yaml \
               --eval --seed 0
```

The run writes `point_cloud/final/point_cloud.ply` under its `save_dir`. Copy it
to the path the session-2 configs expect:

```bash
cp <save_dir>/point_cloud/final/point_cloud.ply \
   results/paper3_probe/contract_full_crowd2_prior.ply
```

The prior map carries its own reconstruction error. On this sequence it is
6.63 cm RMSE against session-1 ground truth, and that figure is the one the
paper quotes for the transient analysis.

### 3.2 Run the three arms

Every arm runs the same configuration and differs only in the map source:

| Arm | `MapLoad` | What it does |
|---|---|---|
| **BE** | disabled | self-builds session 2, no prior map |
| **BX** | enabled, `translate_m: 0.0` | loads the correctly placed prior map |
| **BCneg** | enabled, displaced copy | loads a prior map displaced by 0.6 m |

```bash
# BE — self-building baseline
python slam.py --config configs/experiments/paper3_probe/contract_full_crowd3_BE.yaml \
               --eval --seed 0

# BX — takeover of the correctly placed prior
python slam.py --config configs/experiments/paper3_probe/contract_full_crowd3_BX.yaml \
               --eval --seed 0

# BCneg — displaced control (see §4 for the displacement itself)
python slam.py --config configs/experiments/paper3_probe/contract_full_crowd3_BCneg.yaml \
               --eval --seed 0
```

Repeat with `--seed 1` and `--seed 2`. The paper reports three seeds per pair.

The second cross-session pair (`crowd2` → `crowd`) uses the same prior map and
the `contract_pair2_crowd_{BE,BX,BCneg}.yaml` configs.

### 3.3 Reproducibility locks

The three execution locks are already set in the shipped configs and are not
optional. They are what makes the per-seed comparison meaningful:

```yaml
Training:   { single_thread: true }
Dataset:    { single_thread: true }
Rasterizer: { deterministic_backward: true }
```

With these locks two runs at the same configuration and seed produce
bitwise-identical trajectories. Without them this pipeline is **not**
deterministic: repeated runs at the same configuration and seed have been
observed to differ by tens of percent in terminal map size. Any comparison of
arms is only interpretable under the locks.

### 3.4 What to check in the output

For each BX run, the map-load summary reports:

- `guard_triggers: 0` — no cardinality mismatch reached the guard
- `repair_count: 0` — no counted repair site fired

A run with a non-zero repair count is not a clean takeover, even if it
completed.

---

## 4. The displaced control

`BCneg` loads a copy of the same prior map translated in x by 0.6 m:

```python
import numpy as np
from plyfile import PlyData

p = PlyData.read("results/paper3_probe/contract_full_crowd2_prior.ply")
v = p["vertex"]
v["x"] = (np.asarray(v["x"]) + 0.30).astype(v["x"].dtype)   # file-level shift
PlyData([v], text=False).write(
    "results/paper3_probe/contract_full_crowd2_prior_shift030.ply")
```

The `+0.30` here combines with the load-time `translate_m: 0.30` in the BCneg
config to give a **total displacement of 0.60 m**. Both halves are needed; the
file shift alone is half the control.

The control separates from self-building in the 80-frame probe and at full
length on the first pair. At full length it does **not** separate the placed
prior from the displaced one, because the same convergence that decays the
inherited displacement also decays the imposed shift. That is why the paper
registers the control as a probe-scale instrument and claims capability from
the guard and takeover results instead.

---

## 5. Runtimes

Measured on two RTX 3090 cards, one run per card, with the locks on:

| Run | Frames | Wall time |
|---|---|---|
| Session-1 prior map | 895 | ~110 min |
| One full-length session-2 arm | 854 or 928 | ~60 min |
| 80-frame probe | 80 | ~4 min |

A full first-pair batch (prior map + 3 arms × 3 seeds) takes roughly 7 hours on
two cards.

---

## 6. What is not reproduced here

- **The rendering readings.** The run-time evaluation reported no whole-image
  values on this sequence because its sampling interval coincided with the
  keyframe grid, so every sampled frame was skipped. The offline re-render that
  produced the single-seed values in the paper is a separate evaluation pass,
  not part of the takeover pipeline. The paper makes no rendering-quality claim.
- **The transient sensitivity sweep.** The 100 configurations behind the
  3.4×–9.6× range are computed from the per-frame traces, which are shipped as
  Online Resource 1 of the manuscript.
- **The external baseline (SplaTAM).** It is a separate system with its own
  environment; the paper reports it under its own error convention and does not
  place it in the same table as our arms.
