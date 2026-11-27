# RoGS-Takeover

Reference implementation of the **lifecycle state consistency contract** for
cross-session map takeover in online 3D Gaussian Splatting SLAM.

Companion code for the manuscript:

> **Lifecycle State Consistency for Cross-Session Map Takeover in Online 3DGS SLAM**
> Bin Chen, Jie Su, Jing Zhang — *The Visual Computer* (under review)

---

## What this is

An online 3DGS SLAM system keeps, for every map primitive, a lifecycle ledger:
the owning keyframe, an observation count, a lineage label, and a static lane.
That ledger is only maintained along the mutation paths of a single session.
Loading a map saved by an earlier session into a running one breaks the
cardinality relation between the parameter tensors and the ledger, and the
system fails.

This repository contains the contract that makes the relation explicit and the
repair that lets loaded primitives enter the same lifecycle as natively grown
ones. The contract requires the parameter tensors `P`, the ledger tensors `L`,
and the optimizer statistics `O` to have the same length at all times, and
enforces it on every mutation path:

| Path | Rule |
|---|---|
| `append` | admitted primitives enter the ledger anchored, with zero observations and a lineage label by origin |
| `shrink` | ledger entries are removed with the same mask applied to `P` |
| `load` | a loaded map is initialised as if its primitives had just been born, with the optimizer statistics rebuilt to match |

Three layers enforce it: a fail-loud runtime guard that validates cardinality
after every external mutation, a regression suite that replays the takeover
mutation sequence against a native and a loaded map, and counted repair sites
so that a clean run reports zero repairs.

## Where the contract lives

| Component | File | Anchor |
|---|---|---|
| Load path (contract-compliant `load_ply`) | `gaussian_splatting/scene/gaussian_model.py` | `load_ply(..., kf_anchor=...)` |
| Cardinality guard (15 tensors) | `gaussian_splatting/scene/gaussian_model.py` | `tensor_names` |
| Ledger fields and `UNTRACKED` lineage | `gaussian_splatting/scene/gaussian_model.py` | `UNTRACKED` |
| Default-off switch and counted repairs | `utils/slam_backend.py` | `mapload_enabled`, `mapload_cfg` |
| Deferred candidate admission | `utils/deferred_commit.py` | — |
| Contract regression tests | `tests/mapload/` | `test_ledger_contract.py`, `test_mapload_default_off.py` |
| Takeover experiment configs | `configs/experiments/paper3_probe/contract_*.yaml` | — |

## Installation

Tested with Python 3.10 and CUDA 11.8.

```bash
conda env create -f environment.yaml
conda activate monogs-ours
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

The two rasterisation submodules are **not vendored here**; install them from
their upstream repositories:

- `diff-gaussian-rasterization` — the rasteriser from the original 3D Gaussian
  Splatting release (Inria / MPII)
- `simple-knn` — the CUDA k-nearest-neighbour helper from the same release

Both are distributed under the Inria / MPII non-commercial research licence;
see `LICENSE`.

## Reproducing the contract tests

These need no GPU data and run in under a minute:

```bash
pytest tests/mapload/ -v
```

`test_ledger_contract.py` checks that every mutation path preserves
`len(P) == len(L) == len(O)` and that the guard fires on a corrupted state.
`test_mapload_default_off.py` checks that with `MapLoad` disabled the system is
unchanged.

## Running a takeover experiment

The reported runs use the Bonn RGB-D Dynamic dataset (`crowd2` for the session-1
prior map, `crowd3` and `crowd` for the session-2 recordings). Download it from
the University of Bonn and point `datasets/` at it.

A session-1 prior map is built once:

```bash
python slam.py --config configs/experiments/paper3_probe/contract_full_crowd2_make.yaml --eval --seed 0
```

The resulting `point_cloud/final/point_cloud.ply` becomes the prior map. Point
`MapLoad.ply_path` in the session-2 config at it, then run the three arms:

```bash
# BE  — self-building (no prior map)
python slam.py --config configs/experiments/paper3_probe/contract_full_crowd3_BE.yaml  --eval --seed 0

# BX  — takeover of the correctly placed prior map
python slam.py --config configs/experiments/paper3_probe/contract_full_crowd3_BX.yaml  --eval --seed 0

# BCneg — takeover of a control map displaced by 0.6 m
python slam.py --config configs/experiments/paper3_probe/contract_full_crowd3_BCneg.yaml --eval --seed 0
```

Reproducibility in the reported runs rests on three execution locks, applied
identically to every arm: the tracking frontend and the mapping backend each run
single-threaded, and the rasteriser backward pass is deterministic. They are set
in the shipped configs (`Training.single_thread`, `Dataset.single_thread`,
`Rasterizer.deterministic_backward`). Under these locks two runs at the same
configuration and seed produce bitwise-identical trajectories; without them this
pipeline is not deterministic.

## Scope and limitations

- The contract is scoped to one stack: an append-style ledger with a single
  prune path and a fail-loud guard. Stacks whose maps carry no per-primitive
  state, or whose state is rebuilt from geometry on demand, face a different
  problem.
- Takeover carries a measured accuracy cost. Against self-building, ATE rises by
  1.30–1.39 cm and 0.81–1.60 cm on the two pairs. A per-frame analysis localises
  this to a converging transient consistent with the prior map's own
  reconstruction error. How the cost scales with prior-map quality is a testable
  prediction, not a measured relation: both pairs share one session-1 map.
- No rendering-quality claim is made. The rendering readings in the paper are
  single-seed, three-frame descriptive values.

## Data and result artefacts

This repository ships **code and configurations only**. The raw experiment
outputs (per-seed run directories, point clouds, per-frame APE traces) are not
included. The per-frame traces underlying the transient analysis and the
sensitivity sweep are provided as Online Resource 1 of the manuscript.

## Licence

Non-commercial research use, inherited from the Inria / MPII Gaussian Splatting
licence. See `LICENSE`.

## Citation

```bibtex
@article{chen2026rogstakeover,
  title   = {Lifecycle State Consistency for Cross-Session Map Takeover in Online {3DGS} {SLAM}},
  author  = {Chen, Bin and Su, Jie and Zhang, Jing},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgements

Built on [MonoGS](https://github.com/muskie82/MonoGS) and the original
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
rasteriser.
