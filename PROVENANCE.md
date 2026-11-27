# PROVENANCE — how this repository relates to monogs-ours

- **Source**: `/data/monogs-ours` @ `3d9cfb6e5` (branch `feat/adaptive-itr`), the
  research repository. monogs-ours is not split or modified by this export; it
  remains the authoritative copy of the experiments, the run data, and the
  internal research records.
- **Manuscript**: *Lifecycle State Consistency for Cross-Session Map Takeover in
  Online 3DGS SLAM*, The Visual Computer (under review).
- **Built by**: a whitelist copy, not a `git clone`. This repository has its own
  history and shares no commits with monogs-ours.

## What was copied

| Path | Files | Note |
|---|---|---|
| `slam.py` | 1 | entry point, unmodified |
| `gaussian_splatting/` | 9 | includes the contract: `scene/gaussian_model.py` |
| `utils/` | 60 | includes the default-off switch and the repair counters: `slam_backend.py` |
| `gui/` | 4 | imported by `slam.py`; required to start |
| `configs/rgbd/` | 62 | full inheritance chain (`tum/base_config.yaml` is the root) |
| `configs/experiments/combined_prune/method.yaml` | 1 | the shared method configuration |
| `configs/experiments/paper3_probe/contract_*.yaml` | 7 | the takeover arms |
| `tests/mapload/` | 2 | the contract regression suite |
| `environment.yaml` | 1 | Python 3.10 / CUDA 11.8 |

## What was deliberately **not** copied

These exist in monogs-ours and are withheld while the manuscript is under
review. They are research records, not code:

| Path | Why withheld |
|---|---|
| `paper3/DECISIONS.md`, `paper3/STATUS.md`, `paper3/NEXT_SESSION.md` | decision ledger and session state; contains rejected directions and internal reasoning |
| `paper3/results/evidence/**` | pre-registration documents, judgements, device-floor probes, sensitivity analyses |
| `papers/paper3/**` | manuscript source, cover letter, internal review reports |
| `results/**` (23 GB) | per-seed run directories, point clouds, per-frame traces |
| `weights/**` (8.4 GB) | model weights |
| `resources/**` | literature PDFs and reading notes |
| `DECISIONS.md`, `DISCIPLINE.md`, `LOG.md`, `NOTES.md`, `PITFALLS.md` | project-level research records |

The per-frame error traces that the paper's transient analysis consumes are
delivered separately as **Online Resource 1** of the manuscript, not here.

## Comment scrubbing (integrity note)

Source comments in monogs-ours reference internal design-note identifiers and
pre-registration file names. Those were replaced with neutral text on export.

The scrubbing is **comment-only and provably behaviour-preserving**: for all 25
touched Python files the abstract syntax tree with docstrings removed is
identical to the source, and for all YAML files the non-comment lines are
identical. Verification was run in this tree before the export commit.

Four **runtime strings** were left unmodified on purpose, because changing them
would make these files differ from the source in behaviour:

- `utils/adaptive_itr.py` — three `raise` messages
- `utils/slam_backend.py` — one docstring reference used as a contract citation

They name a pre-registration file, not a decision number, and carry no result
or judgement.

## Relationship to the sibling repositories

The four code repositories are independent, each with its own history:

| Repository | Research content | Status |
|---|---|---|
| `RoGS-SLAM` | 1 — dynamic-scene 3DGS SLAM | MMM 2027, accepted |
| `RoGS-Map` | 2 — anatomy, compression, offline repair of maps | CVM 2027, under review |
| `RoGS-Densify` | 3 (earlier line) — temporal-aware densification | archived; the arm-level effect fell below the measurement floor |
| **`RoGS-Takeover`** (this repo) | **3 (current) — lifecycle contract and cross-session takeover** | The Visual Computer, under review |

**This repository does not apply the module subtraction used by RoGS-Map and
RoGS-Densify.** Those repositories remove the SWBA / LoopClosure / DBA family
and the paper-3 densification modules, and rewire `slam.py` and
`slam_backend.py` accordingly, because their papers make no use of them and
their narratives exclude them.

Here the full pipeline is retained instead. The reason is that `slam.py`
imports `utils/loop_closure.py` directly, so removing it requires the same
rewiring work those repositories did; and the module subtraction changes the
entry point, which would make this repository harder to check against the
source. Under every takeover arm configuration the subtracted-elsewhere
branches are inactive, so carrying them adds no behavioural difference to the
reported runs — it only adds code.

A reader comparing this repository with `RoGS-Map` will therefore find modules
here that are absent there. That is intended and is not a divergence in the
reported configuration.

## Verification performed before the export commit

| Check | Result |
|---|---|
| `pytest tests/mapload/` in this tree | 13 passed (no dataset needed) |
| All four contract configs load via this repository's own `load_config` | OK; `MapLoad.enabled` = True / False / True / None for BX / BE / BCneg / prior-build |
| Python syntax across the tree | 0 errors |
| AST equality with source (docstrings stripped) | 25/25 files identical |
| Datasets, weights, run outputs committed | 0 files |
