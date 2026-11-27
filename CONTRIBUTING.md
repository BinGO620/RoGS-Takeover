# Contributing

This repository is the reference implementation for a manuscript under review.
It is maintained as a fixed snapshot of the code used for the reported runs, so
the scope of acceptable changes is narrower than in a general-purpose project.

## What is welcome

- **Reproducibility reports.** If a step in `REPRODUCING.md` does not work on
  your setup, open an issue with your OS, GPU, CUDA version, the command you
  ran, and the error. This is the most useful contribution at this stage.
- **Clarification of the contract.** If a mutation path's behaviour is unclear
  from the code, an issue pointing at the specific path is welcome. The contract
  is described in §3 of the manuscript and implemented in
  `gaussian_splatting/scene/gaussian_model.py`.

## What will not be merged while the manuscript is under review

- Changes that alter the reported results or the configuration of the takeover
  arms. The results are tied to specific configurations; changing them
  invalidates the correspondence between this repository and the paper.
- Refactors that change the entry point or the module layout. The repository is
  kept checkable against the source it was exported from; see `PROVENANCE.md`.
- New features or mechanisms. The contract is scoped deliberately, and the paper
  states that scope.

## Bugs

If you find a genuine defect — a crash, a broken test, an inconsistency between
the code and `REPRODUCING.md` — open an issue describing what you expected and
what happened. Please include a minimal reproduction if you can.

## Licence

By contributing you agree that your contribution is distributed under the same
licence as the rest of this repository (Inria / MPII non-commercial research
licence; see `LICENSE`). Do not contribute code that is under a different or
incompatible licence.

## Contact

Open an issue for anything that can be discussed publicly. For the manuscript
itself, use the corresponding-author addresses listed in `CITATION.cff`.
