# Handoff for coding agents

Read before running or changing experiments:

1. `docs/REPRODUCE.md`: new-machine setup and launch commands.
2. `docs/environment_reproduction.md` and `docs/artifact_reproduction.md`: exact environments and non-code inputs.
3. `reproducibility/experiments.json` and `docs/experiment_results.csv`: saved recipes and completed held-out results.
4. `docs/motus_low_gpu_memory.md`: current lower-memory profile.

Dated README sections are historical records. Their PIDs and local output
paths are not current machine state; inspect local status files and processes.

## Experiment contract

- RLInf SAC updates a noise controller; the Motus pi0.5 decoder stays frozen.
- Current noise is `20 x 32`, decoder horizon 32, execution chunk 20, five flow
  steps. Copy the first 12 noise rows into the remaining decoder positions.
  Historical repeated-32D controllers are a different architecture.
- `move_can_pot` and `click_bell` use 400 actions; matched `adjust_bottle` runs
  use 200. Follow the selected saved recipe.
- Current learned reward is `native_success - 0.3 * (1 - progress)`. ICON uses
  the exact fixed training demonstration. Robometer has no context. Object
  matching is a separate experiment.
- Current training: 400 collections, eight episode attempts per collection,
  global batch 128, per-rank microbatch 64, readiness 25, 20 critic / 10 actor
  updates per eligible collection, terminal padding excluded from sampling.
- Final evaluation: 150 held-out scenes, native success, no optimization or
  reward-model inference. Training MA100 and smoke scores are separate metrics.
- `--low-gpu-memory` preserves learning settings and offloads SAC state between
  phases. The queue owns GPU-0 reward servers only while their training needs them.

## Active jobs

- Check `outputs/**/experiment_status.json` and host-visible processes first.
  A sandbox process list can omit host GPU jobs.
- Running queues pin source/config/model/demo hashes and file identities. Do
  not edit their inputs while they run. New docs and reproduction records are safe.
- Follow the user's requested action; do not start/stop long GPU jobs for a
  read-only request or signal unrelated GPU processes.
- New trials require fresh output directories. Do not bypass locks or proof gates.
- Use launcher environment helpers for Vulkan, NCCL and short isolated Ray paths.

## Backup and validation

The `third_party/` gitlinks pin upstream bases. Local changes are backed up in
`vendor_patches/`; dirty submodules are expected after restoration. Apply/check
with `scripts/vendor_source_backup.py`. Refresh/check the snapshot after vendor
edits and before committing; do not advance gitlinks to unpublished commits.

Keep model weights, simulator assets, full runs, caches, virtual environments
and credentials out of Git. The artifact manifest identifies external inputs
and deliberately versioned small context archives. Recollecting a demo changes
the experiment; preserve its exact frame hash for reproduction.

Run relevant tests with the RLInf Python. CPU suite:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src:$PWD/third_party/rlinf" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  rlinf/bin/python -m pytest -q tests
```

GPU tests in `tests/test_dsrl_offload_gpu.py` require
`RUN_DSRL_GPU_TESTS=1` and available GPU headroom. Simulator smokes perform real
updates, then discard their controller/replay before fresh training. Record
what was actually verified and distinguish protocol reproduction from guaranteed scores.
