# Reproduce the experiments on another computer

This is the entry point for a person or Codex setting up a new machine.
Read `AGENTS.md` first. The backup includes the current source, exact dependency
inventories, fixed context pixels, saved recipes and a ledger of 31 completed
150-scene evaluations. Large model weights, simulator assets and learned final
checkpoints are separate artifacts.

## What to preserve

| In Git | Restore separately |
| --- | --- |
| Parent code, six can-pot configs, all experiment YAMLs and tests | Pretrained policy/reward/Qwen weights at recorded revisions |
| Four pinned vendor bases plus `vendor_patches/` | Extracted RoboTwin assets from the pinned dataset |
| Exact compressed fixed demos, including object-bank references | OpenPI tokenizer, isolated NCCL and SAPIEN OIDN libraries |
| Python package snapshots and installed-source overlays | Final learned checkpoint for re-evaluating an existing policy |
| 31 result records, 54 resolved configs, available launch hashes | Replay/optimizers/RNG if a training-resume experiment is needed |

See [artifact provenance and transfer](artifact_reproduction.md) and
[environment reconstruction](environment_reproduction.md). Git includes the
small bootstrap/proof files the launchers previously required from ignored
`outputs/` directories. Restore them with the artifact helper; never fabricate
a passing validation report or replace a context with a newly collected demo.

The reference machine has two 96-GB RTX PRO 6000 Blackwell GPUs, Ubuntu 24.04,
about 503 GiB RAM and a large NVMe drive. The matched learner uses two GPU ranks.
The lower-memory profile was tested with another GPU job present; changing
world size, batch size, precision or update count makes a different recipe.
Keep at least 100 GiB free on the selected output filesystem and 32 GiB free in
`/dev/shm`; allow more space for several retained trained policies.

## Restore a checkout and its inputs

The existing public queue configs retain the canonical path
`/home/yifei/icr`. Use that checkout path for the simplest faithful restoration.
If another location is required, the saved recipes below use path variables;
the artifact guide explains which copied metadata/config paths are rebased.
Do not bulk-replace paths in a checkout with a running experiment.

```bash
git clone --recurse-submodules git@github.com:FlyCole/icr.git /home/yifei/icr
cd /home/yifei/icr
git rev-parse HEAD
python3 scripts/vendor_source_backup.py apply
python3 scripts/vendor_source_backup.py check
```

The ICON submodule needs access to its recorded GitHub repository. Keep its
exact gitlink revision; do not update submodules to branch tips. Vendor changes
are restored from patches, so submodules appearing dirty afterward is expected.

Follow [the environment guide](environment_reproduction.md) to create
`rlinf/` (Python 3.11.15), `robometer/` (Python 3.10.20) and `.venv-icon/`
(Python 3.10.20), then restore their small source overlays. In particular,
install the OpenPI Transformers fork last in RLInf; its imported version differs
from the stock Transformers distribution metadata. Keep the two reward
environments separate because they install the same Python package name.

The most reliable large-input route is a verified bundle from the source
machine. These commands are preparation/transfer commands, not training:

```bash
# Source machine: inspect size first; use a new destination on a spacious disk.
python3 scripts/repro_artifacts.py export-bundle --bundle /mnt/transfer/icr-inputs
python3 scripts/repro_artifacts.py export-bundle --bundle /mnt/transfer/icr-inputs --apply

# Copy /mnt/transfer/icr-inputs to the destination computer, then there:
# Fresh destination only: keep SAPIEN's stock OIDN 2.0.1 files out of the import.
# Skip this move if lib/ is absent; never do it to a running source environment.
mv rlinf/lib/python3.11/site-packages/sapien/oidn_library/lib \
   rlinf/lib/python3.11/site-packages/sapien/oidn_library/lib.wheel-2.0.1-backup
python3 scripts/repro_artifacts.py import-bundle --bundle /mnt/transfer/icr-inputs --apply
python3 scripts/repro_artifacts.py restore-small --apply
export OPENPI_DATA_HOME="$PWD/data/runtime/openpi"
python3 scripts/repro_artifacts.py check --full-hashes
```

For task-specific SFT comparisons, include the optional legacy checkpoint as
described in the artifact guide. That guide also gives the pinned redownload
route when transferring a bundle is inconvenient. Recreate environments before
restoring native files into their site-packages. Keep the resulting native
filenames and the recorded hashes.

Do not copy the old `outputs/` tree wholesale: stale locks/PIDs/status files
are machine-specific, and most historical replay buffers were intentionally
removed. Restoring archived cache proof files preserves historical evidence;
it does not claim that the destination GPU has already been tested.

## Validate the destination

Run the source/artifact checks above, then these read-only/CPU checks:

```bash
rlinf/bin/python reproducibility/environments/restore_installed_overlays.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src:$PWD/third_party/rlinf" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  rlinf/bin/python -m pytest -q tests
```

With available GPU 0 headroom, check SAC CPU-offload and checkpoint parity:

```bash
RUN_DSRL_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$PWD/src:$PWD/third_party/rlinf" \
  rlinf/bin/python -m pytest -q tests/test_dsrl_offload_gpu.py
```

For ICON, rerun the cached/uncached comparison on the destination. Use fresh
reports rather than overwriting the source-machine proof files:

```bash
env -u LD_PRELOAD -u NCCL_P2P_DISABLE CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$PWD/third_party/robometer_v2" \
  .venv-icon/bin/python scripts/benchmark_icon_context_kv_cache.py \
  --tasks adjust_bottle move_can_pot click_bell --precision fp32 \
  --logit-atol 0.0001 --logit-rtol 0.00001 \
  --progress-atol 0.00001 --probability-atol 0.00005 \
  --output outputs/reproduction_validation/icon_fp32.json

env -u LD_PRELOAD -u NCCL_P2P_DISABLE CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$PWD/third_party/robometer_v2" \
  .venv-icon/bin/python scripts/benchmark_icon_context_kv_cache.py \
  --tasks adjust_bottle move_can_pot click_bell --precision native \
  --output outputs/reproduction_validation/icon_native.json
```

The full queue performs its own real SAC smoke before each method, discards
that smoke controller/replay, then starts fresh training. `--smoke-only` runs
only the first selected method's smoke and exits; use a separate fresh output
directory for the later full queue.

## Run the current matched comparison

The current backbone is `motus-robotics/pi0.5_robotwin2`, revision
`effc6e1a99bd4d0edf9447e8ba313def67649cfe`. It predicts 32 actions, executes 20,
and uses five flow steps. SAC controls independent 20×32 noise, copying the
first 12 rows to fill the decoder horizon. The frozen decoder is not updated.

Training uses 400 collections × 8 episode attempts, global batch 128,
microbatch 64 per rank, readiness 25 collections, and 20 critic / 10 actor updates
per eligible collection. Total optimizer counts are 7,520 critic and 3,760 actor
updates. Learned rewards are `s - 0.3 * (1 - p)`, with native simulator success
`s` and ICON/Robometer progress `p`; no reward normalization or progress success
threshold is used. Native success controls termination and padding is filtered
from replay sampling. ICON uses one fixed demonstration throughout the task.

On the destination, create a dedicated empty output directory on a filesystem
with enough space. The example is relative to the canonical checkout; it may
instead be an absolute NVMe directory or a symlink to one.

```bash
mkdir -p outputs/reproduce_move_can_pot_lambda03
export OPENPI_DATA_HOME="$PWD/data/runtime/openpi"

rlinf/bin/python scripts/run_motus_click_bell_noise20_queue.py \
  --task move_can_pot --methods simulator icon robometer \
  --penalty-weight 0.3 --low-gpu-memory \
  --output-root "$PWD/outputs/reproduce_move_can_pot_lambda03" --check-only

nohup ./rlinf/bin/python -u scripts/run_motus_click_bell_noise20_queue.py \
  --task move_can_pot --methods simulator icon robometer \
  --penalty-weight 0.3 --low-gpu-memory \
  --output-root "$PWD/outputs/reproduce_move_can_pot_lambda03" \
  > outputs/reproduce_move_can_pot_lambda03/supervisor.log 2>&1 < /dev/null &

tail -n 30 -F outputs/reproduce_move_can_pot_lambda03/current.log
```

Each method runs smoke → fresh training → its own final 150-scene evaluation.
The queue starts/stops GPU 0 reward services on ports 8002/8000 automatically;
leave these ports free. The simulator method has no reward server. Results go
to `<task>/<method>_run/evaluation.json`, then `summary.json`; failures go to
`experiment_status.json`. `progress.log` records stage/collection progress.

For bottle use `--task adjust_bottle` and a different output directory; its
matched cap is 200. Use `--penalty-weight 0.1` for the completed lambda 0.1
comparison. `click_bell` has a 400-action cap. Despite their `_lambda01` names,
the inherited bottle/can-pot YAMLs receive the actual lambda through recorded
Hydra overrides. Do not rename or edit those files to change lambda.

## Re-evaluate a recorded policy or rerun an older recipe

Choose an ID from [the ledger](experiment_results.csv) or
[`experiments.json`](../reproducibility/experiments.json). The saved snapshots
have no dependency on ignored TensorBoard config files. The wrapper checks
their SHA and resolves repository/output/checkpoint paths without changing
learning fields. `--check-only` prints the exact command without creating a
Ray directory, output directory, GPU process or server.

Matched frozen 400-action baseline (recorded 26/150):

```bash
rlinf/bin/python scripts/run_recorded_experiment.py \
  --experiment motus_sft_canpot_400 --stage eval \
  --output-root outputs/reproduce_canpot_sft_400 --check-only
# Remove --check-only to execute that evaluation.
```

Re-evaluate the existing trained simulator policy (recorded 58/150): transfer
its final step 400 checkpoint first, then point to the transferred directory:

```bash
rlinf/bin/python scripts/run_recorded_experiment.py \
  --experiment motus_noise20_canpot_simulator_400_lowmem --stage eval \
  --checkpoint /mnt/transferred/global_step_400 \
  --output-root outputs/reproduce_canpot_simulator_eval --check-only
```

For recorded fresh training, select `--stage train`. The base policy and all
training settings come from the saved config; `--checkpoint` is evaluation-only.
Historical learned-reward training requires its matching reward service to be
started separately. Prefer the managed current queue for the latest comparison;
do not substitute another context/server/reward formula into a historical recipe.

## Interpreting reproduced results

All ledger scores are native simulator success over 150 held-out scenes.
Training episodes revisit the disjoint training seed set; training MA100 is
not a final score. Do not compare the 200-action can-pot baseline 14% against
the 400-action DSRL score without also showing the matched 400-action baseline 17.33%.

Re-evaluating transferred final weights is different from retraining a new
controller. Same versions, pixels, assets, seeds and hyperparameters preserve
the experiment protocol, but GPU/native-kernel/rendering differences can change
trajectories and a new training run's score. No multi-training-seed statistical
claim is established by these single-training-seed results.

Each record lists remaining historical gaps. The original 200-collection
simulator training config was removed; several old checkpoints/replays are
absent; old converter-bug runs need their historical implementation. Available
launch hashes identify old content but do not recreate missing source files.
Do not relabel a current-code rerun as an exact reproduction of a known buggy
historical result. The source backup plus the latest recorded recipes is the
supported baseline for new matched experiments.
