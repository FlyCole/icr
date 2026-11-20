# Motus DSRL with lower GPU memory usage

The `--low-gpu-memory` execution profile keeps the learning settings and moves
the actor's online model, target model, FP32 target averages, gradients and
optimizer state to CPU between GPU phases. The rollout policy already supports
CPU offloading. CPU transfers add overhead.

There is no need to increase the number of learning iterations: the full run
still uses 400 collections. The extra time comes from moving tensors between
CPU and GPU, with the same batch sizes and optimizer update schedule.

Reward services are started only for their own training stage, on GPU 0, and
stopped before native-success evaluation. Simulator training uses no reward
server. Existing external services are left alone; an occupied required port
causes the managed profile to refuse startup.

The following settings are unchanged:

| Setting | Value |
| --- | --- |
| Initial policy | Frozen Motus `pi0.5_robotwin2` |
| Task / action limit | `move_can_pot` / 400 |
| Predicted / executed horizon | 32 / 20 |
| DSRL noise | 20 × 32 independent coordinates |
| Flow steps | 5 |
| Collections / episodes per collection | 400 / 8 |
| Global / per-rank microbatch | 128 / 64 |
| Critic / actor rounds per eligible collection | 20 / 10 |
| Replay readiness | 25 collections |
| Actor / critic learning rates | 5e-5 / 1e-4 |
| Entropy α / per-action γ / target τ | 0.0005 / 0.999 / 0.005 |
| Learned reward | `native_success - 0.3 * (1 - progress)` |
| ICON context | Same fixed successful task demonstration |
| Terminal padding | Filtered from replay sampling |
| Final evaluation | 150 held-out seeds per method |

## Start the full comparison

The previous attempt has no main training checkpoint: its 24 collections were
warmup. Start fresh from Motus and use the separate prepared NVMe output directory.
Run from `/home/yifei/icr`:

```bash
nohup ./rlinf/bin/python -u scripts/run_motus_click_bell_noise20_queue.py \
  --task move_can_pot \
  --methods simulator icon robometer \
  --penalty-weight 0.3 \
  --low-gpu-memory \
  --output-root /home/yifei/icr/outputs/motus_noise20_move_can_pot_lambda03_lowmem \
  > /home/yifei/icr/outputs/motus_noise20_move_can_pot_lambda03_lowmem/supervisor.log 2>&1 < /dev/null &
```

Each method runs an integration check, fresh training and its own final
evaluation before the next method starts. The simulator, ICON and Robometer
controllers and replay buffers are independent. Existing output is never resumed
or overwritten by this queue.

## Monitor

Follow the currently active training or evaluation log:

```bash
tail -n 30 -F /home/yifei/icr/outputs/motus_noise20_move_can_pot_lambda03_lowmem/current.log
```

For stage and collection updates:

```bash
tail -F /home/yifei/icr/outputs/motus_noise20_move_can_pot_lambda03_lowmem/progress.log
```

`Ctrl+C` stops the viewer. Final results go to `summary.json` under the same
output directory. The authoritative failure/stage record is
`experiment_status.json`; `supervisor.log` includes startup failures that occur
before that record is created.

## Validation

Repository checks passed **670 tests**, with 18 optional/dependency-based skips.
The three opt-in GPU parity/checkpoint tests passed separately. Final queue
preflight and vendor source-backup verification also passed.

A three-collection simulator check completed on both GPUs in 386 seconds,
using the real Motus checkpoint, 400-action limit, global batch 128, microbatch
64 and 20 critic rounds per eligible collection. Its warmup was shortened to
two collections for validation only. Both ranks saved 40 critic updates and
20 actor updates, with finite losses and nonzero gradients; their learned
controllers and target weights matched exactly. Checkpoint and replay saving
also completed. [Local validation receipt](../outputs/motus_lowmem_validation_20261119/validation_report.json).
The disposable validation checkpoint was removed afterward; metrics, memory
samples and validation receipts are retained.

During rollout after learning, each actor process used about 3 GiB of VRAM,
down from roughly 39 GiB of residency in the failed run. At two-second sampling,
total device usage peaked at 47.2 GiB on GPU 0 and 85.3 GiB on GPU 1, including
the unrelated job already using GPU 1. This validates a short run with the
current shared GPU workload; it does not measure the entire 400-collection run
or reward-server stages.

Separate fixed-input tests compare the real FSDP SAC heads with and without
CPU offload. Losses, gradients, online/target weights, optimizer moments,
schedulers, FP32 target averages and random-generator state match exactly
across three updates. This checks the transfer path; a complete training run
is not promised to reproduce every floating-point result bit for bit.
A separate checkpoint test restores a fresh CPU-offloaded FSDP worker and
confirms that its next SAC update also matches exactly.
