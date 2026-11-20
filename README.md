# RoboTwin + π0.5 + DSRL + Robometer / ICON

This checkout trains RLinf's SAC controller to steer a frozen π0.5 policy on RoboTwin. Completed `adjust_bottle` experiments use a task-specific SFT decoder or the released multi-task Motus decoder. The current Motus experiment replaces repeated noise with a 20-row noise controller and compares ICON, Robometer and native simulator rewards. Earlier multi-task results are retained below. PPO recipes are historical experiments.

## Start on another computer

Read **[the reproduction guide](docs/REPRODUCE.md)** first. It links the exact
environment snapshots, model/asset revisions, fixed ICON demonstrations, vendor
patches, and saved experiment configurations. **[AGENTS.md](AGENTS.md)** gives
Codex the same reading order and execution rules automatically.

Git contains code, configurations, vendor patches, small reproducibility
inputs, and the results ledger. Large pretrained/trained weights and simulator
assets require the separate artifact procedure; copying virtual environments
or cloning Git alone is insufficient. Preserve the exact context-frame hashes,
software versions, seeds and action caps when comparing results. A fresh
training run on different hardware is not guaranteed to return an identical score.

The [versioned results ledger](docs/experiment_results.csv) and
[saved recipes](reproducibility/experiments.json) describe completed formal
evaluations. Dated sections below preserve earlier experiments; old PIDs,
absolute output links and statements about running processes are local history.

Backup validation on November 20: **692 CPU tests passed, 18 skipped**. A
temporary clean checkout restored all four vendor patches and the exact context
frames successfully. Installed-source overlays and the local artifact inventory
also passed their checks. A fresh environment installation and GPU rerun on the
destination remain part of the reproduction guide.

## Move-can-pot with lower GPU memory usage — November 20, 2026

The frozen Motus baseline with a 400-action limit completed **26/150 successes
(17.33%)**. The lower-memory simulator DSRL run completed training and evaluated
its final step-400 checkpoint at **58/150 (38.67%)**, a gain of **21.33 percentage
points**. Both use the same 400-action cap, 20-action execution and five flow steps.

| Policy | Action limit | Held-out result | State at this documentation snapshot |
| --- | ---: | ---: | --- |
| Frozen Motus, earlier baseline | 200 | 21/150 (14.00%) | Completed; different action cap |
| Frozen Motus, matched baseline | 400 | 26/150 (17.33%) | Completed |
| Motus + simulator sparse DSRL | 400 | 58/150 (38.67%) | Completed |
| Motus + ICON DSRL, lambda 0.3 | 400 | Pending | Training; no final evaluation yet |
| Motus + Robometer DSRL, lambda 0.3 | 400 | Pending | Queued after ICON |

The queue now supports `--low-gpu-memory`: SAC models and optimizer state move
to CPU between GPU phases, and each reward server loads on GPU 0 only during
its own method's training. Learning hyperparameters, including global batch
128, microbatch 64, independent 20×32 noise and the 400-action limit, are
preserved. See the [profile, commands and validation](docs/motus_low_gpu_memory.md)
for a fresh simulator → ICON → Robometer comparison. The simulator checkpoint
audit confirmed 7,520 critic / 3,760 actor updates and identical controller
replicas. All formal evaluations use 150 held-out scenes and native success,
without reward-server calls or optimizer updates.

The first attempt, before this memory profile, stopped after 24 warmup
collections with an OOM and made no main SAC updates. It is not a completed
comparison and must not be resumed as a trained checkpoint.

## Cleanup and source backup — November 6, 2026

Cleanup reclaimed **577.08GiB** by removing17 inactive completed-run replay buffers, nine disposable smoke checkpoint directories, canceled/staged run copies and duplicated PPO distributed training shards. Another56 inactive Ray, diagnostic and generated cache paths were removed. All18 distinct final inference checkpoints, pretrained models, context demos, completed evaluation metrics and the active lambda0.3 run were preserved; its220 source/model/context pins were verified unchanged. [Checkpoint retention receipt](docs/checkpoint_retention_20261106.json), [full local cleanup receipt](outputs/cleanup_audit/cleanup_20261106.json).

The retained completed policies support evaluation. Their replay buffers were removed, so they no longer support exact training resume; each affected checkpoint has a removal marker with the former replay metadata. Negative results retain final inference weights as comparison evidence. The cleanup snapshot recorded27 completed150-seed evaluations. The [versioned experiment results](docs/experiment_results.csv) have since been updated with the completed lambda0.3 bottle comparisons and400-action can-pot results.

Useful collectors, evaluators, launchers, diagnostics and regression tests are included in the source backup. Model/data/environment/run artifacts and generated caches remain local and ignored; `.env` files are also ignored. The four vendor patch backups match the live submodule source. Validation passed610 CPU tests with14 dependency-based skips.

## Adjust-bottle lambda0.3 comparison — November 6, 2026

This series completed on November6. ICON scored **57/150 (38.00%)** and Robometer
**67/150 (44.67%)**, each from its own final step-400 checkpoint. It kept the
fixed Motus context, independent20×32 noise,200-action cap and matched SAC
settings of the lambda0.1 trial, changing the reward to
`native_success - 0.3 * (1 - estimated_progress)`. Each trained400 collections
(3,200 episode attempts); reward normalization remained disabled. The matched
lambda0.1 results were48.00% for ICON and52.67% for Robometer. These are separate
single-training-seed comparisons, not repeated-seed confidence estimates.

The [launcher](scripts/run_motus_click_bell_noise20_queue.py) accepts `--task adjust_bottle --penalty-weight 0.3`. It applies the weight through Hydra overrides to the existing lambda0.1 base configs and records the actual weight in saved resolved configs, command arguments, manifests and reward sidecars. Fresh outputs use NVMe-backed `outputs/motus_noise20_adjust_bottle_lambda03/`; follow [queue status](outputs/motus_noise20_adjust_bottle_lambda03/experiment_status.json). Final results are written to `adjust_bottle/{icon,robometer}_run/evaluation.json` and combined in `summary.json`.

The detached lambda0.3 queue launched at13:30:58 Shanghai time on November6,
supervisor PID1813135, after65 queue tests and host preflight passed, and later
finished both evaluations. [Completed local results](outputs/motus_noise20_adjust_bottle_lambda03/summary.json).
Its historical PID must not be treated as a current process identifier.

## Adjust-bottle lambda0.1 reward queue — November 5, 2026

The user switched the lambda0.1 comparison to **`adjust_bottle` first**. The click-bell queue below was stopped during ICON training, before any final evaluation or Robometer training. The [queue launcher](scripts/run_motus_click_bell_noise20_queue.py) supports task selection and runs **ICON training/evaluation → Robometer training/evaluation** with `--task adjust_bottle`. Each method runs a disposable two-collection integration check before fresh training; it never resumes the stopped click-bell controller or replay.

The lambda0.1 bottle queue completed on November6 at06:13:01 Shanghai time. ICON scored **72/150 (48.00%)** and Robometer **79/150 (52.67%)**, each from its own final step400 checkpoint after400 collections and3,200 episode attempts. [Completed results and checkpoint-load evidence](outputs/motus_noise20_adjust_bottle_lambda01/summary.json). Supervisor316397 cleaned up its owned workers and ICON service without errors. The preceding click-bell run stopped cleanly at collection8; its partial diagnostic outputs are retained.

Both new bottle trials retain the Motus backbone, independent20×32 learned noise, decoder horizon32 with a repeated12-row tail,20-action execution, five flow steps, terminal-padding filtering and identical SAC settings. The native bottle episode cap is **200 actions**, matching the completed bottle comparisons. Each fresh run trains400 collections ×8 episode attempts (3,200 attempts), then evaluates its own step400 checkpoint on150 held-out seeds using native simulator success and no reward service or policy updates. The retained frozen-Motus baseline is [48/150 (32.00%)](outputs/motus_multitask/adjust_bottle/vanilla_evaluation.json).

Reward is `s - 0.1 * (1 - p)`, with `reward.model.progress_penalty_weight: 0.1`, unscaled native success, `reward_scale: 1`, and no reward normalization. ICON uses the one fixed successful [Motus bottle context](data/context_demos/motus/adjust_bottle/demo_000/metadata.json), collected on training seed1; object-bank matching is disabled. Its cached service is owned on8002. Robometer uses the existing verified8000 service. The historical bottle configs below remain at lambda1.0.

| Method | New training config | New final150-seed config |
| --- | --- | --- |
| ICON, fixed context, lambda0.1 | [ICON](configs/robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20_lambda01.yaml) | [ICON evaluation](configs/robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20_lambda01_eval.yaml) |
| Robometer, lambda0.1 | [Robometer](configs/robotwin_adjust_bottle_motus_pi05_dsrl_robometer_noise20_lambda01.yaml) | [Robometer evaluation](configs/robotwin_adjust_bottle_motus_pi05_dsrl_robometer_noise20_lambda01_eval.yaml) |

Fresh outputs and checkpoints use the NVMe-backed `outputs/motus_noise20_adjust_bottle_lambda01/`. Follow [queue status](outputs/motus_noise20_adjust_bottle_lambda01/experiment_status.json). Results are written to `adjust_bottle/{icon,robometer}_run/evaluation.json`, then collected in `summary.json` after both evaluations finish. Training retains at most eight MP4s per method; final evaluation recording is disabled. The launcher checks native task/caps, the fixed demo, lambda, noise architecture and exact checkpoint loading before advancing.

## Click-bell lambda0.1 reward queue — November 5, 2026

The [click-bell launcher](scripts/run_motus_click_bell_noise20_queue.py) now runs **ICON → Robometer**, completing each method's fresh training and final evaluation before starting the next. The earlier simulator → ICON → Robometer queue (supervisor209311) was canceled during simulator training at collection25. Its unfinished outputs and two owned Ray temporary directories were deleted, and its partial state is not resumed. The successful context demonstrations and retained completed comparisons remain separate. [Cleanup receipt](outputs/cleanup_audit/click_bell_canceled_20261105.json).

The lambda0.1 click-bell queue launched at19:36:34 Shanghai time, supervisor PID278323, after129 CPU tests and host preflight passed. It was subsequently stopped during partial ICON training at the user's request to prioritize the bottle comparison above. [Launch receipt](outputs/motus_noise20_click_bell_lambda01/launch.json). It has no final policy evaluation or Robometer training result.

The new single-stage chunk reward is `s - lambda * (1 - p)`: `s` is the native simulator success indicator (0 or1), `p` is the ICON or Robometer progress estimate, and `reward.model.progress_penalty_weight` sets `lambda`, explicitly0.1 in both new training configs. With this setting a nonsuccessful chunk receives a reward in[-0.1,0], while a successful chunk receives[0.9,1]. `reward_scale` remains1 and rewards are not normalized. Native success is added once by the reward adapter; the separately weighted simulator reward stays0 to avoid adding it again. Older reward recipes explicitly retain weight1.0, reproducing `progress - 1 + native_success`.

ICON uses the single fixed successful [Motus context demonstration](data/context_demos/motus/click_bell/demo_000/metadata.json), collected on training seed100001, for every live bell variant and arm. Object-bank matching is disabled for this experiment; the separately collected object-bank references are retained. The queue owns its cached ICON service on8002 during ICON collection and training. Robometer uses the verified existing8000 service without context. All final evaluations measure native simulator success without reward-model calls or policy updates.

Both methods use frozen Motus, a learned20×32 noise controller, decoder horizon32 with the first12 noise rows copied into the tail,20 executed actions, five flow steps, a400-action episode cap, terminal-padding filtering and identical SAC settings. Each trains for400 collections ×8 episode attempts (3,200 attempts), then evaluates its own step400 checkpoint on the same150 held-out seeds. Each method first runs a disposable two-collection integration check; its controller and replay are discarded before fresh training. The retained frozen-Motus baseline is [65/150 (43.33%)](outputs/motus_multitask/click_bell/vanilla_evaluation.json).

| Method | Training config | Final150-seed config |
| --- | --- | --- |
| ICON, fixed context, lambda0.1 | [ICON](configs/robotwin_click_bell_motus_pi05_dsrl_icon_noise20.yaml) | [ICON evaluation](configs/robotwin_click_bell_motus_pi05_dsrl_icon_noise20_eval.yaml) |
| Robometer, lambda0.1 | [Robometer](configs/robotwin_click_bell_motus_pi05_dsrl_robometer_noise20.yaml) | [Robometer evaluation](configs/robotwin_click_bell_motus_pi05_dsrl_robometer_noise20_eval.yaml) |

Outputs and checkpoints use the fresh NVMe-backed `outputs/motus_noise20_click_bell_lambda01/`. Follow [queue status](outputs/motus_noise20_click_bell_lambda01/experiment_status.json) or `tail -f outputs/motus_noise20_click_bell_lambda01/supervisor.log`. Each result is written to `click_bell/{icon,robometer}_run/evaluation.json`; both results are collected in `summary.json` after completion. Training retains at most eight MP4s per method; these final evaluations have video recording disabled. The launcher requires a fresh destination and stops with recorded status if a stage fails.

## Source backup and restoration

Git stores the experiment code, configurations, regression tests and documentation. Model weights, datasets, environments, run outputs, logs and generated caches remain local and are ignored.

The four repositories under `third_party/` stay pinned to their upstream commits. Their local source changes are also backed up in `vendor_patches/`, including new files and exact base revisions, so pushing this repository preserves the modifications without separate upstream pushes. After cloning with submodules, restore and verify those changes:

```bash
git clone --recurse-submodules https://github.com/FlyCole/icr.git
cd icr
python scripts/vendor_source_backup.py apply
python scripts/vendor_source_backup.py check
```

After editing a submodule, run `python scripts/vendor_source_backup.py snapshot` before committing the updated `vendor_patches/` files. Submodules still appear modified locally after restoration; the patch snapshots are the parent repository's backup of those edits. The helper's `check` command verifies the backup without changing runtime source files.

## Storage and integrity audit — November 2, 2026

The authorized cleanup completed at17:27 Shanghai time. **No checkpoint structure corruption was detected.** All136 active Motus source/config/model/context pins were unchanged,19 owned Python files parsed successfully, and6,246 historical PyTorch ZIP archives passed central-directory/member-extent checks. The archive audit did not load every tensor or scan full payload CRCs. All ten distinct comparison checkpoints, completed evaluations/TensorBoards, the valid interrupted official PPO step10 checkpoint, and the active Motus queue/model/context demos were retained. [Checkpoint retention audit](outputs/cleanup_audit/historical_recommendations.json), [archive inventory](outputs/cleanup_audit/historical_checkpoints_inventory.json).

To free space without discarding valid results, the two completed stable ICON/Robometer checkpoint trees were copied to `/media/yifei/workspace_nvme/icr_retained/outputs/` with the same relative paths. All1,618 copied files were verified by full SHA256 before the root-filesystem originals were removed. Their existing `outputs/.../checkpoints` paths now resolve through symlinks, so evaluation commands remain valid. This reclaimed109.06GiB. A further32 obsolete failed-startup/incomplete-run/temp artifacts, including23 inactive `/tmp/ray` sessions, were deleted (484MiB); no healthy comparison was removed. Root free space increased from8.10GiB to117.63GiB, with approximately1.86TiB remaining on the external NVMe. [Completed cleanup receipt and exact paths](outputs/cleanup_audit/cleanup_2026-11-02.json).

## Rollout capture and recording defaults — November 5, 2026

RoboTwin defaults to `wrist_camera_capture: chunk_end`: head-camera observations remain dense, while both wrist cameras are captured at reset and at the last real action of each chunk, including early termination. The fresh three-view observation sent to Motus remains intact. Set `all_steps` to restore full wrist capture after every action. Depth, segmentation and point-cloud collection retain full-camera capture; online LeRobot collection requires `all_steps` so its per-action dataset remains complete.

Training defaults to **at most8 MP4 files across all environment workers and pipeline stages**, spaced across the configured training budget. Set `max_videos: K` for another cap, `0` for no training MP4s, or `-1` for every recording. With the current two workers,400 collections and four rounds per collection, each worker records four of its1,600 episode slots. A recording can contain tiled environments when a worker owns more than one. Evaluation recording remains unlimited when enabled unless its own cap is explicitly set.

Configure these fields in the [shared DSRL configuration](configs/_robotwin_adjust_bottle_pi05_dsrl_base.yaml), or override them in a public experiment YAML:

```yaml
env:
  train:
    task_config:
      wrist_camera_capture: chunk_end  # all_steps restores the previous capture mode.
    video_cfg:
      save_video: true
      max_videos: 8  # Run-wide MP4 budget; 0 disables MP4s, -1 records all.
```

Policy endpoint images, reward head-camera histories and action masks retain their existing contents. Reward-score JSON sidecars continue for every scored episode, including object/demo identity, exact causal progress, native outcome and logical frame indices. A sidecar's `video_recorded` identifies whether its MP4 was selected. Skipped slots bypass video pixel copies, overlays, footer drawing and encoding; they do not disable reward scoring. Keep `save_video: true` with `max_videos: 0` when JSON diagnostics are desired without MP4s.

Real GPU checks on training scenes1 and7 confirmed bitwise equality of every head frame, reset and endpoint wrist images, states, commands, rewards and terminal masks between capture modes. Across20 actions, wrist captures/readbacks drop from40 to2. This native setup also captures a static front camera, so total camera captures drop from80 to42; front-camera behavior is retained. Single-chunk timings under shared load are recorded, but do not establish full-training throughput. [Seed1 proof](outputs/staged_rollout_defaults_20261105/capture_seed1.json), [seed7 proof](outputs/staged_rollout_defaults_20261105/capture_seed7.json).

These changes were prepared separately while the preceding object-context experiment was running. Its source identity checks require that it finish training and final150-scene evaluation before installation. The deferred installer verifies completion and unchanged originals, retains backups and refreshes vendor patches. [Installation status](outputs/staged_rollout_defaults_20261105/apply_status.json). An interrupted or failed preceding run leaves the prepared changes unapplied.

## Object-matched ICON context — November 4, 2026

ICON can now select a successful demonstration by the live native object asset. For `adjust_bottle`, the task samples exactly `001_bottle/base13` and `001_bottle/base16`. Selection ignores the arm, object pose and scene seed. It raises an error for missing or unknown identities rather than silently using another object's demo. The previous single-reference mode remains the default (`context_match_mode: fixed`).

**Completed November5:** the object-matched ICON trial scored **62/150 (41.33%)** using its own final step400 checkpoint, after400 collections and3,200 training episode attempts. Fixed-context noise20 ICON scored66/150 (44.00%), simulator-reward noise20 scored77/150 (51.33%), and frozen Motus scored48/150 (32.00%). Object matching produced four fewer successes than fixed context in these single-seed runs; this does not establish the cause or a reliable method difference. [Final result and checkpoint-load proof](outputs/motus_noise20_object_context/adjust_bottle/icon_run/evaluation.json). Evaluation finished at03:49:43 Shanghai time; the camera/video runtime changes were installed afterward at03:50:08 and did not affect this result. The [complete24-evaluation inventory](outputs/diagnostics/experiment_inventory_20261105.csv) includes exact metric/config/checkpoint paths and current checkpoint availability.

The [context-bank settings](configs/icon_context_bank_adjust_bottle.yaml) list the required assets, collection policy and server settings. The completed [bank manifest](data/context_demos/motus/adjust_bottle/object_bank/manifest.json) contains one successful frozen-Motus reference per asset:

| Native asset | Training scene seed | Arm / actions to first success | Preview |
| --- | --- | --- | --- |
| `001_bottle/base13` | 1 | Left / 119 | [Video](data/context_demos/motus/adjust_bottle/object_bank/base13/demo_000/preview.mp4) |
| `001_bottle/base16` | 7 | Right / 137 | [Video](data/context_demos/motus/adjust_bottle/object_bank/base16/demo_000/preview.mp4) |

Each demo stores all raw head-camera frames from reset through first success, without padding. ICON uses eight uniformly selected context frames, as before. These references cover the evaluation **asset catalog**; they were collected on training seeds, independently of held-out evaluation trajectories. No brand names are inferred from asset IDs.

The option is explicit at the three interfaces: server `context_match_mode: object` plus `context_demo_bank_path` and `context_demo_path: null`; environment `env.train/eval.context_match_mode: object`; reward bridge `reward.model.context_match_mode: object`. The [matched training config](configs/robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20_object_context.yaml) and [evaluation config](configs/robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20_object_context_eval.yaml) set these fields. Existing fixed-context configs retain their prior behavior. The launcher checks that server, environment and reward modes agree.

Live native identity travels with the terminal observation and reward query, never as an extra policy feature. The server selects each sample's reference before collation, including mixed-object batches; its response and the training video-score sidecar record the selected object and demo SHA256. The existing context KV cache keys include processed context pixels and instruction, so different references remain separate. Reward remains raw `progress - 1 + native_success`; SAC, noise shape and training budget match the previous noise20 recipe.

Real bounded checks scored one training and one held-out episode per asset. All selected hashes matched the bank, including a left-arm live episode using the green bottle's right-arm reference. Reordered mixed-object requests produced bitwise-identical scores. Each split added20 cache hits, two misses and zero fallbacks. [Train-query report](outputs/diagnostics/icon_object_context_validation/train_queries/report.json), [held-out-query report](outputs/diagnostics/icon_object_context_validation/eval_queries/report.json). These are routing checks, not success-rate estimates.

The combined CPU suite passed166 tests, plus15 tiny-Qwen3-VL cache tests. The real two-collection SAC smoke passed40 critic/20 actor updates and verified finite parameter changes in all six controller groups between saved steps1 and2, identical distributed replicas, and40 valid chunks with correct object/demo hashes (20 per asset). Cache counters were38 hits, two misses and zero fallbacks. [SAC smoke proof](outputs/motus_object_context_smoke/adjust_bottle/icon_run/smoke_validation.json). Its owned workers/service are stopped; the two redundant full-model smoke files were removed after validation, reclaiming25.03GiB, while compact heads, optimizer/replay evidence and logs remain. The smoke is not a resumable full inference checkpoint or a performance comparison.

Normal final policy evaluation uses simulator success and does not call ICON. The separate [query checker](scripts/evaluate_icon_context_bank.py) verifies ICON selection on held-out observations. The new [launcher](scripts/run_motus_object_context_experiment.py) owns its port8002 service, runs a short integration check, then starts fresh400-collection training and evaluates its final checkpoint on150held-out seeds:

```bash
cd /home/yifei/icr
rlinf/bin/python -u scripts/run_motus_object_context_experiment.py
```

The first full object-context experiment uses `outputs/motus_noise20_object_context`, linked to NVMe. Its [recorded status](outputs/motus_noise20_object_context/experiment_status.json) and [final evaluation](outputs/motus_noise20_object_context/adjust_bottle/icon_run/evaluation.json) identify that trial and its checkpoint. New trials require a fresh `--output-root`; use `--smoke-only` for just the bounded SAC check. The [collector](scripts/collect_icon_context_bank.py) supports `--check-only` and partial-bank `--resume`; completed references are immutable and need not be recollected.

## Motus 20 × 32 noise controller — November 3, 2026

The previous cached multi-task queue was stopped at the user's request at14:38 Shanghai time on November3. Its completed results remain unchanged; do not restart that supervisor after the controller source change.

**November4 completion:** the fixed-context noise20 ICON policy scored **66/150 (44.00%)**, and the simulator-reward policy scored **77/150 (51.33%)**, versus the retained frozen-Motus baseline **48/150 (32.00%)**. Both400-collection runs and final evaluations completed. The separate object-context trial above completed on November5 with62/150 (41.33%). [Reward/checkpoint audit](outputs/diagnostics/noise20_reward_comparison_20261104.json).

**Robometer noise20 completed November5 at18:49:40 Shanghai time:** its own final step400 checkpoint scored **62/150 (41.33%)**. Fresh training completed400 collections and3,200 episode attempts using endpoint-only wrist capture and the run-wide eight-MP4 default. The verified original Robometer service on8000 supplied reward without ICON context. [Final result and checkpoint-load proof](outputs/motus_noise20/adjust_bottle/robometer_run/evaluation.json), [run status](outputs/motus_noise20/adjust_bottle/robometer_run/experiment_status.json), [training log](outputs/motus_noise20/adjust_bottle/robometer_train.log).

The launch passed48 CPU checks and its actual two-collection SAC smoke:40 critic/20 actor updates, finite losses and nonzero gradients,640-dimensional noise, and identical controller replicas. The disposable13.4GB full-model smoke copy was removed after validation. Fresh training starts independently from Motus and new controller/replay state. [Smoke proof](outputs/motus_noise20/adjust_bottle/robometer_run/smoke_validation.json).

The new `adjust_bottle` experiment samples20 distinct32-dimensional noise rows with a flat Gaussian MLP head, rather than repeating one32-dimensional vector. It appends the first12 rows to fill Motus's32-position decoder horizon, then executes20 physical actions. SAC replay, all ten critics and joint log probability use the original20×32 values; the copied tail is excluded. This implements temporal noise control inspired by [Robometer Appendix E-2](https://arxiv.org/html/2603.02115v1#A5.SS2), while retaining this checkout's MLP controller and image/state encoders. Legacy configs preserve the earlier head shapes and interpretation. New controllers start fresh; old repeated-noise checkpoints cannot initialize these640-dimensional heads.

The [shared training config](configs/_robotwin_motus_pi05_dsrl_noise20.yaml) retains400 collections, eight episode attempts per collection, a200-action episode cap,20-action execution, five flow steps, global batch128, per-rank microbatch64, actor/critic learning rates5e-5/1e-4,20critic/10actor updates per eligible collection, and terminal-padding filtering. Warmup remains24 collections before the first updates at collection25, followed by7,520critic and3,760actor updates. Fixed entropy coefficient is0.0005, divided by20 from the previous0.01 because joint log probability now sums640 coordinates. Rewards are not normalized. Training logs also include `train/actor/noise_event_dim`, `noise_temporal_std`, and `entropy_per_noise_dim`.

| Reward | Training config | Final150-seed config |
| --- | --- | --- |
| ICON, raw `progress - 1 + native_success`, fixed Motus demonstration and context KV cache | [ICON](configs/robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20.yaml) | [ICON evaluation](configs/robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20_eval.yaml) |
| Robometer, same raw reward and query history | [Robometer](configs/robotwin_adjust_bottle_motus_pi05_dsrl_robometer_noise20.yaml) | [Robometer evaluation](configs/robotwin_adjust_bottle_motus_pi05_dsrl_robometer_noise20_eval.yaml) |
| Native simulator success: +1 once at first success, zero otherwise | [Simulator sparse](configs/robotwin_adjust_bottle_motus_pi05_dsrl_simulator_noise20.yaml) | [Simulator evaluation](configs/robotwin_adjust_bottle_motus_pi05_dsrl_simulator_noise20_eval.yaml) |

The [launcher](scripts/run_motus_noise20_experiment.py) performs an actual two-collection smoke with optimizer updates and checkpoint/replay verification, discards its training state, runs fresh400-collection training, and evaluates its own final checkpoint on150held-out seeds. ICON starts its own cached reward server on8002 and stops it before evaluation. Robometer uses the existing validated server on8000; simulator reward requires no reward service. Every evaluation uses native simulator success with no policy updates. A lock permits one experiment at a time on the two shared GPUs.

```bash
cd /home/yifei/icr
# Historical launch commands; ICON and simulator runs are already complete.
# The launcher requires a fresh output destination for any new experiment.
rlinf/bin/python -u scripts/run_motus_noise20_experiment.py --method robometer
```

Outputs are under `outputs/motus_noise20/adjust_bottle/`, linked to the spacious NVMe drive. Check [recorded status](outputs/motus_noise20/experiment_status.json), [ICON console log](outputs/motus_noise20/adjust_bottle/icon_train.log) and `icon_train/tensorboard/`. Final ICON results are in `icon_run/evaluation.json`; simulator uses the corresponding method prefix. The supervisor owns its child sessions and leaves existing8000/8001services alone.

## Earlier Motus multi-task π0.5 comparison — November 3 snapshot

**Status updated November5.** Both `adjust_bottle` DSRL runs reached step400 and completed their final150-seed evaluations. The earlier queue was stopped on November3: `move_can_pot` ICON reached158/400 collections without a final evaluation, and the remaining learned-policy stages did not run. Each score below uses native simulator success on150 held-out episodes, with matched task settings and action execution:

| Task | Frozen Motus SFT | Motus + ICON / DSRL, step 400 | Motus + Robometer / DSRL, step 400 |
| --- | --- | --- | --- |
| `adjust_bottle` | [48/150, 32.00%](outputs/motus_multitask/adjust_bottle/vanilla_evaluation.json) | [4/150, 2.67%](outputs/motus_multitask_kv_cache/adjust_bottle/icon_evaluation.json) | [6/150, 4.00%](outputs/motus_multitask_kv_cache/adjust_bottle/robometer_evaluation.json) |
| `move_can_pot` | [21/150, 14.00%](outputs/motus_multitask/move_can_pot/vanilla_evaluation.json) | Stopped at158/400; no final evaluation | Not run |
| `click_bell` | [65/150, 43.33%](outputs/motus_multitask/click_bell/vanilla_evaluation.json) | Not run | Not run |

**Both completed Motus bottle controllers substantially degraded performance relative to frozen SFT.** These results do not show an improvement from either learned reward. The [ICON checkpoint audit](outputs/motus_multitask_kv_cache/adjust_bottle/icon_checkpoint_validation.json) and [Robometer checkpoint audit](outputs/motus_multitask_kv_cache/adjust_bottle/robometer_checkpoint_validation.json) confirm 7,520 critic and 3,760 actor updates per run, identical controller replicas, and saved full inference weights plus replay. Each evaluation record above identifies its own `global_step_400` checkpoint loaded on both rollout ranks with H32/execute20. This verifies the recorded training/evaluation path; the cause of the degradation has not been established. The single training seed and aggregate success counts cannot isolate an effect of context or KV caching.

At this snapshot, `move_can_pot` ICON has run for 2:52:55, averaging 67.8 seconds per collection. Its latest 100-episode moving training success rate is 9%; this is a training statistic, and no final held-out score is available yet. The last recorded collection took 66.0 seconds, while SAC updates took 1.54 seconds. [Training metrics](outputs/motus_multitask_kv_cache/move_can_pot/icon_train/metrics.log), [console log](outputs/motus_multitask_kv_cache/move_can_pot/icon_train.log).

The original queue was stopped at the user's request before introducing ICON's KV cache. Its incomplete `adjust_bottle/icon_train` directory was deleted; it had no checkpoint to resume. The diagnostic log and stopped manifest remain preserved, along with all three completed frozen-Motus baselines above. **The validated continuation launched at19:20:23 Shanghai time on November2, supervisor PID1605823**, under **`outputs/motus_multitask_kv_cache`**, with fresh controllers, optimizer state and replay for every task/reward pair. It reuses the completed vanilla evaluations because policy evaluation does not invoke the reward estimator. The sequence is task-by-task ICON smoke → ICON400 training → ICON150 evaluation → Robometer smoke → Robometer400 training → Robometer150 evaluation. The cached ICON/SAC smoke verified actual actor/critic updates before full training. Host preflight verified180 source/model/context/proof pins. At the November3 snapshot the active trainer is PID3684829, with its task-specific cached ICON service on port8002. The old supervisor PID1281557 is stopped; do not restart it.

Follow the [live stage status](outputs/motus_multitask_kv_cache/experiment_status.json) and [supervisor log](outputs/motus_multitask_kv_cache/supervisor.log). Stage status changes when a child starts or finishes; use the active task's training log or TensorBoard for optimizer progress. Final consolidated results will appear in `outputs/motus_multitask_kv_cache/summary.json` after the whole queue; that file is not present at this snapshot. `baseline_reuse.json` records the retained baseline evidence.

```bash
python -m json.tool /home/yifei/icr/outputs/motus_multitask_kv_cache/experiment_status.json
tail -F /home/yifei/icr/outputs/motus_multitask_kv_cache/move_can_pot/icon_train.log
```

### Fixed-context KV cache

The best reuse boundary is inside each reward-server GPU replica: the language prompt, fixed eight context images and their progress tokens precede all query images in ICON's causal Qwen3-VL decoder. The [cache](third_party/robometer_v2/robometer/models/context_kv_cache.py) stores that prefix's original-precision KV tensors and final hidden states, then evaluates only the query suffix. Cache keys include the exact prompt, processed context pixels, image grids and multimodal positions; different instructions never share a conditioned prefix. Each replica has an eight-entry/1GiB LRU cap. Query histories remain independent between requests. Full-sequence multimodal RoPE positions and Qwen DeepStack features are preserved, and model/precision/device changes invalidate reuse. Prediction heads, reward conversion and simulator termination retain their existing behavior.

The [reward server](third_party/robometer_v2/robometer/evals/eval_server.py) enables this only when a fixed context demo is supplied. Its launch options are `context_kv_cache=true`, `context_kv_cache_max_entries=8`, `context_kv_cache_max_mib=1024`; `context_kv_cache=false` restores full-sequence inference. `/health` exposes per-replica hits, misses, memory, evictions and fallbacks. The new supervisor requires real smoke-test cache hits and zero fallback before full training. The [completed bottle smoke](outputs/motus_multitask_kv_cache/adjust_bottle/icon_smoke_validation.json) verified finite actor/critic losses and nonzero gradients,36 prefix hits/4 misses (90% per-row hit rate), four cached prefixes using500,754,432bytes, and zero fallback across20 requests. Full training starts from fresh controller/replay state after that smoke.

Validation passed **15 real tiny-Qwen3-VL CPU tests** and two full-checkpoint, three-task GPU suites of15 cases each, covering one/two/eight query frames, batches, differing instructions and repeated requests. **Cached and uncached BF16 outputs are not bitwise identical:** changed computation shapes introduce rounding differences. The largest observed difference in final progress was **0.003591** (0.359 percentage points), in success probability **0.009766**, and in raw logits **0.167969**. Initial strict BF16 reports remain preserved as failed rather than being overwritten. The final native-precision acceptance bounds are progress0.005, success probability0.01, and logits0.2 +1% relative. A separate FP32 proof passed substantially tighter bounds: final progress differed by at most **3.76e-6**, success probability **2.25e-5**, and raw logits **1.02e-4**. The first cached call and repeated cached calls were bitwise equal in all tested cases. These are measured bounds on these inputs, not a universal error guarantee. [Native results](outputs/icon_kv_cache_validation/checkpoint_native_validated.json), [strict FP32 results](outputs/icon_kv_cache_validation/checkpoint_fp32_validated.json), [validation/deployment record](outputs/icon_kv_cache_validation/validation_summary.json).

Production retains BF16 for speed. Warm cached model forward was **1.40–1.92× faster** (median1.79×); including separately measured unchanged collation gives an estimated **1.32–1.64×** (median1.57×). This measures reward inference, not total simulator/training speed. One prefix occupies approximately120MiB. [Benchmark code](scripts/benchmark_icon_context_kv_cache.py), [CPU tests](tests/test_icon_context_kv_cache.py).

### Frozen Motus policy and comparison protocol

The selected decoder is [motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2), revision `effc6e1a99bd4d0edf9447e8ba313def67649cfe`: the released π0.5 SFT baseline, distinct from the larger Motus world/action model. Its model card reports joint training on 50 RoboTwin tasks with 2,500 clean and 25,000 randomized demonstrations. Downloaded metadata confirms a **32-action prediction horizon**, 32 internal action dimensions, 14-dimensional Aloha state/actions, three cameras, discrete state input, Aloha adaptation and delta arm-joint actions with absolute grippers. The new experiments execute **20 actions** before observing and replanning. **Five flow steps are our runtime choice**, not a verified inference setting from the release. This is an adaptation of the released weights, rather than a claim to reproduce the authors' benchmark scores; their exact inference recipe remains [undocumented in the open issue](https://github.com/thu-ml/Motus/issues/45).

| Task | Configured episode / rollout / simulator action limit | Maximum real chunks per episode |
| --- | --- | --- |
| `adjust_bottle` | 200 / 200 / 200 | 10 |
| `move_can_pot` | 200 / 200 / 200 | 10 |
| `click_bell` | 400 / 400 / 400 | 20 |

All three use the same frozen Motus π0.5 weights and their own task instructions, clean scenes and Aloha-AgileX embodiment. The prepared configs override `move_can_pot`'s upstream Piper default. They retain the raster renderer, disabled denoiser and single-thread rendering used in the completed comparisons. The table records the comparison's configured caps; the separate failure collector uses the official per-task horizon table, including400 actions for `adjust_bottle` and `move_can_pot`. Each task has its own completed150-seed Motus baseline above; the older task-specific bottle SFT score118/150 belongs to a different decoder.

Of the six planned independent DSRL runs, both bottle runs and their evaluations completed. The queue then stopped with `move_can_pot` ICON at158/400; no further stages ran. The newer bottle noise20 experiments above were separate fresh runs. Each task/method starts with new controller heads and replay; smoke updates are discarded. The shared Motus decoder stays frozen throughout. ICON uses its task's fixed eight-frame reference; original Robometer has no demonstration. Query history remains up to eight dense prefix frames, and reward remains unnormalized `progress - 1 + native_success` once per valid20-action chunk with terminal-padding filtering. Final policy evaluation uses native simulator success without reward-model inference or policy updates.

| Passed preparation | Four-episode simulator smoke | Successful training reference: actual seed / actions |
| --- | --- | --- |
| `adjust_bottle` | 1/4 | [1 /119](data/context_demos/motus/adjust_bottle/demo_000/metadata.json) |
| `move_can_pot` | 0/4 | [100000 /118](data/context_demos/motus/move_can_pot/demo_000/metadata.json) |
| `click_bell` | 2/4 | [100001 /120](data/context_demos/motus/click_bell/demo_000/metadata.json) |

These four-episode counts are integration checks, **not performance estimates**. Both rollout ranks loaded the intended H32/execute20 checkpoint for each task. Every reference reached native success using frozen Motus on a training seed with zero optimizer steps. [Per-task load/context evidence](outputs/motus_prepare/launch_preconditions.json). The separate bottle ICON/SAC smoke completed two iterations and four episode attempts; actor/critic losses and gradient norms were finite, with both gradient norms nonzero. It saved no checkpoint, and its owned simulator/Ray/ICON processes were cleaned up. [SAC validation](outputs/motus_prepare/sac_adapter_smoke_validation.json), [cleanup/status](outputs/motus_prepare/sac_adapter_smoke_status.json). This validates the update path, not learning improvement.

### Training recipe and measured runtime

The shared training recipe is recorded in the [resolved active config](outputs/motus_multitask_kv_cache/move_can_pot/icon_train/tensorboard/config.yaml):

| Setting | Motus comparison value |
| --- | --- |
| Trainable parameters | DSRL noise actor and critics; frozen Motus π0.5 decoder; FP32 controller heads |
| Collection budget | 400 iterations × 2 environments × 4 rounds = 3,200 episode attempts per run |
| Decoder / execution | Predict 32 actions, execute 20, five flow steps; 14-dimensional state/actions and 32-dimensional noise |
| Batch | Global 128 / micro 64 per actor rank; two synchronized ranks |
| Replay | Cache/window 512 per rank; filter post-terminal padding; readiness 25 received batches per rank |
| Optimizer rounds | 20 critic / 10 actor updates per eligible iteration; 376 eligible iterations give 7,520 / 3,760 updates |
| Learning rates | Actor `5e-5`; critic `1e-4` |
| Critics | Ten Q heads, all used; three 128-unit hidden layers |
| Discount / target update | Per-action gamma `0.999` (`0.999**20 ≈ 0.98019` per full chunk); target tau `0.005` after every critic update |
| Entropy | Fixed alpha `0.01`; automatic entropy tuning disabled |
| Learned reward | `single_stage`, scale 1, unnormalized `progress - 1 + native_success`; at most eight query frames per growing prefix |
| Saving / evaluation | Final step 400 checkpoint with inference weights, optimizer and replay; 150 held-out episodes after training |

| Completed Motus training loop | Measured time | Mean time per collection |
| --- | --- | --- |
| `adjust_bottle` ICON400 | 8:17:26 | 74.6 seconds |
| `adjust_bottle` Robometer400 | 7:16:36 | 65.5 seconds |

These loop times exclude separate final evaluations and reflect the shared workstation workload. [ICON training metrics](outputs/motus_multitask_kv_cache/adjust_bottle/icon_train/metrics.log), [Robometer training metrics](outputs/motus_multitask_kv_cache/adjust_bottle/robometer_train/metrics.log). The provisional 2–3-day estimate covers the whole six-run queue, rather than the remaining time at this snapshot. `click_bell` has twice the action budget of the other tasks. The KV-cache benchmark measures reward inference; total collection time also depends on simulation, policy outcomes and concurrent jobs.

### Configurations, checkpoints and adapter evidence

| Prepared configs (`<task>` is one of the three names above) | Purpose |
| --- | --- |
| `robotwin_<task>_motus_pi05_sft_eval` | Frozen Motus baseline,150 seeds |
| `robotwin_<task>_motus_pi05_dsrl_icon` / `_icon_eval` | Fresh ICON training / final policy evaluation |
| `robotwin_<task>_motus_pi05_dsrl_robometer` / `_robometer_eval` | Matched original Robometer training / evaluation |

Shared settings are in [_robotwin_motus_pi05_dsrl.yaml](configs/_robotwin_motus_pi05_dsrl.yaml), [_robotwin_motus_pi05_eval.yaml](configs/_robotwin_motus_pi05_eval.yaml), and [_robotwin_motus_pi05_dsrl_eval.yaml](configs/_robotwin_motus_pi05_dsrl_eval.yaml). Public task configs set the task, all three action limits and fresh output names. The [cached continuation](scripts/continue_motus_with_icon_kv_cache.py) requires passed preparation, actual cache parity reports, tested source hashes and completed baseline evidence before launching. It preserves the stopped original [supervisor](scripts/run_motus_multitask_experiment.py) and its manifest. Do not start duplicate stages when the continuation is active.

The [collector](scripts/collect_icon_context.py) now accepts `--task`, `--checkpoint`, `--norm-stats`, `--action-horizon`, `--action-chunk`, `--max-actions` and `--flow-steps`. The references above already exist; this bottle command documents collection and should not be rerun against a pinned reference:

```bash
cd /home/yifei/icr
CUDA_VISIBLE_DEVICES=0 rlinf/bin/python scripts/collect_icon_context.py \
  --task adjust_bottle --checkpoint checkpoints/Motus-Pi05-RoboTwin2 \
  --norm-stats checkpoints/Motus-Pi05-RoboTwin2/assets/pi0.5_clean_randomize_joint_training/norm_stats.json \
  --action-horizon 32 --action-chunk 20 --max-actions 200 --flow-steps 5 \
  --output data/context_demos/motus/adjust_bottle/demo_000
```

Use the corresponding task/output and `--max-actions 400` for `click_bell` when preparing a distinct reference. Collection retains reset through first native success, raw head-camera RGB and actual training-seed provenance; it excludes terminal padding and held-out scenes. The three new references are fixed for this comparison. The continuation uses a task-specific cached ICON service on port8002, leaving the old bottle service/reference separate; the earlier temporary integration-check service has stopped.

Storage uses a dedicated persistent NVMe volume with approximately2TiB free when prepared. Logical checkpoint path `checkpoints/Motus-Pi05-RoboTwin2` points to `/media/yifei/workspace_nvme/icr_motus/checkpoints/Motus-Pi05-RoboTwin2`; `outputs/motus_multitask` points to `/media/yifei/workspace_nvme/icr_motus/outputs/motus_multitask` and preserves the vanilla baselines. The fresh continuation root `outputs/motus_multitask_kv_cache` points to `/media/yifei/workspace_nvme/icr_motus/outputs/motus_multitask_kv_cache`. Each task writes `icon_train`, `icon_eval`, `robometer_train` and `robometer_eval` below that new root. Final checkpoints remain `<task>/<method>_train/pi05_motus_<task>_dsrl_<method>/checkpoints/global_step_400`, including full inference weights and training state. No policy-only checkpoint format was introduced.

The initial single-stream download failed with an incomplete HTTP response; the resumable parallel byte-range transfer subsequently completed. **The full7,473,091,464-byte model SHA256 was verified:** `de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb`. [Completed download/integrity record](outputs/motus_prepare/download.json), [part map](outputs/motus_prepare/download_parts.json), [transfer log](outputs/motus_prepare_download_parallel.log). Normalization SHA256 is `1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb`.

CPU preparation passed seven adapter-audit tests. The [completed full-file adapter audit](outputs/motus_prepare/adapter_cpu_audit.json) verifies the SHA, maps812 source tensors to all667 expected RLInf tensors with matching shapes using meta tensors, and checks the actual Motus normalization/I/O against installed upstream OpenPI Aloha transforms. Camera order/orientation, state tokens, joint/gripper conversion, inverse normalization and20-action cropping match exactly. The earlier [partial-header audit](outputs/motus_prepare/checkpoint_header_audit.json) and [transform report](outputs/motus_prepare/transform_cpu_parity.json) remain historical preparation evidence. GPU loading, simulator execution and SAC checks now pass as recorded above; full decoder numerical parity with the authors' implementation and held-out capability are not established by these checks.

Metadata provenance remains qualified: actual saved training config `pi05_base_finetune_on_aloha_with_robotwin_clean_randomized_mixed` differs from the model-card example and includes dataset-only repacking `state: actions` followed by `add_state_from_actions`. The exact custom function/training fork was not recovered. Our inference supplies native RoboTwin `joint_action/vector` (commanded joint targets plus normalized grippers), following the standard inference adapter; no additional state or prompt conversion was justified by the available evidence. A saved scene `default_prompt` alone does not prove an always-prepended prefix, and the serialized factory normalization default is overridden for π0.5. Treat this as a **locally validated adapter for the released checkpoint**, with its own matched baselines, not exact reproduction of the authors' protocol.

## Completed stability comparison — November 2, 2026

The matched stability sequence **completed on November 2 at14:57:49 Shanghai time**. Final step-400 policies scored **ICON134/150 (89.33%)** and **Robometer141/150 (94.00%)**; the reused matched SFT20 baseline is **118/150 (78.67%)**. The [completed supervisor status](outputs/chunk20_stable/experiment_status.json) records both full runs and their150-seed evaluations, with no active child. Both rollout ranks loaded the intended final checkpoint with50 predicted/20 executed actions for each policy. [ICON load/result evidence](outputs/chunk20_stable/icon_evaluation.json), [Robometer load/result evidence](outputs/chunk20_stable/robometer_evaluation.json).

The completed sequence was **ICON two-iteration integration check → fresh ICON400 training → ICON150 evaluation → fresh Robometer400 training → Robometer150 evaluation**. It reused the completed SFT20 baseline without repeating SFT evaluation. Each full run started from the original frozen SFT decoder with fresh noise-controller heads and replay; smoke updates were not resumed. The fixed ICON demonstration and both raw reward formulas were preserved. These stable results are separate from the older50-round recipe's ICON131/150 and Robometer138/150. Each recipe has one training seed; aggregate differences do not measure training-seed uncertainty.

| Stable policy | Held-out native success | Training-loop time | Final evaluation time |
| --- | --- | --- | --- |
| ICON | [134/150,89.33%](outputs/icon_dsrl_chunk20_stable_eval_150/metrics.log) | 6:23:13 | 11:04 |
| Robometer | [141/150,94.00%](outputs/robometer_dsrl_chunk20_stable_eval_150/metrics.log) | 5:55:38 | 11:01 |
| Matched SFT20, reused | [118/150,78.67%](outputs/sft_chunk20_eval_150/metrics.log) | No RL | No repeat evaluation |

The two-iteration smoke exited successfully with finite actor/critic losses and nonzero gradients, four completed outcomes,28 valid reward chunks and12 excluded padding chunks. The largest error against raw `progress - 1 + native_success` was approximately `3e-8`. [Update validation](outputs/chunk20_stable/smoke_validation.json), [frame/reward audit](outputs/chunk20_stable/video_reward_validation.json). The original02:10 supervisor stopped after its post-run checker confused environment episode counts with replay-buffer counts; its metadata is archived under `outputs/chunk20_stable/attempt_001_checker_error`. The corrected checker passed13 CPU tests. Relaunch reused this successful smoke evidence, **never its weights or replay**, and started the full ICON run fresh.

| Purpose | Config |
| --- | --- |
| ICON stability training | [robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable](configs/robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable.yaml) |
| ICON stability evaluation | [robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable_eval](configs/robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable_eval.yaml) |
| Matched Robometer stability training | [robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable](configs/robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable.yaml) |
| Matched Robometer stability evaluation | [robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable_eval](configs/robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable_eval.yaml) |

| Stable setting | Meaning and budget |
| --- | --- |
| Collection | Two environments × four rounds = eight episode attempts per iteration;400 iterations =3,200 attempts per reward model. |
| Execution | Predict50 actions, execute20, then observe and replan;200-action episode cap. Allocated action slots include later terminal padding. |
| Actor / critic learning rate | `5e-5` / `1e-4`, reduced from the completed comparison's `1e-4` / `3e-4`. The actor learns diffusion noise; π0.5 stays frozen. |
| Batch | Global128 / micro64 per GPU, with two synchronized actor ranks. Replica counts do not multiply optimizer steps. |
| Critics / sampled | Ten small Q heads share observation encoders; all ten train and their mean is used. Subsample size0 disables subsampling, rather than disabling critics. |
| Optimizer rounds | `update_epoch:20`, `critic_actor_ratio:2`:20 critic updates and10 actor updates per eligible collection iteration. |
| Replay warmup |25 received trajectory batches per rank; each batch contains four episode attempts. Iterations1–24 collect only; the first updates occur at iteration25 after200 total attempts have been collected. |
| Total updates |376 eligible iterations give **7,520 critic /3,760 actor updates**, versus18,800 of each in the completed50-round comparison. |
| Target update | After every critic round, `target = 0.995 × old_target + 0.005 × online`; slowly moving targets stabilize value bootstrapping. |
| Discount | Per-action gamma0.999 becomes `0.999**20 ≈ 0.98019` per executed chunk. |
| Entropy | Fixed alpha0.01 encourages exploration in the actor objective; alpha is not learned. Entropy target/LR settings are inactive, and critic targets omit the entropy bonus. |
| Reward / context | Unnormalized `progress - 1 + native_success`, scale1, eight query frames and terminal-padding filtering. ICON retains the same eight-frame training demonstration; Robometer has no context. |
| Saving / evaluation | Only final step400 is saved. Each final policy uses the same150 scene seeds with native simulator success; no reward-model inference or learning occurs during policy evaluation. |

This is a controlled optimization comparison, not an exact reproduction of the paper. The two training loops took12:18:51 combined, plus22:05 of final evaluation. Supervisor wall time from the corrected launch to completion was approximately12:44. The earlier12–13-hour estimate was consistent with the completed sequence; interaction and rendering dominate, so fewer optimizer updates did not halve runtime.

Inspect the completed sequence and logs:

```bash
cat /home/yifei/icr/outputs/chunk20_stable/experiment_status.json
tail -n 20 /home/yifei/icr/outputs/chunk20_stable/supervisor.log
tail -n 20 /home/yifei/icr/outputs/icon_dsrl_chunk20_stable_s1234.log
```

Full-training logs are `outputs/icon_dsrl_chunk20_stable_s1234.log` and `outputs/robometer_dsrl_chunk20_stable_s1234.log`. Saved final checkpoints are `outputs/icon_dsrl_chunk20_stable_s1234/pi05_dsrl_icon_chunk20_stable/checkpoints/global_step_400` and `outputs/robometer_dsrl_chunk20_stable_s1234/pi05_dsrl_robometer_chunk20_stable/checkpoints/global_step_400`. Final evaluation directories are `outputs/icon_dsrl_chunk20_stable_eval_150` and `outputs/robometer_dsrl_chunk20_stable_eval_150`; each contains `metrics.log`. The [launch record](outputs/chunk20_stable/launch.json), [configuration validation](outputs/chunk20_stable/config_validation.json), and [pinned-source manifest](outputs/chunk20_stable/experiment_manifest.json) record provenance.

### Plotting the training success comparison

Recreate the comparison figure from the completed stability runs:

```bash
cd /home/yifei/icr
rlinf/bin/python scripts/plot_dsrl_success_comparison.py
```

Generated outputs: [PNG](outputs/chunk20_stable/figures/success_rate_comparison.png), [PDF](outputs/chunk20_stable/figures/success_rate_comparison.pdf), [SVG](outputs/chunk20_stable/figures/success_rate_comparison.svg), [CSV](outputs/chunk20_stable/figures/success_rate_comparison.csv), and [provenance/settings JSON](outputs/chunk20_stable/figures/success_rate_comparison.json). The figure shows ICON and Robometer's moving success rates over the latest100 **completed training episodes**, against cumulative episode attempts. Points with fewer than100 completed episodes in the window are omitted:388 plotted points per model span104–3,200 completed episodes. CSV retains all400 logged points per model, including the partial-window flags. The dashed SFT line is its fixed **held-out evaluation** score118/150, provided as context rather than a training curve; final held-out DSRL scores are separately captioned. Final training MA100 values are83%/87%, distinct from held-out89.33%/94%. There is no additional smoothing or training-seed error band. Every CSV value was checked against TensorBoard, and the rendered PNG was inspected. Add `--include-partial-window` to show startup points or `--y-min 0` for a full percentage axis. This command does not train or evaluate a policy.

## Current config names and training metric — November 2, 2026

Config names now describe the reward source and execution variant. The adopted training recipe no longer needs version or size labels. Historical recipes are under `configs/archive/`; the [rename map](outputs/icl_debug/config_rename_map_2026-11-02.json) records exact old-to-new names and unchanged effective settings. Existing experiment directories, checkpoint paths and dated manifests retain their original names.

| Purpose | Config |
| --- | --- |
| Simulator reward, filtered replay,50-action execution | [robotwin_adjust_bottle_pi05_dsrl](configs/robotwin_adjust_bottle_pi05_dsrl.yaml) |
| Simulator final-policy evaluation | [robotwin_adjust_bottle_pi05_dsrl_simulator_eval](configs/robotwin_adjust_bottle_pi05_dsrl_simulator_eval.yaml) |
| ICON,20-action execution | [robotwin_adjust_bottle_pi05_dsrl_icon_chunk20](configs/robotwin_adjust_bottle_pi05_dsrl_icon_chunk20.yaml) |
| ICON stability trial | [robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable](configs/robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable.yaml) |
| ICON stability evaluation | [robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable_eval](configs/robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable_eval.yaml) |
| Robometer,20-action execution | [robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20](configs/robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20.yaml) |
| Robometer final-policy evaluation | [robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_eval](configs/robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_eval.yaml) |
| Robometer stability trial | [robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable](configs/robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable.yaml) |
| Robometer stability evaluation | [robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable_eval](configs/robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable_eval.yaml) |
| Matched SFT evaluation | [robotwin_adjust_bottle_pi05_sft_chunk20_eval](configs/robotwin_adjust_bottle_pi05_sft_chunk20_eval.yaml) |

TensorBoard logs **`env/success_rate_ma100`**, the success mean over the latest100 completed training episodes across all environment ranks. `env/success_rate_ma100_count` exposes the denominator: before100 completions, the mean uses the available episodes. Native success and timeout outcomes count once; post-terminal padding and evaluation episodes are excluded. With eight attempts per iteration, the thirteenth collection fills the100-episode window. Parallel outcomes use deterministic collection/epoch/terminal-chunk/stage/rank/slot order. The driver-local window resets after restart or resume; async runners sort events available in each logging batch.

Both October31 chunk20 TensorBoard directories contain the backfilled metric at all400 logged steps, verified against3,200 saved episode outcomes each. Those earlier runs' final100 training-episode rates are ICON86% and Robometer89%, separate from held-out evaluation and from the new stable runs. [Backfill provenance](outputs/icl_debug/success_rate_ma100_backfill_2026-11-02.json). Original success series were preserved.

Checkpoint cleanup retained one final checkpoint for each of eight useful variations: ICON20, Robometer20, ICON50, simulator50 filtered/unfiltered, Robometer50 dense single-stage, legacy Robometer50 endpoint/delta, and PPO100. Their replay and optimizer state remain. Only the superseded simulator200 pilot, incomplete PPO5 and obsolete Robometer reward-conversion400 checkpoints were deleted; their logs and evaluation records remain. Cleanup recovered74.52GiB, leaving approximately118.43GiB free before the stable pair ran; this is not a current free-space reading. Base SFT/reward models and fixed context data were preserved. [Retention manifest](outputs/icl_debug/checkpoint_retention_2026-11-02.json). Archived simulator-pilot evaluation needs regenerated weights or an explicit retained model override.

The [Robometer paper, Appendix E-2/Table XXII](https://arxiv.org/html/2603.02115v2#A5.SS2) reports DSRL actor/critic LR5e-5/1e-4, batch64, four critics with two sampled, and four critic updates per actor for its single-stage experiment. The completed stability trial matches the learning rates but uses batch128, ten mean-aggregated critics, and a ratio of two. Update counts use different units: the official runner trains per new action chunk after each episode, while RLInf groups eight episode attempts per collection. [Full parameter comparison and code references](outputs/icl_debug/robometer_dsrl_hyperparameters_2026-11-02.json).

## Matched Robometer chunk20 comparison — complete

The corrected Robometer chunk20 run finished on October 31 at 20:34 Shanghai time. Its final step-400 checkpoint scored **138/150 (92.00%)**, compared with **ICON 131/150 (87.33%)** and **matched SFT 118/150 (78.67%)**. Training took 6:05:11 and final evaluation 10:58. Both rollout ranks loaded the intended final checkpoint. [Comparison and load evidence](outputs/robometer_chunk20/matched_comparison.json).

The [training config](configs/robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20.yaml) matches ICON in learner, environment and reward-formula settings: 20 executed/50 predicted actions, 200-action episodes, eight attempts per iteration, 400 iterations, 50 SAC rounds, batch128/micro64, replay readiness25, filtered replay, FP32 heads, actor/critic LR1e-4/3e-4, fixed alpha0.01, gamma0.999 and unnormalized `progress - 1 + native_success`. Service URLs and output paths differ. [Saved configuration comparison](outputs/robometer_chunk20/config_validation.json).

The original Robometer service on GPU1/port8000 loaded the repaired logits converter in its own process and passed its functional regression before model loading. [Loaded-source fingerprint](outputs/robometer_chunk20/server_source_fingerprint.json). ICON uses its separately pinned model on port8001 with the original fixed successful demonstration. No service or context changes were made during the November 2 maintenance.

The [evaluation config](configs/robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_eval.yaml) retains the evaluated checkpoint path under `outputs/robometer_dsrl_chunk20_large_s1234`. [Completed supervisor status](outputs/robometer_chunk20/experiment_status.json) and [evaluation metrics](outputs/robometer_dsrl_chunk20_large_eval_150/metrics.log) remain available. The supervisors refuse to overwrite completed output directories.

## ICON with 20-action execution — October 31 experiment

The user authorized a fresh ICON experiment that predicts50 actions but executes20 before observing and replanning. The200-action limit is retained, giving up to10 real replay transitions per episode. Reward remains **`progress - 1 + native_success` once per executed chunk**, at scale1, with no normalization for the shorter chunk. The existing successful context demo, eight-frame causal queries, camera settings, frozen SFT base and padding filter are retained.

The detached [supervisor](scripts/run_icon_chunk20_experiment.py) runs a two-iteration integration check with actual SAC updates, starts fresh400-iteration training, evaluates its final checkpoint over150 held-out seeds, then evaluates the original SFT policy over the same150 seeds with20-action execution. Both evaluation recipes retain a50-action decoder horizon and200-action episode limit. They use simulator success and do not require ICON inference. The SFT control helps distinguish the effect of shorter execution from learned steering.

**Completed October31 at12:15 Shanghai:** the final ICON policy scored[131/150,87.33%](outputs/icon_dsrl_chunk20_large_eval_150/metrics.log), and the matched original SFT policy scored[118/150,78.67%](outputs/sft_chunk20_eval_150/metrics.log). The difference is13successes, or8.67percentage points, from one training seed. Reported loop times were6:30:08training,10:56updated-policy evaluation and11:50SFT evaluation. [Completed supervisor state](outputs/icon_chunk20/experiment_status.json) records both ranks loading the correct checkpoints with20-action execution. The preceding100-round integration check is retained in [update metrics](outputs/icon_chunk20/smoke_validation.json) and the [video/reward audit](outputs/icon_chunk20/video_reward_validation.json).

| Parameter | Chunk20 experiment |
| --- | --- |
| Training config | [robotwin_adjust_bottle_pi05_dsrl_icon_chunk20.yaml](configs/robotwin_adjust_bottle_pi05_dsrl_icon_chunk20.yaml) |
| Final policy evaluation | [robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_eval.yaml](configs/robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_eval.yaml) |
| Matched SFT evaluation | [robotwin_adjust_bottle_pi05_sft_chunk20_eval.yaml](configs/robotwin_adjust_bottle_pi05_sft_chunk20_eval.yaml) |
| Iterations / collection | 400 /2 environments×4 rounds=8 attempts per iteration;3,200 attempts total |
| SAC rounds | 50 per eligible iteration after readiness25;18,800 actor and critic steps each |
| Global / per-GPU microbatch | 128 /64; one accumulation step on two GPUs |
| Actor / critic learning rates | 1e-4 /3e-4 |
| Alpha / gamma | Fixed0.01 /0.999; per-chunk discount≈0.98019 |
| Replay | Fresh; terminal-padding filtering;512 cached trajectory batches per rank |
| Checkpoint | Final global_step_400 only; separate output directory |
| Video | Training videos with reward progress labels; evaluation video disabled |

Fifty SAC rounds preserve the old recipe's nominal replay reuse:20 rounds×128 samples/32 possible new chunks versus50×128/80. This is a controlled starting point, not a demonstrated optimum. The microbatch increase reduces accumulation overhead without changing the global batch. The completed training loop took6:30:08; final replay and weights occupy about56GiB. Ray used short, experiment-specific paths in `/dev/shm`.

Live state and stage logs:

```bash
cat /home/yifei/icr/outputs/icon_chunk20/experiment_status.json
tail -n 20 /home/yifei/icr/outputs/icon_chunk20/supervisor.log
tail -n 30 /home/yifei/icr/outputs/icon_dsrl_chunk20_large_s1234.log
```

The final checkpoint is `outputs/icon_dsrl_chunk20_large_s1234/pi05_dsrl_icon_chunk20_large/checkpoints/global_step_400`. Final metrics are in `outputs/icon_dsrl_chunk20_large_eval_150/metrics.log`; matching SFT metrics are in `outputs/sft_chunk20_eval_150/metrics.log`. [Configuration validation](outputs/icon_chunk20/config_validation.json) and [launch provenance](outputs/icon_chunk20/launch.json) are retained. The supervisor finished successfully; the completed50-action experiments below use a different execution frequency and update budget.

## Experiment results — updated November 2, 2026

The table lists completed evaluations over the same 150 held-out RoboTwin seeds with raster rendering (`render_shader: default`, denoiser disabled). Training episode attempts can revisit the 1,000 training seeds. Training success and small monitoring evaluations are not the final held-out score.

| Policy / training reward | Budget: iterations / attempts / update rounds | Held-out success | Reported training time |
| --- | --- | --- | --- |
| Original π0.5 SFT | Pretrained; no RL | [122/150, 81.33%](outputs/sft_raster_eval_150/metrics.log) | — |
| Frozen π0.5, zero-noise control | No learning | [122/150, 81.33%](outputs/dsrl_zero_noise_eval_150/metrics.log) | — |
| Original DSRL / simulator success | 200 / 800 / 3,820 | [122/150, 81.33%](outputs/dsrl_sim_raster_eval_150/metrics.log) | 00:32:49 |
| DSRL v2 / simulator success | 200 / 800 / 3,820 | [120/150, 80.00%](outputs/dsrl_sim_v2_alpha001_eval_150/metrics.log) | 00:35:36 |
| DSRL v2 large / simulator success, unfiltered replay | 400 / 3,200 / 7,520 | [129/150, 86.00%](outputs/dsrl_sim_v2_large_eval_150/metrics.log) | 01:50:31 |
| DSRL v2 large / simulator success, filtered replay | 400 / 3,200 / 7,520 | [126/150, 84.00%](outputs/dsrl_sim_v2_large_filtered_eval_150/metrics.log) | 01:51:00 |
| Original DSRL / Robometer, endpoint/delta reward | 200 / 800 / 3,820 | [121/150, 80.67%](outputs/robometer_dsrl_raster_eval_150/metrics.log) | 00:42:17 |
| DSRL v2 large / Robometer, filtered, historical endpoint/delta reward | 400 / 3,200 / 7,520 | [40/150, 26.67%](outputs/robometer_dsrl_v2_large_filtered_eval_150/metrics.log) | 02:04:08 |
| DSRL v2 / Robometer, filtered, dense8 single-stage reward | 100 / 800 / 1,520 | [114/150, 76.00%](outputs/robometer_dsrl_dense8_single_stage_eval_150/metrics.log) | 01:17:42 |
| **DSRL v2 large / ICON, filtered, dense8 single-stage reward** | **400 / 3,200 / 7,520** | [**124/150, 82.67%**](outputs/icon_dsrl_dense8_single_stage_large_eval_150/metrics.log) | **05:20:25** |
| Original π0.5 SFT,20-action execution | Pretrained; no RL | [118/150,78.67%](outputs/sft_chunk20_eval_150/metrics.log) | — |
| **DSRL / ICON,20-action execution, filtered** | **400 /3,200 /18,800** | [**131/150,87.33%**](outputs/icon_dsrl_chunk20_large_eval_150/metrics.log) | **06:30:08** |
| **DSRL / Robometer,20-action execution, filtered** | **400 /3,200 /18,800** | [**138/150,92.00%**](outputs/robometer_dsrl_chunk20_large_eval_150/metrics.log) | **06:05:11** |
| **DSRL / ICON,20-action execution, stable optimization** | **400 /3,200 /7,520 critic +3,760 actor** | [**134/150,89.33%**](outputs/icon_dsrl_chunk20_stable_eval_150/metrics.log) | **06:23:13** |
| **DSRL / Robometer,20-action execution, stable optimization** | **400 /3,200 /7,520 critic +3,760 actor** | [**141/150,94.00%**](outputs/robometer_dsrl_chunk20_stable_eval_150/metrics.log) | **05:55:38** |
| PPO / simulator success, failed recipe | 100 / 800 / 1,000 optimizer steps | [0/150, 0.00%](outputs/ppo_sim_raster_eval_150/metrics.log) | 02:34:55 |

Update rounds are SAC rounds except in the PPO row; synchronous actor replicas do not multiply this count. Times are the reported training-loop elapsed times. The 100-iteration dense Robometer run includes 20 monitoring episodes every 25 iterations; ICON has no intermediate evaluation. Standalone final evaluations take approximately 10–13 minutes and are reported separately from training.

**Current conclusion:** stable20-action optimization scores Robometer141/150 and ICON134/150 against reused SFT118/150. The earlier50-round20-action recipe scored Robometer138/150 and ICON131/150. These are one-training-seed comparisons; per-scene outcomes and repeated training seeds are needed to quantify a reliable method difference. The older50-action experiments use different execution and update budgets.

The historical Robometer results represent different reward recipes. The26.67% run used endpoint/delta scoring before the logits-conversion repair; it is not a corrected control for ICON. Dense8 single-stage Robometer scored76%, but used only100iterations and its live converter revision was not captured: the then-running server predated the source repair, with no recorded restart before that pilot. The new400-iteration/chunk20comparison uses a freshly started service with an in-process repair fingerprint; its final evaluation is92.00%.

V2 uses FP32 controller/critic heads, fixed alpha 0.01 and corrected gradient ownership. The large recipe uses two simulators, four collection rounds per iteration, global batch 128, microbatch 32 per GPU, 20 SAC rounds after replay readiness 25, actor/critic learning rates 1e-4/3e-4, a 200-action horizon, 50-action chunks and five flow steps. The filtered simulator and ICON runs share these learner settings; simulator reward is sparse 0/+1, while learned rewards use `progress - 1 + simulator_success`. Renderer, cameras and held-out evaluation settings match. Detailed current evidence is in the [October 31 inventory](outputs/icl_debug/experiment_inventory_2026-10-31.json).

The latest [ICON checkpoint audit](outputs/icl_debug/icon_final400_audit_2026-10-31.json) confirms 7,520 updates on both actor ranks, finite FP32 controller/critic and optimizer state, matching replicas and changes to initially zero actor-output biases. Training success was 2,608/3,200 attempts (81.5%), rising from 77.5% in the first 100 iterations to 84.5% in the last 100. This verifies that the controller learned; the held-out result remains 82.67%. The audit inspected compact controller states and did not reread the full frozen decoder or replay. The [protocol review](outputs/icl_debug/comparison_protocol_2026-10-31.json) records evaluation alignment and remaining comparison limits.

**Partial and diagnostic runs:** the original Robometer attempt stopped after four warmup iterations and was superseded by the completed 200-iteration run. Earlier PPO restarts and the scaled official-based PPO run are not completed benchmarks: the latter recorded 11/1,000 iterations, checkpoint 10 and a 7/8 monitoring result, without a full 150-seed evaluation. Short rendering, resume and two-iteration integration checks verify functionality and are excluded from the result table. The [October 29 inventory](outputs/experiment_summary_2026-10-29.json) records older attempts; some old checkpoint/audit files have since been removed during cleanup.

## Recommended next experiments

1. **Compare the completed matched stability runs with their training trajectories.** The sequence at the top of this README finished both final150-seed evaluations. Plot the100-episode training success windows separately from held-out evaluation scores; the lower learning rates and lower actor/critic update counts changed together, so this comparison does not isolate a single hyperparameter.
2. **Inspect the exact100-episode success window alongside critic diagnostics.** The metric counts completed episodes across ranks, rather than averaging100 iterations. Absolute Q and actor-loss levels are not accuracy measures; assess critic-error/gradient trends together with native success.
3. **Record per-scene evaluation outcomes and repeat promising recipes across training seeds.** Compare settings at the same data budget and preserve a separate final test set after development decisions. The completed Robometer/ICON/SFT aggregate results do not measure training-seed variability.


The launch recipes below document completed and prepared experiments. Use new output names for repeats; a completed result does not imply its old checkpoint is still retained. Current summaries and saved evaluation logs take precedence over historical preparation notes.

```text
RoboTwin observations → learned 32-dimensional noise policy → frozen π0.5 → 50 robot actions
          ↓                                                        ↓
  reset + real action frames ←──── next observation / true terminal frame
          ↓
  sample eight prefix frames → Robometer / ICON progress + simulator success base → replay → SAC
```

The policy sees the head and two wrist cameras; both reward estimators see RGB head-camera images. ICON additionally receives its fixed reference. The default DSRL reward mode is `single_stage`: it scores up to eight uniformly sampled frames from the whole growing episode history, including the reset image and latest real action frame. Intermediate action images are captured with `env.train.skip_intermediate_renders: false`; action masks exclude terminal padding. The server receives sampled frames, not every captured frame. No separate reset-only score request is needed in this mode.

The default reward is `progress - 1` for unfinished or unsuccessful chunks and `progress` for the chunk that RoboTwin marks successful. Robometer success probabilities never trigger termination or change the base reward. One scalar occupies the final slot of each 50-action chunk and SAC sums it once. The simulator's sparse reward has mixing weight zero; native simulator success is explicitly used to select the -1/0 base reward. Episode contexts reset independently and post-terminal padding receives zero reward and no HTTP request. Explicit `progress_delta` and `progress` modes remain available; the PPO config retains its existing explicit delta mode. Invalid/out-of-range model responses stop the run.

## Environments and installed assets

| Environment | Runtime | Purpose |
| --- | --- | --- |
| `rlinf/` | Python 3.11, Torch 2.7.1 + CUDA 12.8 | RoboTwin, π0.5, SAC/PPO, lightweight HTTP adapter |
| `robometer/` | Python 3.10, Torch 2.8.0 + CUDA 12.8, Transformers 4.57.6 | Official Robometer inference server |
| `.venv-icon/` | Python 3.10, Torch 2.8.0 + CUDA 12.8, Transformers 4.57.6 | ICON inference with a fixed successful reference |

All three environments already exist on this machine. Keep them separate. The SFT policy is in `checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle`; the verified Robometer inference bundle is in `checkpoints/Robometer-4B-inference`. Simulator assets are installed, with random backgrounds disabled. Current configs set `render_denoiser: none`.

DSRL and PPO training and evaluation also set `task_config.single_thread_render: true`. Scene creation, reset, camera reads, cache clearing, and teardown use one persistent thread. This requires one environment per environment worker: keep `total_num_envs: 2` with the two-GPU placement.

The workstation's ray-tracing path previously stalled during full-motion video rendering after the thread change. The configs therefore use `task_config.render_shader: default`, SAPIEN's raster renderer, and disable ray tracing before the first native cache or scene access. Use the same renderer for SFT, DSRL, and PPO comparisons with either reward, and report the renderer with results because raster images differ from the original ray-traced images. The denoiser remains disabled in the experiment configs and `clear_cache_freq` remains one.

On 2026-10-28, at the user's request, SAPIEN's bundled OIDN was repaired from 2.0.1 to 2.3.3, which supports Blackwell. The complete matching native library set is under `rlinf/lib/python3.11/site-packages/sapien/oidn_library/lib`; `_oidn_tricks.py` loads the new core/API paths. Torch, NCCL, the driver and SAPIEN's Python version were unchanged. The old library directory and loader are backed up under `data/oidn/backups/20261028T063837Z`; [installation manifest](data/oidn/installation.json) records source URL, hashes, paths and verification. This installed-package patch must be reapplied if SAPIEN is reinstalled. To roll back with RLinf stopped, move the current `oidn_library` aside and restore both `oidn_library` and `_oidn_tricks.py` from that backup.

The OIDN repair passed actual ray tracing and denoising on both physical GPUs: three 320×240 cameras, 32 samples per pixel, path depth eight, repeated scene creation, and 129 total camera captures. GPU selection must match CUDA's current device; RLinf's per-worker GPU isolation was tested, as was explicit CUDA selection for GPU 1 with both GPUs visible. The original verification report was removed during cleanup; the installation manifest above remains available. Reusable diagnostic: `scripts/check_sapien_oidn.py`. These small checks do not establish long-run RoboTwin ray-tracing stability or make 256 concurrent environments fit. Current training/evaluation settings remain raster to preserve the existing comparison protocol. Any explicit ray-tracing trial needs a fresh process with `SAPIEN_DISABLE_RAY_TRACING` unset, `render_shader: rt`, and `render_denoiser: oidn`.

To reconstruct the reward environment using uv and the checked-in inference requirements:

```bash
cd /home/yifei/icr
uv venv --python 3.10 robometer
uv pip install --python robometer/bin/python torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python robometer/bin/python -r requirements/robometer-inference.txt
uv pip install --python robometer/bin/python --no-deps -e third_party/robometer
uv pip install --python rlinf/bin/python --no-deps -e .
```

`requirements/robometer-inference.txt` omits training, dataset-upload, and baseline-only dependencies. The machine's resolved versions are recorded in `data/robometer/requirements-inference.lock.txt`. Preserve the existing RLinf environment and local source patches; its installation follows the [official RLinf guide](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html).

## ICON context-conditioned reward

ICON is integrated using [robometer-v2 `full_training`](https://github.com/sNiper-Qian/robometer-v2/tree/b68c27ef7df11cdebdfef995c78644c3cb33cdf3) and [ICON-4B](https://huggingface.co/sniper-qian/ICON-4B/tree/e75b5fca997ab18f620ecb1931e5d9acba7c1f9b). The source and checkpoint are pinned to these revisions. Its environment is `.venv-icon/`; the original `robometer/` environment and inference API remain available. The inference bundle `checkpoints/ICON-4B-inference` preserves all trained conditioning settings and uses standard Transformers with the existing local Qwen3 backbone.

**Completed October 30 at 23:32 Shanghai time:** 400 training iterations in 5:20:25, followed by a 9:47 evaluation with **124/150 successes (82.67%)**. The supervisor exited with `stage: complete`; its status is retained in `outputs/icon_reward/experiment_status.json`. The launch artifact is `outputs/icon_reward/run_large_experiment.py`. A reward server can remain running after the experiment finishes; check `/health` before starting another service.

```bash
cat /home/yifei/icr/outputs/icon_reward/experiment_status.json
tail -n 40 /home/yifei/icr/outputs/icon_dsrl_dense8_single_stage_large_eval_150/metrics.log
```

The fixed reference was collected by `scripts/collect_icon_context.py` using the original SFT policy, with no optimizer or DSRL controller. Training seed 0 timed out; seed 1 succeeded after 127 real actions. `data/context_demos/adjust_bottle/demo_000/frames.npy` contains 128 raw head-camera RGB frames from reset through first native success; `metadata.json` records both seeds, native outcome, camera, checkpoint/code provenance and file hashes. Eight deterministic chronological context indices are cached by the server. Held-out evaluation seeds were not used. An [MP4 preview](data/context_demos/adjust_bottle/demo_000/context_demo.mp4) contains the same 128 frames at 30 FPS (4.27 seconds); reward inference continues to use raw `frames.npy`.

**Minimal interface change:** the ICON service attaches this fixed demonstration internally. The RL worker still sends the same query-only `/evaluate_batch_npy` requests as Robometer. Demo frames never enter policy observations, query history, reward aggregation or replay. Starting ICON uses the same server module with one additional argument, `context_demo_path`. The service uses GPU 1, port 8001; use the command below when no service is already listening there:

```bash
cd /home/yifei/icr/third_party/robometer_v2
env -u LD_PRELOAD -u NCCL_P2P_DISABLE CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/home/yifei/icr/data/runtime/matplotlib \
  PYTHONPATH=/home/yifei/icr/third_party/robometer_v2 \
  /home/yifei/icr/.venv-icon/bin/python -m robometer.evals.eval_server \
  model_path=/home/yifei/icr/checkpoints/ICON-4B-inference \
  context_demo_path=/home/yifei/icr/data/context_demos/adjust_bottle/demo_000 \
  num_gpus=1 max_workers=1 server_url=127.0.0.1 server_port=8001
```

`use_frame_steps=true` expands an input trajectory into separate growing-prefix scoring examples and collects their endpoint predictions. The RL bridge already constructs a causal growing prefix, so it sends `use_frame_steps=false`: one forward pass scores the supplied query frames. Context progress tokens are excluded from the returned query scores. Both code paths now preserve context. Multipart context serialization and the inherited logits-to-progress conversion are repaired; the HTTP client additionally requires one returned score per query frame. `/health` exposes the actual model path, context configuration, reference hash, seed and sampled indices.

| Comparison | Configuration | Status |
| --- | --- | --- |
| Simulator success, filtered | `robotwin_adjust_bottle_pi05_dsrl` | 126/150, 84.00% |
| Robometer, filtered, dense8 single-stage, 100 iterations | `archive/robotwin_adjust_bottle_pi05_dsrl_robometer_recovery` | Completed: 114/150, 76.00%; training 1:17:42 |
| Robometer, filtered, dense8 single-stage, 400 iterations | `robotwin_adjust_bottle_pi05_dsrl_robometer` | Prepared; equal-budget control remains to be run |
| ICON, filtered, dense8 single-stage, 400 iterations | `robotwin_adjust_bottle_pi05_dsrl_icon` | Completed: 124/150, 82.67%; training 5:20:25 |

The three 400-iteration configurations retain two environments and four rounds per iteration: eight episode attempts per iteration and 3,200 total. Replay readiness is 25 received batches per rank, followed by 20 SAC rounds per eligible iteration, for 7,520 rounds total. Global batch is 128 and microbatch is 32 per GPU; actor/critic learning rates are 1e-4/3e-4, fixed alpha is 0.01, and controller/critic heads use FP32. Actor seed is 1234, environment seed is 0, horizon is 200 actions, action chunks contain 50 actions, and flow uses five steps. All start fresh from SFT with terminal-padding filtering.

ICON and corrected Robometer resolved configs differ only in output names and server URL. Their reward remains `progress - 1 + simulator_success`; model success probabilities never trigger termination. The simulator control uses sparse 0/+1 rewards and retains cheaper endpoint rendering with no training video. The historical 40/150 Robometer result used older endpoint/delta rewards and buggy conversion, so it is not the matched corrected baseline.

Validation passed 116 CPU tests and actual GPU loading/scoring. A two-iteration check collected 16 independent training episodes, performed four SAC rounds with finite losses and nonzero actor/critic gradients, and validated all 16 progress-video sidecars and padding exclusions. Reference-self scoring and a static-reset control verified the transport protocol; these are not independent reward-accuracy measurements. Evidence: [integration report](outputs/icl_debug/icon_integration_validation.json), [configuration comparison](outputs/icl_debug/icon_dsrl_config_validation.json), [offline protocol scores](outputs/icl_debug/icon_offline_scoring.json), [experiment manifest](outputs/icon_reward/experiment_manifest.json). The completed run took 5:20:25, followed by a 9:47 evaluation. Post-warmup ICON iterations averaged about 48.9 seconds in the inspected window; dense Robometer training averaged 45.9 seconds excluding monitoring evaluations. The older endpoint-only Robometer run averaged 18.8 seconds. Environment interaction rose from 2.4 to 26.0 seconds with dense camera capture; SAC updates remained about 2.5–2.8 seconds. These timings include the shared workstation workload. Ray temporary storage is scoped under `/dev/shm/icr-ray-icon-large-*` to avoid the root filesystem's 95% warning.

The final checkpoint is `outputs/icon_dsrl_dense8_single_stage_large_s1234/pi05_dsrl_icon_dense8_single_stage_large/checkpoints/global_step_400`. The supervisor successfully evaluated it with native simulator success and the same 150 held-out seeds, writing `outputs/icon_dsrl_dense8_single_stage_large_eval_150.log` and that directory's `metrics.log`. Policy-only evaluation does not use ICON or the demo. To reproduce training from the standard RLinf terminal exports below, use `--config-name robotwin_adjust_bottle_pi05_dsrl_icon`; start fresh and choose new output names when reproducing an existing experiment.

## RLinf terminal

Use these exports for training and evaluation. The isolated NCCL 2.31.2 library plus disabled peer-to-peer transport is the configuration verified on this two-GPU Blackwell machine.

```bash
source /home/yifei/icr/rlinf/bin/activate
cd /home/yifei/icr/third_party/rlinf
export REPO_PATH="$PWD"
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export ROBOTWIN_PATH=/home/yifei/icr/third_party/robotwin
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="/home/yifei/icr/src:$REPO_PATH:$ROBOTWIN_PATH:${PYTHONPATH:-}"
export RLINF_EXT_MODULE=icl_reward
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
export NCCL_P2P_DISABLE=1
export LD_PRELOAD=/home/yifei/icr/data/nccl/2.31.2/nvidia/nccl/lib/libnccl.so.2
```

## 1. Test the revised DSRL recipe with simulator rewards

**Filtered simulator-reward comparison:** edit `configs/robotwin_adjust_bottle_pi05_dsrl.yaml`. Its main settings are now explicit and commented in that file, so changing this experiment does not require editing the shared parent used by other runs. It matches the completed 86% large simulator recipe, with post-terminal replay filtering enabled and separate output names. Reward remains sparse simulator +1 on first success, zero otherwise; Robometer and its single-stage reward are disabled. The filter preserves the first terminal chunk/reward and excludes subsequent chunks from SAC minibatches; raw stored trajectories retain padding.

| What to change | Keys in the filtered config | Current value |
| --- | --- | --- |
| Training budget and saving | `runner.max_epochs`, `runner.max_steps`, `runner.save_interval` | 400 / 400 / 400 |
| Collection per iteration | `env.train.rollout_epoch` | 4 rounds × 2 environments = 8 episode attempts |
| Batch sizes | `actor.global_batch_size`, `actor.micro_batch_size` | 128 global / 32 per GPU microbatch |
| SAC rounds per iteration | `algorithm.update_epoch` | 20 after replay warmup |
| Actor / critic learning rate | `actor.optim.lr`, `actor.critic_optim.lr` | 1e-4 / 3e-4 |
| Fixed entropy coefficient | `algorithm.entropy_tuning.initial_alpha` | 0.01 |
| Replay readiness | `algorithm.replay_buffer.min_buffer_size`, `algorithm.train_actor_steps` | 25 / 25 received trajectory batches per rank |
| Training-time evaluation | `runner.val_check_interval`, `env.eval.rollout_epoch` | Disabled (-1); one round means 2 monitoring episodes |
| Output names | `runner.logger.log_path`, `runner.logger.experiment_name` | Separate filtered experiment |
| Replay padding filter | `algorithm.replay_buffer.filter_terminal_padding` | true |

Keep two environments for the current two single-threaded workers; increase `rollout_epoch` for more data. With two actor ranks, the global batch must be divisible by twice the microbatch. Change both training limits together and update the save interval/checkpoint path if the final iteration changes. The 200-action episode/rollout/task limit and 50-action chunk are exposed together; retain them for this filtering comparison. If enabling training-time evaluation, `env.eval.rollout_epoch: 10` gives 20 monitoring episodes. Standalone evaluation below always uses 150 episodes.

From the existing RLinf terminal with the exports above:

```bash
cd /home/yifei/icr/third_party/rlinf
set -o pipefail
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name robotwin_adjust_bottle_pi05_dsrl \
  2>&1 | tee /home/yifei/icr/outputs/dsrl_sim_v2_large_filtered_s1234.log
```

The prepared run collects 3,200 episode attempts and performs 7,520 SAC rounds after warmup, starting fresh from SFT. No reward server is needed. Evaluate its final checkpoint with:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_pilot_eval \
  rollout.model.model_path=/home/yifei/icr/outputs/dsrl_sim_v2_large_filtered_s1234/pi05_dsrl_v2_large_filtered/checkpoints/global_step_400 \
  runner.logger.log_path=/home/yifei/icr/outputs/dsrl_sim_v2_large_filtered_eval_150 \
  2>&1 | tee /home/yifei/icr/outputs/dsrl_sim_v2_large_filtered_eval_150.log
```

CPU Hydra composition confirmed that filtering and output names were the only resolved differences from the unfiltered large recipe. The user subsequently completed this run in 1:51:00 and evaluated the final checkpoint: [126/150, 84.00%](outputs/dsrl_sim_v2_large_filtered_eval_150/metrics.log).

The previous v2 pilot and unfiltered large recipes below remain historical references.

Use the RLinf terminal exports above. The v2 pilot starts fresh from SFT, trains only DSRL's small controller/critic, and saves to a new directory. It needs no Robometer server. Keep two actor/environment ranks and the existing raster renderer.

| Setting | Previous completed run | v2 pilot |
| --- | --- | --- |
| Entropy coefficient | Automatic, starts 1.0, ends about 0.437 | Fixed 0.01 |
| Controller/critic parameters, Adam moments, head computation | BF16 | FP32 |
| Gradient clipping | Included other-optimizer gradients | Active optimizer gradients only |
| Collection iterations / episode attempts | 200 / 800 | 200 / 800 |
| SAC updates after replay warmup | 20 per eligible iteration; 3,820 total | Same |
| Actor / critic LR | 1e-4 / 3e-4 | Same |
| Global batch / microbatch per GPU | 16 / 4 | Same |
| Episode horizon / action chunk / denoising steps | 200 / 50 / 5 | Same |
| Discount / target EMA | 0.999 / 0.005 | Same |
| Checkpoints | Iterations 100 and 200 | Final iteration 200 only |

The [official DSRL example](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html) uses LIBERO step penalties with initial alpha 1.0. This task instead gives sparse +1 success rewards. The old run's entropy was near the bounded 32-dimensional maximum, and its deterministic controller outputs stayed close to zero. A smaller fixed coefficient tests a different reward–entropy balance. Fixed 0.01 is a task-specific hypothesis, not an official default or a demonstrated improvement. Other learning hyperparameters remain fixed to keep this pilot interpretable. To attribute an effect specifically to entropy, use the optional repair-only control below, which shares both optimizer fixes.

First run this two-iteration GPU smoke check. The replay overrides deliberately enable actual updates immediately; a two-iteration run with the normal ten-batch warmup would test no learning. Expect four SAC updates total, finite actor/critic losses and nonzero gradients, and alpha 0.01. This saves no checkpoint or video:

```bash
set -o pipefail
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_pilot \
  runner.max_epochs=2 runner.max_steps=2 runner.save_interval=-1 \
  algorithm.update_epoch=2 \
  algorithm.replay_buffer.min_buffer_size=1 algorithm.train_actor_steps=1 \
  runner.logger.log_path=/home/yifei/icr/outputs/dsrl_sim_v2_smoke \
  2>&1 | tee /home/yifei/icr/outputs/dsrl_sim_v2_smoke.log
```

Then measure the zero-noise decoder control on the same 150 held-out seeds. This uses original SFT weights and supplies zero initial noise, bypassing the controller; it is not a trained DSRL policy. The existing SFT baseline uses random initial noise, while learned DSRL evaluation normally uses its mean. This control helps distinguish a benefit from learned steering from a benefit caused by the evaluation noise convention:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_pilot_eval \
  rollout.model.model_path=/home/yifei/icr/checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
  rollout.model.openpi.dsrl_eval_noise_mode=zero \
  runner.logger.log_path=/home/yifei/icr/outputs/dsrl_zero_noise_eval_150 \
  2>&1 | tee /home/yifei/icr/outputs/dsrl_zero_noise_eval_150.log
```

After the smoke check passes, run the pilot from SFT with fresh replay and optimizers:

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_pilot \
  2>&1 | tee /home/yifei/icr/outputs/dsrl_sim_v2_alpha001_s1234.log
```

Expect roughly 35–60 minutes for training, based on the old 32.8-minute run; FP32 head computation has not been profiled on this machine. Extrapolate from the first 20 iterations. Replay warms for nine iterations; iterations 10–200 perform 20 SAC updates each. Each iteration collects two environments × two rounds = four episode attempts. `train_actor_steps: 10` is another replay-readiness threshold, not ten critic-only optimizer steps. The final checkpoint is `outputs/dsrl_sim_v2_alpha001_s1234/pi05_dsrl_v2/checkpoints/global_step_200`. Old checkpoints and old recipe files are preserved; do not resume an old BF16 optimizer into this fresh comparison.

Evaluate that checkpoint with its learned deterministic controller:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_pilot_eval \
  2>&1 | tee /home/yifei/icr/outputs/dsrl_sim_v2_alpha001_eval_150.log
```

Each 150-seed evaluation should take about 10–15 minutes. Videos are disabled in the new evaluation config; enable `env.eval.video_cfg.save_video=true` if needed. It retains the old frame-capture schedule, seeds, renderer, normalization, horizon and action chunk. Compare the result with both SFT's 122/150 and the zero-noise control. A one- or two-success difference is weak evidence; confirm a promising gain with another training seed before claiming improvement.

If the mean-policy score remains flat, evaluate the same checkpoint's stochastic controller before spending more on training:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_pilot_eval \
  rollout.model.openpi.dsrl_eval_noise_mode=sample \
  runner.logger.log_path=/home/yifei/icr/outputs/dsrl_sim_v2_sample_eval_150
```

For an attribution control, train `--config-name archive/robotwin_adjust_bottle_pi05_dsrl_entropy_repair`. It uses the same corrected FP32 heads/clipping and 800-attempt budget, but retains the old automatic alpha starting at 1.0. Evaluate with the v2 eval config, overriding the model path to `outputs/dsrl_sim_v2_repair_only_s1234/pi05_dsrl_v2/checkpoints/global_step_200` and using a separate output directory. The fixed-alpha pilot alone cannot isolate optimizer fixes from entropy changes.

**Larger-budget simulator test completed: 129/150 (86.00%).** Training took 1:50:31 with 400 iterations and 7,520 configured SAC rounds. This is seven more successes than SFT/zero noise in one training run. The earlier replay audit recorded 2,549/12,800 chunks (19.9%) as post-success padding; the subsequent filter excludes those chunks from SAC sampling. The filtered run later scored 126/150 (84%). These aggregate scores alone do not establish a filtering effect. The original detailed audit/figure was removed during cleanup; the [training log](outputs/dsrl_sim_v2_large_s1234/metrics.log) and [held-out evaluation](outputs/dsrl_sim_v2_large_eval_150/metrics.log) remain.

**Recorded larger experience and batch recipe.** The prepared config is `configs/archive/robotwin_adjust_bottle_pi05_dsrl_simulator_unfiltered.yaml`. It retains v2's model, fixed alpha 0.01, learning rates, chunk size and 20 SAC rounds per eligible iteration. It increases to 400 iterations, two simulators × four sequential rounds = eight episode attempts per iteration, 3,200 attempts total, global batch 128, microbatch 32 per GPU, replay/actor readiness 25, and replay cache/window 512 slots per rank. Training begins at iteration 25, after 200 collected attempts; 376 eligible iterations × 20 = 7,520 SAC rounds. The stored rollout budget is 12,800 chunk slots, including terminal padding. Nominal replay sample draws increase from 61,120 to 962,560; this is about four times the sample reuse per collected slot. Do not also double SAC rounds for this test.

There are 1,000 configured training reset seeds and 150 disjoint held-out seeds. The larger run revisits the training set; 3,200 attempts does not mean 3,200 distinct resets. Keep two parallel simulators because the verified single-thread rendering path requires one simulator per worker. Increase collection through `rollout_epoch`, not simulator concurrency. The endpoint-only large simulator run subsequently completed in 1:50:31, and its filtered counterpart in 1:51:00. Dense learned-reward runs capture far more images and take longer; their measured timings are in the summary above. Save only the final checkpoint, estimated around 30 GiB including replay.

First check the larger GPU batch with two short iterations, four SAC rounds, and no checkpoint:

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_unfiltered \
  runner.max_epochs=2 runner.max_steps=2 runner.save_interval=-1 \
  algorithm.update_epoch=2 \
  algorithm.replay_buffer.min_buffer_size=1 algorithm.train_actor_steps=1 \
  runner.logger.log_path=/home/yifei/icr/outputs/dsrl_sim_v2_large_smoke
```

After this check passes, start the larger run fresh from SFT:

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_unfiltered \
  2>&1 | tee /home/yifei/icr/outputs/dsrl_sim_v2_large_s1234.log
```

Evaluate its final checkpoint with the existing FP32-head evaluation config:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_pilot_eval \
  rollout.model.model_path=/home/yifei/icr/outputs/dsrl_sim_v2_large_s1234/pi05_dsrl_v2_large/checkpoints/global_step_400 \
  runner.logger.log_path=/home/yifei/icr/outputs/dsrl_sim_v2_large_eval_150
```

Main/smoke/evaluation configs and FSDP policies were validated on CPU before the user completed the larger GPU run and evaluation above. Historical configuration validation was recorded before this completed run; resolved configuration (removed during cleanup). A flat held-out score after this run should trigger a decoder noise-sensitivity check before further increases in training budget.

The legacy short integration recipe remains available:

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_legacy_simulator
```

This legacy pilot runs 20 collection iterations, two environments with two collection rounds, global batch 16, microbatch 4, and two SAC updates per eligible iteration. Evaluation and checkpoint saving occur every five iterations. It retains BF16 heads and automatic alpha; use v2 for the new learning experiment. Shared legacy settings are in `configs/archive/_robotwin_adjust_bottle_pi05_dsrl_legacy_base.yaml`; v2 overrides are in `configs/archive/_robotwin_adjust_bottle_pi05_dsrl_pilot_base.yaml`.

Outputs go to `outputs/dsrl_adjust_bottle`. Checkpoints contain full inference weights plus compact trainable SAC state, optimizer state, target-network state, and replay data. Continue a saved run by adding `runner.resume_dir=/absolute/path/to/checkpoints/global_step_5`; retain the same model architecture and two actor ranks.

## 2. Start the Robometer server in a separate terminal

The October 30 reward-conversion fix requires restarting an already running server. Stop that foreground process with Ctrl+C, then use the command below. Editing the Python source does not update the function already imported by a running server.

Check whether the existing server is available before starting another:

```bash
curl -sSf http://127.0.0.1:8000/health
```

The reward server uses GPU 1 at `http://127.0.0.1:8000`; use the health check to confirm its current state. A foreground server prints logs in its terminal. Logs from the original integration check are in `outputs/robometer/server.log`. To start it in the foreground when no server is running:

```bash
cd /home/yifei/icr/third_party/robometer
env -u LD_PRELOAD -u NCCL_P2P_DISABLE CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/home/yifei/icr/third_party/robometer \
  /home/yifei/icr/robometer/bin/python -m robometer.evals.eval_server \
  model_path=/home/yifei/icr/checkpoints/Robometer-4B-inference \
  num_gpus=1 max_workers=1 server_url=127.0.0.1 server_port=8000
```

Stop this foreground service with Ctrl+C after reward training or scoring finishes. Training shares GPU 1 with the reward service; it does not start or stop the service automatically.

Score an unannotated rollout video from the RLinf terminal:

```bash
python -m icl_reward.score_video \
  /home/yifei/icr/outputs/dsrl_verified_reload/video/eval/seed_0/0.mp4 \
  --task 'Pick the bottle with white neck and red top up with the right arm ensuring it stays head-up.' \
  --chunk-frames 50 --max-frames 8 --trim-end-frames 1 \
  --output /home/yifei/icr/outputs/robometer_offline/success_scores.json
```

The JSON records causal progress, progress differences, optional success probabilities, and request latency. `--trim-end-frames` removes known recorder padding; use the actual padding count for each input. Progress scores are the training signal; success probabilities are diagnostic.

## 3. Train DSRL with Robometer rewards

**Current defaults, October 30:** denser growing histories with eight sampled images, plus the paper-inspired single-stage reward using **simulator success only**. The official single-stage design uses predicted progress plus a -1/0 base; this adaptation retains RoboTwin's success and timeout logic instead of the paper's reward-model success threshold. [Paper, Sections III-D and Appendix E-2](https://arxiv.org/html/2603.02115v2).

All real post-action head-camera frames are retained in the episode's temporary reward history. Each scoring request uniformly samples at most eight images from that history, always including reset and current images. If the first 50 actions all execute, the sampled observation indices are `[0,7,14,21,28,35,42,50]`; after 100 actions they are `[0,14,28,42,57,71,85,100]`. An episode ending early retains only actual executed frames; masks remove repeated terminal padding. Dense history is not added to actor observations or replay storage. The model sees a causal episode prefix, never future frames.

At default scale one, one reward is assigned per 50-action decision chunk:

| Simulator outcome | Reward |
| --- | --- |
| Still running | `progress - 1`, within [-1,0] |
| Simulator success | `progress`, within [0,1] |
| Timeout without success | `progress - 1`, within [-1,0] |
| Padding after the episode ended | Zero; no scoring request |

There is no Robometer success threshold. `reward.model.reward_mode: single_stage` is the default in all Robometer DSRL entry configs. `env_reward_weight: 0` prevents separately adding the simulator's sparse +1; the adapter already uses the explicit native success boolean to switch the base from -1 to zero. The scalar is stored once at slot49, so it is not multiplied by50. SAC timeout bootstrapping remains unchanged. Set `reward.model.reward_mode=progress_delta` to select the previous difference reward; this still uses dense histories unless endpoint-only rendering is explicitly requested. Do not combine old replay with this new reward convention.

**Optional scoring frequency:** both Robometer and ICON accept `reward.model.score_interval=25` to score actions25 and50 within each50-action chunk. The default is50;10 scores at10/20/30/40/50. Each point uses its own causal growing-history prefix with at most eight query frames. Requests with matching shapes can be batched, so scoring twice does not necessarily issue two HTTP requests. Full intermediate rendering must remain enabled. The actual final action is always scored once: success at action37 produces scores at25 and37, while success at7 produces only one score. Padding is never scored. Scores are placed at their corresponding action slots, and videos/score JSONs mark each scored frame.

`reward.model.score_aggregation=mean` is the default. In `single_stage`, the chunk total is `reward_scale * (mean(sampled_progress) - 1 + native_success)`: shaping is divided by the number of actual scores, and the native success bonus is added once at the final real endpoint without division. Thus two scores change which progress estimates inform training without automatically doubling the shaping magnitude. `sum` instead gives `reward_scale * (sum(sampled_progress - 1) + native_success)` and changes the reward/entropy balance. In `progress` mode the corresponding mean/sum is applied to progress itself. `progress_delta` always sums sequential differences; they telescope to the existing endpoint-minus-baseline reward, so additional scoring alone does not change its chunk target.

For a matched two-score experiment, add the following overrides to either learned-reward training command and select a fresh output directory:

```bash
reward.model.score_interval=25 reward.model.score_aggregation=mean
```

This still creates one DSRL noise decision and one replay transition per50-action chunk. SAC sums the populated reward slots without discounting inside a chunk, then discounts the next chunk's value by `gamma**50`. Current v2/large configs use `gamma=0.999` (chunk discount≈0.9512) and fixed `alpha=0.01`. Alpha weights the actor's entropy term in `alpha * log_probability - Q`; it encourages exploration and is not a discount factor. More frequent reward scoring adds model work but no additional SAC updates or policy replanning. Use fresh replay when changing this reward convention; the default50 configuration reproduces existing runs.

Training videos are enabled at30FPS. Footers show the latest estimated progress, diagnostic delta and actual training reward; the first reset is marked unscored because this mode does not make a reset-only request. Each MP4 has a score JSON with frame counts, native success/termination/truncation and exact endpoint indices. For success after action7 of a50-action chunk, the endpoint label occurs at frame7 and later copies are marked unscored padding. Labels are drawn on saved copies and never enter model inputs. Full rendering adds work; the old endpoint-only timing estimate no longer applies. Measure the integration check before planning a full run.

Use the existing RLinf environment/exports and the corrected Robometer server from section2. Restart an old server if it has not imported the logits-conversion repair. First run two iterations to verify the dense path and optimizer update:

```bash
cd /home/yifei/icr/third_party/rlinf
set -o pipefail
export RLINF_EXT_MODULE=icl_reward
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_robometer_recovery \
  runner.max_epochs=2 runner.max_steps=2 runner.save_interval=-1 \
  runner.val_check_interval=-1 algorithm.update_epoch=2 \
  algorithm.replay_buffer.min_buffer_size=1 algorithm.train_actor_steps=1 \
  runner.logger.log_path=/home/yifei/icr/outputs/robometer_dsrl_dense8_single_stage_smoke \
  2>&1 | tee /home/yifei/icr/outputs/robometer_dsrl_dense8_single_stage_smoke.log
```

Expect four SAC rounds, finite losses, alpha0.01, signed rewards following the table, and labeled full-motion videos plus JSONs under the smoke output's `video/train/`. This check saves no checkpoint. The shared reward/collection path subsequently passed the ICON GPU smoke, and the completed Robometer 100-iteration run verifies its full training/evaluation path.

**Completed dense Robometer pilot (live conversion repair unverified):** 100 iterations, 800 training attempts and 1,520 SAC rounds; batch 128/microbatch 32, replay readiness 25, fixed alpha 0.01 and the existing learning rates. It took 1:17:42 including 20-episode monitoring every 25 iterations, then scored [114/150, 76.00%](outputs/robometer_dsrl_dense8_single_stage_eval_150/metrics.log). The command below records the recipe; use new output names for a repeat. It starts from SFT with empty replay:

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_robometer_recovery \
  2>&1 | tee /home/yifei/icr/outputs/robometer_dsrl_dense8_single_stage_s1234.log
```

New outputs are separate from historical delta runs: `outputs/robometer_dsrl_dense8_single_stage_s1234`. The checkpoint is `pi05_dsrl_dense8_single_stage/checkpoints/global_step_100`; videos/score JSONs are under `video/train/`. Full150-seed evaluation requires no reward server:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_pilot_eval \
  rollout.model.model_path=/home/yifei/icr/outputs/robometer_dsrl_dense8_single_stage_s1234/pi05_dsrl_dense8_single_stage/checkpoints/global_step_100 \
  runner.logger.log_path=/home/yifei/icr/outputs/robometer_dsrl_dense8_single_stage_eval_150 \
  2>&1 | tee /home/yifei/icr/outputs/robometer_dsrl_dense8_single_stage_eval_150.log
```

The full-budget option is `--config-name robotwin_adjust_bottle_pi05_dsrl_robometer`:400iterations /3,200attempts /7,520SAC rounds, the same batches/learning rates/alpha, and final400only. Its new output is `outputs/robometer_dsrl_dense8_single_stage_large_s1234/pi05_dsrl_dense8_single_stage_large/checkpoints/global_step_400`. Run the shorter checks before selecting this budget. The shorter original entry config also uses the new defaults but retains its legacy optimizer/model recipe; prefer the recovery or v2-large configs above.

The filtered simulator control remains `robotwin_adjust_bottle_pi05_dsrl`. Its original capture mode is endpoint-only; explicitly set `env.train.skip_intermediate_renders=false` for a matched rendering-workload comparison. The previously reported86% simulator score used unfiltered replay, so it does not isolate the filtering effect.

Historical audit: the previous endpoint/delta Robometer400 checkpoint scored40/150(26.67%). Replay had12first-chunk rewards4.04–5.82,60.94%of squared raw reward, and poor late success alignment(AUC0.508). The definite logits-conversion bug is repaired, HTTP progress is constrained to[0,1], and terminal padding filtering was verified. The newer dense8 single-stage 100-iteration pilot scored 76%; changing conversion, frame sampling and reward design together does not isolate any one change. The earlier interpretation is retained in the [development handoff](REWARD_MODEL_BASELINES_HANDOFF.md); its detailed audit files were removed during cleanup.

The latest shared implementation passed **116 focused CPU tests**, actual ICON GPU scoring and a two-iteration DSRL integration run with real updates. The completed Robometer pilot and ICON large run both have full held-out evaluations. See the [integration report](outputs/icl_debug/icon_integration_validation.json) and [configuration comparison](outputs/icl_debug/icon_dsrl_config_validation.json). Earlier preparation reports were removed during cleanup.

## 4. Evaluate a saved DSRL checkpoint over 150 seeds

The final ICON checkpoint has already been evaluated automatically. To repeat that evaluation manually, use the RLinf terminal exports above and a new output directory:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name archive/robotwin_adjust_bottle_pi05_dsrl_simulator_pilot_eval \
  rollout.model.model_path=/home/yifei/icr/outputs/icon_dsrl_dense8_single_stage_large_s1234/pi05_dsrl_icon_dense8_single_stage_large/checkpoints/global_step_400 \
  runner.logger.log_path=/home/yifei/icr/outputs/icon_dsrl_dense8_single_stage_large_eval_150_repeat
```

This evaluates two environments × 75 rounds, performs no learning and needs no reward server or reference demo. The learned controller uses deterministic mean noise; the frozen π0.5 decoder and native simulator success define the policy test. Videos are disabled in this config; add `env.eval.video_cfg.save_video=true` to record them.

For another retained v2 checkpoint, change `rollout.model.model_path` and the output directory. The saved evaluation configs/logs identify which checkpoint was used for each completed result. Older checkpoint paths may no longer exist after cleanup; check the current experiment inventory before attempting a rerun.

## 5. Compare PPO with simulator and Robometer rewards

**Status, 2026-10-29:** the simulator PPO run below completed 100 iterations, but its final held-out evaluation scored **0/150**, versus **122/150 (81.33%)** for SFT. DSRL + simulator reward scored **122/150 (81.33%)**, and DSRL + Robometer scored **121/150 (80.67%)**, with the matching raster protocol. The PPO commands below record the failed recipe; revise it before another long run. Forward-only, two-GPU checks with synchronized weights found all 40 SFT chunks and all 10 final-PPO chunks had probability ratios exactly 1.0, including native FSDP replay. They also confirmed a separate critic-value discrepancy with `value_after_vlm: false` (maximum absolute gap 0.14852 for SFT, 0.02192 for final PPO). The proposed prefix critic and gentler update schedule have not been applied or trained. The separate October 28 scaled official-based PPO checkpoint 10 has only an eight-episode monitoring result, not a formal evaluation. These parity findings are historical records; their original debug artifacts were removed during cleanup. The completed policy evaluation remains available above.

The two PPO recipes start independently from the original SFT checkpoint. They use the same environment, policy, seed and optimization settings; only the reward source and output names differ. Unlike DSRL, PPO updates the π0.5 action expert and value head while keeping the vision/language trunk frozen. Use the RLinf terminal exports above and run GPU jobs sequentially.

| Setting | Both PPO recipes |
| --- | --- |
| Outer collection iterations | 100 |
| Parallel environments / collection rounds | 2 / 4 |
| Episode attempts per iteration / total | 8 / 800 |
| Episode action limit / action chunk | 200 / 50 |
| Stored chunk slots per iteration | 32, including terminal padding when episodes end early |
| PPO epochs / nominal optimizer steps per iteration | 5 / 10 |
| Global batch / batch per GPU / microbatch per GPU | 16 / 8 / 1 |
| Action expert / value learning rates | `5e-6` / `1e-4` |
| Discount / GAE lambda / PPO clip | 0.99 / 0.95 / 0.2 |
| Checkpoint saving | Final iteration 100 only |
| Training videos / automatic held-out evaluation | Disabled / disabled |

Training uses `auto_reset: false` and `ignore_terminations: false`, so PPO masks terminal padding in its losses. The PPO configurations and reward path have CPU validation. The combined PPO + Robometer GPU run has not yet been verified; run this two-iteration check with the reward server healthy before committing to the longer learned-reward run. It saves no checkpoint:

```bash
set -o pipefail
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name robotwin_adjust_bottle_pi05_ppo_robometer \
  runner.max_epochs=2 runner.max_steps=2 \
  runner.save_interval=-1 runner.val_check_interval=-1 \
  runner.logger.log_path=/home/yifei/icr/outputs/ppo_robometer_smoke_raster \
  2>&1 | tee /home/yifei/icr/outputs/ppo_robometer_smoke_raster.log
```

For the simulator-reward baseline, the Robometer service is unnecessary and can be stopped to free GPU memory:

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name robotwin_adjust_bottle_pi05_ppo \
  2>&1 | tee /home/yifei/icr/outputs/ppo_sim_raster_s1234.log
```

Start the Robometer service using section 2 and confirm `/health`, then train the separate learned-reward policy from SFT:

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name robotwin_adjust_bottle_pi05_ppo_robometer \
  2>&1 | tee /home/yifei/icr/outputs/ppo_robometer_raster_s1234.log
```

The learned-reward run uses the same causal progress differences as DSRL, with simulator-reward weight zero and Robometer-reward weight one. PPO retains its GAE/value-head objective; no SAC replay settings apply. Both runs save full policy checkpoints. Previous PPO checkpoints occupy approximately 29 GiB each; allow at least 60 GiB for both final checkpoints, plus video headroom. The requested cleanup of obsolete smoke checkpoints and DSRL checkpoint 100 recovered 68.2 GiB, leaving about 87.9 GiB free. Final DSRL checkpoint 200, the latest earlier PPO checkpoint 5, and evaluation outputs are retained. Check free space again before long runs because the disk is shared.

After training, stop the reward service if desired and evaluate each checkpoint sequentially on the same 150 held-out seeds:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name robotwin_adjust_bottle_pi05_full_eval \
  rollout.model.model_path=/home/yifei/icr/outputs/ppo_sim_raster_s1234/pi05_ppo/checkpoints/global_step_100 \
  runner.logger.experiment_name=pi05_ppo_sim_eval_150 \
  runner.logger.log_path=/home/yifei/icr/outputs/ppo_sim_raster_eval_150 \
  2>&1 | tee /home/yifei/icr/outputs/ppo_sim_raster_eval_150.log

python evaluations/eval_embodied_agent.py \
  --config-path /home/yifei/icr/configs \
  --config-name robotwin_adjust_bottle_pi05_full_eval \
  rollout.model.model_path=/home/yifei/icr/outputs/ppo_robometer_raster_s1234/pi05_ppo_robometer/checkpoints/global_step_100 \
  runner.logger.experiment_name=pi05_ppo_robometer_eval_150 \
  runner.logger.log_path=/home/yifei/icr/outputs/ppo_robometer_raster_eval_150 \
  2>&1 | tee /home/yifei/icr/outputs/ppo_robometer_raster_eval_150.log
```

These evaluations perform no learning, require no reward server, use the matching raster renderer, and save unannotated full-motion videos at 20 fps. Compare final `success_once` over 150 episodes; the training success metric uses collected training episodes and is not the held-out result.

## Validation recorded on this machine

The latest shared integration passed **116 CPU tests**: 45 progress-conversion cases, 36 HTTP/reward cases, 20 ICON fixed-context/server cases, 10 raw-demo collection cases and five rendering regressions. The [test log](outputs/icl_debug/icon_cpu_tests.log) and [integration report](outputs/icl_debug/icon_integration_validation.json) are retained.

Actual ICON GPU loading and query scoring passed. A two-iteration DSRL check collected 16 training episodes, performed four configured SAC rounds, produced finite losses/nonzero actor and critic gradients, and validated all 16 video-score sidecars. Camera masks excluded intra-chunk padding from reward histories, retained true terminal frames and suppressed post-terminal score requests. This smoke saved no checkpoint.

The context reference was collected without optimization from a training seed, saved as raw RGB with a native success flag and hashed provenance, and supplied separately from query frames. Scoring that reference itself or a static-reset control is a protocol check, not independent evidence of reward accuracy. The completed 100-iteration Robometer and 400-iteration ICON experiments provide the current full-policy results.

Older rendering/resume/PPO/DSRL audits are described in the development handoff and October 29 inventory. Cleanup removed several old debug directories and test checkpoints; the current inventory explicitly records checkpoint availability. Missing artifacts must not be presented as fresh verification.
