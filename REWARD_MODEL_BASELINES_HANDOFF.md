# Reward Model Baselines — Local Development Handoff

Prepared: 2026-10-26
Updated: 2026-11-02 (ICON context KV cache validated; original Motus queue stopped; fresh cached continuation launched)

## Current work — ICON context KV cache and fresh Motus continuation

- User requested fixed-demonstration KV reuse, before/after equivalence tests, and continuation with the optimization. They subsequently explicitly requested stopping the current training and deleting its partial training output before restarting after tests. Root stopped original supervisor1281557, training child1465170 and owned ICON8002 service1457066. `outputs/motus_multitask/experiment_status.json` is now stopped and archived; **do not resume or rewrite the old manifest/status**. The incomplete `adjust_bottle/icon_train` directory (943 files,61,624,320 allocated bytes) was deleted and had no checkpoint. Its diagnostic log, all successful demos, old smoke evidence and all completed baselines remain. [Stop record](outputs/icon_kv_cache_validation/stop_request.json), [deletion receipt](outputs/icon_kv_cache_validation/partial_training_deletion.json).
- All three frozen-Motus150-seed baselines completed and are independently revalidated: **adjust_bottle48/150 (32%), move_can_pot21/150 (14%), click_bell65/150 (43.33%)**. Records remain `outputs/motus_multitask/<task>/vanilla_evaluation.json`, including actual saved configuration, exact TensorBoard episode counts and both rollout ranks loading the intended H32/execute20 Motus checkpoint. No reward estimator participates in vanilla evaluation, so these completed baselines remain valid after introducing ICON caching. The older task-specific SFT118/150 is a different model.
- Implementation is `third_party/robometer_v2/robometer/models/context_kv_cache.py` (`FixedContextKVCache`) with minimal optional dispatch in `models/rbm.py`. Actual ICON checkpoint uses the default RBM wrapping Qwen3-VL, **not** the alternative `qwen_vision_t5_causal` architecture. A reusable causal prefix includes exact task language, eight fixed context images, context progress tokens and the separator before the first query image. Each server GPU replica holds its own bounded LRU:8 entries/1GiB, approximately120MiB per prefix. Keys hash processed context pixels, token IDs, image grids and positions; model weights/configuration/device/autocast changes invalidate reuse. Query history KV is never retained across requests. Temporary request caches are isolated; differing prompt lengths and left/right padding are batched with explicit masks. Full input multimodal RoPE positions are computed before splitting, and context/query DeepStack features are preserved.
- Server flags: `context_kv_cache=true`, `context_kv_cache_max_entries=8`, `context_kv_cache_max_mib=1024`. Caching activates only with a validated fixed demo; original Robometer remains unchanged. `context_kv_cache=false` restores the previous monolithic inference route. `/health` and `/model_info` expose `context_kv_cache` with per-replica `hits`, `misses`, `entries`, `bytes`, `evictions`, `invalidations`, `fallbacks` and reasons. Server replicas are serialized by the existing GPU pool, with an additional cache lock for direct callers; health snapshots do not block behind inference.
- **Numerical honesty:** ordinary BF16 monolithic and cached split inference are mathematically equivalent but not bitwise identical, because different sequence/batch shapes change floating-point kernels. Final production15-case suite passed declared bounds across all three tasks: progress maximum absolute difference **0.00359106**, on-device BF16 success-probability difference **0.009765625**, raw-logit difference **0.16796875**. Acceptance tolerances were progress0.005, success probability0.01, logits0.2 +1% relative. Initial stricter BF16 failures remain archived. Separate real-checkpoint FP3215-case proof passed absolute/relative raw1e-4/1e-5, progress1e-5 and success probability5e-5: observed maxima were progress **3.7551e-6**, success probability **2.2501e-5**, raw progress logits **7.2718e-5**, raw success logits **1.0139e-4**. The first cached and repeated cached outputs were bitwise equal for all tested heads/cases. Do not claim universal bitwise parity with the old BF16 route. Native simulator success still exclusively determines terminal success; single-stage reward uses progress, not the estimator's success head. [Native proof](outputs/icon_kv_cache_validation/checkpoint_native_validated.json), [FP32 proof](outputs/icon_kv_cache_validation/checkpoint_fp32_validated.json), [deployment/source-hash summary](outputs/icon_kv_cache_validation/validation_summary.json).
- Production keeps BF16. Warm reward-model forward speedup was **1.404–1.921x, median1.788x**; estimated unchanged collation plus forward speedup **1.324–1.644x, median1.568x**. These are measured reward-inference timings on real stored RGB, not a total training-speed guarantee. Tests:15 real tiny-Qwen3-VL CPU cache tests;57 existing server/context regression tests; actual GPU suites15 native +15 FP32 cases. Tests and benchmark are `tests/test_icon_context_kv_cache.py` and `scripts/benchmark_icon_context_kv_cache.py`; use `.venv-icon` for installed Transformers4.57.6, without environment changes. FP32 serves as verification, not the deployed precision, because its absolute inference latency is higher.
- **Fresh continuation launched November2 at19:20:23Shanghai, supervisorPID1605823:** `scripts/continue_motus_with_icon_kv_cache.py` uses `outputs/motus_multitask_kv_cache`, a symlink to `/media/yifei/workspace_nvme/icr_motus/outputs/motus_multitask_kv_cache`. The stopped original tree is immutable. Real cached bottle SAC smoke passed; fresh ICON400 training started around19:23Shanghai, trainerPID1615084, owned cached servicePID1605871. Both rollout ranks loaded the correct H32/execute20 Motus policy. First full collection completed at step1/400:8 trajectories,63.477seconds,finite mean reward-model output(-.86572),MA100count8. Replay is still in the configured readiness25 warmup; this is not yet a full-run optimizer update. Real actor/critic updates were verified by the separate cached smoke. Host preflight verified180 pins and2,048,783,552,512 available NVMe bytes. Continuation requires actual native/FP3215-case evidence with all tasks represented, tested cache/RBM/server/config SHA hashes, revalidated baselines, unchanged remaining source/model/context pins, healthy Robometer8000, free owned ICON8002 and NVMe/shared-memory capacity. It retains450 evaluated vanilla episodes, then runs18 queued stages: per task ICON smoke → fresh ICON400 → ICON150 → Robometer smoke → fresh Robometer400 → Robometer150. Six fresh task/method controllers, replay buffers and optimizer states; no aborted training is resumed. Only owned children/ICON8002 are stopped. Existing8000/8001 services and unrelated jobs remain untouched. The [real cached smoke proof](outputs/motus_multitask_kv_cache/adjust_bottle/icon_smoke_validation.json) records finite actor losses(-.03357,.17097),critic losses(.58494,.33983),nonzero actorgrad(.10378,.08929) and criticgrad(3.51168,2.72397),MA100 counts2/4,20 reward requests,36 prefixhits/4 misses (90% row hit rate),4 entries/500,754,432bytes,zero fallback/eviction. Smoke heads/replay are not resumed.
- Optimization, frozen Motus, train/eval seeds, context demos, horizons and raw rewards are unchanged: H32/execute20, five flow steps;400 collections×8 attempts;20 critic/10 actor rounds after warmup25;global128/micro64;actor/critic LR5e-5/1e-4;alpha.01,gamma.999,tau.005,FP32 heads;filtered terminal padding;reward `progress-1+native_success` once per chunk. Full final weights/optimizer/replay saving and training videos are retained. New checkpoint paths: `outputs/motus_multitask_kv_cache/<task>/<method>_train/pi05_motus_<task>_dsrl_<method>/checkpoints/global_step_400`. Evaluations explicitly override the old config-default checkpoint path to that new task/method checkpoint.
- Monitor `outputs/motus_multitask_kv_cache/experiment_status.json`, `supervisor.log`, and `<task>/<method>_train.log`; final comparison will be `summary.json`, with `baseline_reuse.json` recording inherited results. Use a revised provisional2–3-day budget for all six runs/evaluations: earlier initial Motus bottle throughput was approximately70seconds/collection, and `click_bell` doubles the horizon. The earlier35–50-hour estimate was optimistic; the cache accelerates only reward inference. Fresh-run throughput remains to be measured. Do not rerun the old supervisor or a baseline, overwrite pinned demos, append to pinned validation reports/logs, or edit pinned source/config/model files while the continuation is running. README/handoff updates are not pinned.

## Latest storage and integrity maintenance — November 2, 2026

- User authorized identifying valid versus corrupted runs and deleting corrupted/obsolete artifacts. Cleanup completed at17:27:03Shanghai. **No structurally corrupted checkpoint was found.** All136 active Motus source/config/model/context pins were unchanged;19 owned Python files parsed with zero syntax errors. Historical audit checked6,246 PyTorch ZIP archives from the ten retained comparison checkpoints with zero central-directory/member-extent errors. This was structural validation, not a full tensor load or full CRC scan. Retention evidence: [recommendations](outputs/cleanup_audit/historical_recommendations.json), [archive inventory](outputs/cleanup_audit/historical_checkpoints_inventory.json).
- All ten distinct comparison checkpoints were kept, together with completed evaluations/TensorBoards, the valid interrupted official PPO step10 checkpoint, the referenced stable checker archive, and all active Motus assets. Failed or interrupted status alone does not make a saved checkpoint corrupted; low success is not corruption. At that cleanup, active Motus vanilla evaluations continued without restart. The later user-requested stop/restart is documented above. New checkpoints/outputs use the external NVMe, which had enough space throughout the audit.
- The two stable final checkpoint trees were relocated from `outputs/icon_dsrl_chunk20_stable_s1234/pi05_dsrl_icon_chunk20_stable/checkpoints` and `outputs/robometer_dsrl_chunk20_stable_s1234/pi05_dsrl_robometer_chunk20_stable/checkpoints` to `/media/yifei/workspace_nvme/icr_retained/outputs/` with identical relative paths. All1,618 files were fully SHA256-verified after copying; only then were original trees removed and their old paths replaced by symlinks. Existing evaluator paths remain valid, including full weights, optimizer and replay state. Relocation reclaimed117,101,080,576bytes (109.06GiB).
- Deleted32 audited obsolete failed-startup/incomplete-run/temp artifacts (507,805,696allocatedbytes, approximately484MiB), including23 inactive historical `/tmp/ray` sessions and an obsolete task-owned review clone. Active Motus uses `/dev/shm`; no live session was removed. Preserved `/tmp/icr-icon-metadata` and `/tmp/robotwin-uv-bootstrap` because they remain referenced. Exact deletions, SHA manifests and storage proof: [completed cleanup receipt](outputs/cleanup_audit/cleanup_2026-11-02.json).
- Root free space increased from8.10GiB to117.63GiB; external NVMe remained approximately1.86TiB free. These are completion-time measurements. Source/config/model/context files and active queue metadata were not modified by cleanup. For current work, monitor `outputs/motus_multitask_kv_cache/experiment_status.json` before launching anything; the original Motus status is stopped.

## Original Motus preparation and retained baseline provenance

- User explicitly selected a released RoboTwin SFT policy, rather than raw `pi05_base`, and authorized downloading/adapting it, small GPU checks, new vanilla baselines and matched ICON/Robometer DSRL runs in the background. **All preparation gates passed; original queue launched2026-11-02 at16:57:07Shanghai and later stopped at the user's request.** SupervisorPID1281557 is no longer active. Completed frozen-Motus baselines are **adjust_bottle48/150 (32%),move_can_pot21/150 (14%),click_bell65/150 (43.33%)**, independently verified from saved config, exact TensorBoard counts and both rollout-rank checkpoint loads. Results: `outputs/motus_multitask/<task>/vanilla_evaluation.json`. The aborted initial ICON training was deleted; fresh cached continuation is documented above. Host preflight verified136 pins,21 stages and approximately2.16TB free. Complete model SHA, three-task GPU loading/simulator checks, successful training-only reference collection and the isolated ICON/SAC update check are verified. Do not report the old bottle baseline118/150 as the new Motus baseline. Completed historical results remain below.
- Historical original evidence: `outputs/motus_multitask/experiment_status.json` is stopped; `supervisor.log` and the three vanilla logs remain preserved. The live continuation uses `outputs/motus_multitask_kv_cache/experiment_status.json` and writes its final `summary.json` there. Do not restart the stopped original queue or duplicate continuation stages.
- Selected model: `motus-robotics/pi0.5_robotwin2`, pinned revision `effc6e1a99bd4d0edf9447e8ba313def67649cfe`. This is the released π0.5 baseline, not `Motus_robotwin2`'s different world/action architecture. The model card reports50-task joint SFT using2,500 clean +25,000 randomized demonstrations. Downloaded `metadata.pt` was inspected with zipfile/pickletools without unpickling; it confirms π0.5/discrete state,gemma_2b/gemma_300m,32 internal dimensions,**32 prediction steps**,200 tokens,`adapt_to_pi:true`,and delta arm joints with absolute grippers. Existing RLInf checkpoint conversion and Aloha transforms match those declared settings; no vendor runtime/checkpoint change was needed. Exact author inference protocol remains unresolved in `https://github.com/thu-ml/Motus/issues/45`; do not claim official benchmark reproduction.
- Tasks: `adjust_bottle`, `move_can_pot`, `click_bell`; all use Aloha-AgileX and clean scenes. New per-task episode/rollout/native limits are respectively **200/200/200**, **200/200/200**, **400/400/400**. Predict32/execute20 gives at most10,10,20 real chunks per episode. All three limits change together. `move_can_pot`'s upstream Piper default is explicitly overridden. Runtime five flow steps are our choice, not established by the model release.
- Remaining full stages: reuse the completed frozen Motus150-seed baselines for all three tasks, then for each task run fresh cached ICON400 → final150 evaluation → fresh Robometer400 → final150 evaluation. The same frozen Motus weights serve all tasks, while each task/method receives fresh controller heads, optimizer and replay; the integration check is not resumed. This is six task-specific DSRL policies on a shared multi-task decoder, not jointly trained controller weights. Nine full evaluations total1,350 held-out episode attempts, excluding preliminary checks. Final policy evaluation uses native success and needs no reward service or learning.
- Three-task preparation evidence is `outputs/motus_prepare/launch_preconditions.json`, statuspassed. Four-episode simulator smoke counts: **adjust_bottle1/4,move_can_pot0/4,click_bell2/4**; these validate integration and are not capability estimates. Both rollout ranks loaded the intended H32/execute20/five-flow-step checkpoint on each task. Frozen Motus produced successful native training-only demos: **adjust_bottle actualseed1/119actions,move_can_pot100000/118actions,click_bell100001/120actions**, with zero optimizer steps. Each demo is reset→firstsuccess raw head RGB with hashes/provenance under `data/context_demos/motus/<task>/demo_000/metadata.json`.
- Isolated bottle ICON/SAC integration completed two iterations/four episode attempts, exit0. Both actor losses(-.04752,.15147),critic losses(.61663,.25321),actor gradient norms(.10620,.09193),and critic gradient norms(3.60777,1.88853) were finite; gradients nonzero. MA100 counts were2 then4. Evidence `outputs/motus_prepare/sac_adapter_smoke_validation.json` and `_status.json`; no checkpoint was saved and full training does not resume this smoke. Owned simulator/Ray/ICON processes were stopped, cleanup errors empty. This confirms the actual update path, not policy improvement.
- Stable optimization is retained:400 collections×8 attempts=3,200 training attempts/run;25 replay-batch readiness;20 critic and10 actor updates per eligible collection (7,520/3,760 total);global128/micro64;actor/critic LR5e-5/1e-4;fixed alpha.01,gamma.999,tau.005,FP32 heads,terminal-padding filtering. Reward remains raw `progress-1+native_success` once per real20-action chunk,scale1; no density normalization. Dense causal query histories sample up to8 frames. Training video remains enabled, evaluation video disabled. Only final400 is saved with full decoder/optimizer/replay state. The earlier provisional35–50h estimate was not measured Motus runtime; the current cached restart uses a revised2–3-day estimate above, pending actual throughput, especially for400-action `click_bell`.
- Public config family: `robotwin_<task>_motus_pi05_sft_eval`, `robotwin_<task>_motus_pi05_dsrl_icon`, `..._icon_eval`, `..._robometer`, `..._robometer_eval`. Shared parents: `configs/_robotwin_motus_pi05_dsrl.yaml`, `_robotwin_motus_pi05_eval.yaml`, `_robotwin_motus_pi05_dsrl_eval.yaml`. New output names isolate all prior experiments. Prepared supervisor: `scripts/run_motus_multitask_experiment.py`; it requires `outputs/motus_prepare/launch_preconditions.json` with passed smoke/context evidence before starting, verifies frozen model/normalizer/service/context identities and final checkpoint loads, and owns only its child jobs plus its task-specific ICON service on8002. Do not start duplicate jobs; check live status/processes before any launch.
- `scripts/collect_icon_context.py` is now task/configurable: `--task`, `--checkpoint`, `--norm-stats`, `--action-horizon`, `--action-chunk`, `--max-actions`, `--flow-steps` plus existing attempt/seed/output controls. Motus collection explicitly used `--action-horizon 32 --action-chunk 20 --flow-steps 5`, own checkpoint/stats, and `--max-actions 200` or400. All three reference destinations `data/context_demos/motus/<task>/demo_000` now exist and are fixed for this comparison; do not overwrite them. They retain raw dense head RGB reset→first native success, training-only requested/actual seed membership and file/code/model hashes. No old bottle context is substituted. The old ICON service on8001 and original fixed demonstration remain separate; temporary8002 SAC-smoke service was stopped.
- Storage: `checkpoints/Motus-Pi05-RoboTwin2` is a symlink to `/media/yifei/workspace_nvme/icr_motus/checkpoints/Motus-Pi05-RoboTwin2`; `outputs/motus_multitask` points to `/media/yifei/workspace_nvme/icr_motus/outputs/motus_multitask`. Dedicated volume had approximately2TiB free at preparation. Model and full new checkpoints/replay/videos go there, avoiding the root filesystem's remaining~8GiB at preparation time (later cleanup reclaimed109GiB). No checkpoint stripping, policy-only format, user-file cleanup or environment update was performed. Sandbox writes through these symlinks require host/escalated tool access; CPU preparation reports stay under root `outputs/motus_prepare`.
- Logical task output tree: `outputs/motus_multitask/<task>/{vanilla_eval,icon_train,icon_eval,robometer_train,robometer_eval}`. Each final learned policy is `<task>/<method>_train/pi05_motus_<task>_dsrl_<method>/checkpoints/global_step_400`; evaluators point to their own task/method's final checkpoint. Frozen weights: `checkpoints/Motus-Pi05-RoboTwin2/model.safetensors`; normalization: `assets/pi0.5_clean_randomize_joint_training/norm_stats.json` under that checkpoint. Keep the downloaded Motus normalizer, rather than the old task-specific bottle SFT statistics.
- Download **complete and SHA verified**: original single HTTP stream failed with `IncompleteRead`; resumable parallel8MiB byte-range parts subsequently completed. `outputs/motus_prepare/download.json` records `complete:true` at2026-11-02T08:44:42Z. Model size7,473,091,464bytes and full SHA256`de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb` match the pinned LFS identity. Piece map `outputs/motus_prepare/download_parts.json`, log `outputs/motus_prepare_download_parallel.log`. Verified norm SHA256`1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb`; metadata SHA256`ef52d518167a652b74cc1c4d3a49fc18f758c286fc0ee8485d33a3caf3bd99ee`.
- Completed CPU evidence: seven adapter-audit tests; final `outputs/motus_prepare/adapter_cpu_audit.json` verifies full SHA plus actual812-source→667-canonical key/shape conversion against the real π0.5 meta architecture with H32. It also verifies the actual Motus normalizer and batched RLInf input/state tokens/output joint+gripper mapping against installed upstream OpenPI, exactly zero input/output differences. Earlier `checkpoint_header_audit.json` remains partial-header-only evidence; `transform_cpu_parity.json` retains the initial transform report. Small GPU, context and controller-update gates now pass as above; full held-out performance remains pending.
- Preprocessing provenance caveat: actual saved training config is `pi05_base_finetune_on_aloha_with_robotwin_clean_randomized_mixed`, differing from the public model-card example; dataset `repack_transforms.inputs` maps `state:actions` then references `openpi.training.config.add_state_from_actions`. Exact custom function/source fork was not recovered. Standard OpenPI applies dataset repacking only during training; our inference follows the native adapter's `joint_action/vector`, which contains joint drive targets plus normalized grippers. No confirmed previous-action/state mismatch was found. Saved scene `default_prompt` alone does not prove mandatory prefixing; standard injection only fills missing prompts. Serialized factory `use_quantile_norm:false` is overridden for π0.5. Keep present transforms; describe this as a **locally validated adapter for the released checkpoint**, not exact author-protocol or full-decoder numerical parity. Four-episode smoke outcomes do not justify speculative preprocessing changes.

## Latest completed experiment — November 2, 2026

- The authorized matched stability sequence **completed November2 at14:57:49 Shanghai** (`2026-11-02T06:57:49Z`). `outputs/chunk20_stable/experiment_status.json` records `stage:complete`, all five stages completed, last child returncode0 and childPIDnull. Final400 results: **ICON134/150 (89.33%) /Robometer141/150 (94%) /reused matched SFT118/150 (78.67%)**. Both rollout ranks loaded the intended step400 checkpoint with50 predicted/20 executed actions for each policy; records `outputs/chunk20_stable/{icon_evaluation.json,robometer_evaluation.json}`. These are one-training-seed results, not evidence of training-seed robustness.
- Historical launch: corrected supervisor3736105 started02:13:34Shanghai; initial ICON driver3736106. The completed sequence was ICON two-iteration integration check → fresh ICON400 training → ICON150 evaluation → fresh Robometer400 training → Robometer150 evaluation. Supervisor: `scripts/run_chunk20_stable_experiments.py`. SFT20 was not rerun. Full runs started fresh from SFT/controller/replay and did not resume smoke updates or old checkpoints.
- Training-loop times: ICON6:23:13,Robometer5:55:38; final evaluations11:04/11:01. Combined training12:18:51 plus22:05evaluation; approximately12:44 corrected-supervisor walltime. The prior12–13h estimate was consistent with completion. Training/evaluation artifacts and configurations remain separate from the older50-round20-action recipe's ICON131/150 and Robometer138/150.
- Plot code and artifacts are complete: `scripts/plot_dsrl_success_comparison.py`, run fromworkspace root with `rlinf/bin/python scripts/plot_dsrl_success_comparison.py`. Generated files: `outputs/chunk20_stable/figures/success_rate_comparison.{png,pdf,svg,csv,json}`. Curve: MA100 over completed TRAIN episodes against cumulative episode attempts, omitting points withwindowcount<100;388 plotted points per model span104–3200episodes. CSV retains all400 points/model with partial-window flags. Dashed SFT is held-out118/150 context, not a training curve; final held-out DSRL results appear separately in the caption. Final training MA100 is ICON83%/Robometer87%, distinct from held-out89.33%/94%. One training seed, no error bands or additional smoothing. Independent audit checked all800 CSV rows against exact TensorBoard values, evaluation records and source hashes; rendered PNG inspected. `--include-partial-window` includes startup, `--y-min 0` shows the full percentage axis. No policy training/evaluation was run to generate the figure.
- Actual two-iteration smoke passed: finite actor/critic losses,nonzero actor/critic gradients,four completed outcomes,28 valid chunks and12 excluded terminal-padding chunks. Maximum raw-reward formula error `2.9802322387695312e-8`. Evidence: `outputs/chunk20_stable/{smoke_validation.json,video_reward_validation.json}`. Counts derive from exact TensorBoard namespaces and actual video sidecars; no smoke checkpoint was saved.
- Historical first attempt: supervisor3725392 at02:10 ran smoke child3725393 successfully(exit0), then its checker stopped the sequence after broad `num_trajectories` matching included replay-buffer counts. Metadata is archived in `outputs/chunk20_stable/attempt_001_checker_error`. The corrected checker passed13 CPU tests. Current launch uses `--reuse-integration-check` to reuse validation evidence only; full ICON controller/replay start fresh from SFT. No smoke weights/replay or previous400 checkpoint are resumed.
- Training configs: `robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable` and `robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable`; evaluations add `_eval`. Both share actor/critic LR5e-5/1e-4,global128/micro64,20 critic rounds/10 actor updates per eligible collection,ten mean-aggregated Q heads with no subsampling,tau.005 every critic round,gamma.999 per action (`gamma**20≈.98019`),fixed alpha.01,and25 replay-batch readiness. π0.5 stays frozen; the actor learns latent noise, not VLA weights.
- Each collection receives one replay batch per rank, containing four episode attempts. Iterations1–24 collect without optimization; the first updates occur at iteration25 after200 total attempts have been collected. Iterations25–400 give376 eligible collections: **7,520 critic /3,760 actor updates** per run.400 collections×8 attempts=3,200 attempts,200-action cap,20 executed/50 predicted actions. Synchronization over two actor replicas does not double optimizer counts.
- Reward/context settings remain eight-frame causal query histories,unchanged eight-frame successful training demo for ICON,`progress-1+native_success` once per valid chunk,scale1,no density normalization,terminal-padding filtering,and native simulator termination. Fixed alpha.01 means entropy target/LR are inactive; critic targets omit entropy backup. This is a stability trial, not an exact paper reproduction or a demonstrated gain.
- Authoritative monitoring artifacts: `outputs/chunk20_stable/{experiment_status.json,supervisor.log,launch.json,config_validation.json,experiment_manifest.json}`. Smoke log: `outputs/icon_dsrl_chunk20_stable_smoke.log`. Full logs: `outputs/icon_dsrl_chunk20_stable_s1234.log` and `outputs/robometer_dsrl_chunk20_stable_s1234.log`. Training TensorBoards/video/metrics use the corresponding directories without `.log`.
- Saved final checkpoints: `outputs/icon_dsrl_chunk20_stable_s1234/pi05_dsrl_icon_chunk20_stable/checkpoints/global_step_400` and `outputs/robometer_dsrl_chunk20_stable_s1234/pi05_dsrl_robometer_chunk20_stable/checkpoints/global_step_400`. Final-only saving. Completed evaluation directories: `outputs/icon_dsrl_chunk20_stable_eval_150` and `outputs/robometer_dsrl_chunk20_stable_eval_150`; policy-only evaluation reported simulator successes over the same150 seeds, with no reward-model inference or optimizer updates.
- Storage at launch was approximately118GiB free; each reference checkpoint occupied54.53GiB. Supervisor checked free space at stage boundaries. That is historical capacity evidence, not current free-space status. Do not remove retained user checkpoints without authorization. Completed source/config/model/context pins are preserved in the experiment manifest.
- Previous50-round20-action comparison remains **Robometer138/150 (92%) /ICON131/150 (87.33%) /SFT118/150 (78.67%)**. Do not overwrite or conflate it with the stable20-critic/10-actor recipe. Earlier sections describing either sequence as active are historical.

## Earlier maintenance before the launch — November 2, 2026

- User explicitly keeps the fixed ICON context, including cross-scene differences; current focus is optimization stability. That earlier maintenance changed no context/reward settings and launched no training/evaluation job. The subsequent authorized launch is recorded above.
- New training metric: `env/success_rate_ma100` plus `_count`, exact latest100 completed training outcomes across ranks; startup uses available completions. Padding and unfinished episodes are excluded; evaluation does not enter the window. Window is driver-local and resets after restart/resume. Current sync order is collection/epoch/terminal-chunk/stage/rank/slot; async order is within each received metric batch. Implementation and TensorBoard round-trip passed17 CPU tests.
- Backfilled both completed chunk20 TensorBoards from all3,200 episode sidecars each, preserving existing events. Final100 training rates: ICON86%, Robometer89%. Evidence: `outputs/icl_debug/success_rate_ma100_backfill_2026-11-02.json`.
- All current config filenames omit version/size labels. Canonical trial: `robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable`; eval adds `_eval`. Completed comparison configs are `..._dsrl_icon_chunk20`, `..._dsrl_robometer_chunk20` and their `_eval` variants. Shared parents are `_..._dsrl_base`, `_..._dsrl_simulator`, `_..._dsrl_learned_reward`; old recipes are descriptively archived. All28 configs compose and21 migrated recipes preserve exact effective settings. Map: `outputs/icl_debug/config_rename_map_2026-11-02.json`. Dated output/manifests keep original names/hashes; resolve old config names through this map.
- Completed Robometer20:138/150=92%; ICON20:131/150=87.33%; SFT20:118/150=78.67%. Robometer ended October31 20:34Shanghai. Historical notes below calling it running are superseded.
- Authorized cleanup deleted exactly three checkpoint directories: simulator small200, PPO5 and obsolete bug-era Robometer400. Retained eight final useful comparison checkpoints with full replay/optimizer state. Freed74.52GiB; approximately118.43GiB free. No base models, reward servers, context data, logs or evaluation records removed. Inventory plus execution proof: `outputs/icl_debug/checkpoint_retention_2026-11-02.json`. Active simulator eval now loads retained filtered400; archived pilot eval refers to retired weights.
- Paper check uses Robometer v2 (May13), AppendixE-2/TableXXII, not the LIBERO reward-ablation table or offline IQL table. Paper single-stage: batch64, actorLR5e-5, criticLR1e-4, tau.005, gamma.995 per physical action,4critics/sample2,10actorupdates per logicaltrainstep,4critic peractor, auto-entropyLR1e-4/target0, warmup1200,10ksteps,8executed/10predicted actions. Official runner invokes logicaltrainsteps per newchunk after an episode; these counts cannot be copied directly to RLInf's eight-episode collection. Current upstream code defaults differ from the paper table. Full source links and comparison: `outputs/icl_debug/robometer_dsrl_hyperparameters_2026-11-02.json`.

## Historical preparation and experiment notes

The sections below preserve earlier decisions and identifiers. Use the latest section, current README and live artifacts for current state.

## 1. Current state — read this first

### Matched Robometer chunk20 comparison — latest active experiment

- User asked to run a matched Robometer evaluation in the background after discussing ICON87.33% versus matchedSFT78.67%. No Robometer checkpoint had20-action execution/400iterations/50SACrounds, so a truly matched comparison requires fresh training before evaluation. The assistant explicitly told the user that the queued sequence includes that training. Do not silently label an evaluation of the older50-action/100-iteration checkpoint as matched.
- Newconfigs:`robotwin_adjust_bottle_pi05_dsrl_v2_large_robometer_chunk20` and `robotwin_adjust_bottle_pi05_dsrl_v2_robometer_chunk20_eval`. All312validatedICONsettings match its actual saved config; Robometer differs only in5server/output-related fields. Preserved20exec/50decoder/200actions,400iterations,8attempts/iteration,50SACrounds,global128/micro64,readiness25→18,800actor/criticsteps,alpha.01,gamma.999,LR1e-4/3e-4,filteredreplay,dense8andunnormalized`progress-1+native_success`.
- OriginalRobometer server freshly started onGPU1/8000 with`robometer/bin/python`, PYTHONPATHonly`third_party/robometer`, model`checkpoints/Robometer-4B-inference`, no context. PID2517162. Actualserverprocessbootstrap`outputs/robometer_chunk20/start_server.py` recordedpatched`models/utils.py`SHA256`f8622bef9e2e61eb7f92969244b2af378dd2f19db261ed96b251c98444d0d694`,from_logitsdefaulttrue,andlogitssum1caseexpected.573316261/actual.573316216. This establishesloadedrepairfortheNEWexperiment,notretrospectivelyfortheold100run. Sourcefingerprint/health/modelinfo/serverlogsareallunder`outputs/robometer_chunk20`. ExistingICON8001leftuntouched.
- Detachedsupervisor2518344:`scripts/run_robometer_chunk20_experiment.py`, status`outputs/robometer_chunk20/experiment_status.json`. Stages:2iterationintegrationcheckwith100SACrounds/nocheckpoint→fresh400training→150seedfinalevaluation. It verifiesfinalglobal_step400onbothrolloutrankswith20execution/50horizon. ExistingICON/SFT20evaluationsarecheckedandreused,notrerun. Freshtrainoutput`outputs/robometer_dsrl_chunk20_large_s1234`, experiment`pi05_dsrl_robometer_chunk20_large`, finaleval`outputs/robometer_dsrl_chunk20_large_eval_150`;combinedreport`outputs/robometer_chunk20/matched_comparison.json`appearsaftercompletion. Checklivecurrentstatusbeforestartinganything; nofinalRobometer20resultyet.
- Config/report:`outputs/robometer_chunk20/config_validation.json`,30CPUchecks+resolvedYAMLs. Freefilesystemspace~99GiBatlaunch;comparableICONcheckpoint56GiBfitswith~43GiBremaining. Keepfinal-onlycheckpointandshortscopedRAMRaypaths`/dev/shm/r20-<pid>-<stage initial>`. Do notdeleteexistingusercheckpoints/cachesorstopunrelatedprocesses.
- IntegrationcheckcompletedOct31 14:17:08Shanghai:100SACrounds,criticloss.071/.0076,actorloss−.019/−.152,actorgrad.376/.556,criticgrad.291/.082,allfinite/nonzero. Freshfulltrainingdriver2526869thenstartedunder2518344;SFTsource/horizon50/execution20verifiedonbothrolloutranks. FourvideoJSONsauditedexactunnormalizedrewardandpadding. Reports:`outputs/robometer_chunk20/{smoke_validation,video_reward_validation}.json`. Fullrunhasreadiness25anddoesnotcontinuethesmoke'supdatedheads.
- ICON20pipelineisCOMPLETEatOct31 12:15Shanghai: updated131/150(87.33%),matchedSFT118/150(78.67%),difference13successes/8.67percentagepoints. Trainingloop6:30:08,updatedeval10:56,SFTeval11:50. Status`outputs/icon_chunk20/experiment_status.json`;bothrankloaderproofsareintheupdated/baselineJSONs. PreviousICONlaunch/warmupnotesbelowarehistoricalandmustnotbereportedcurrent.

### Authorized ICON chunk20 experiment — October31

- User explicitly requested20-action execution directly with ICON, unnormalized estimated rewards, increased SAC rounds, config preparation and detached training/evaluation. This authorizes the current GPU work and supersedes earlier manual-run preferences for this experiment. Inspect `outputs/icon_chunk20/experiment_status.json` and host processes before doing anything; do not start a duplicate.
- New training config:`configs/robotwin_adjust_bottle_pi05_dsrl_v2_large_icon_chunk20.yaml`. Fresh SFT/controller/replay,400iterations,8attempts/iteration,200actions/episode,20executed/50predicted,50SACrounds,global128/micro64,readiness25→18,800actor/criticsteps each. Actor/criticLR1e-4/3e-4,alpha.01,gamma.999,tau.005 unchanged; discount automaticallyγ^20≈.98019. Reward is exactly `progress-1+native_success` once per20-action chunk,scale1; no density normalization. Native success/padding exclusion, eight query/context frames and demo remain unchanged.
- Supervisor:`scripts/run_icon_chunk20_experiment.py`, metadata:`outputs/icon_chunk20`, maintraining:`outputs/icon_dsrl_chunk20_large_s1234`, experiment:`pi05_dsrl_icon_chunk20_large`. Stages: two-iteration integration check (one collection round/iteration,50SACrounds,readiness1,no checkpoint), fresh full400training, final150seedDSRLevaluation, matching150seedSFT20baseline. Training state never resumes from the smoke or previous50-action checkpoint. Final400checkpoint is the only saved checkpoint.
- Evaluationconfigs:`robotwin_adjust_bottle_pi05_dsrl_v2_icon_chunk20_eval` and `robotwin_adjust_bottle_pi05_sft_chunk20_eval`. Both explicitly execute20 with50networkhorizon,200actioncap,original150uniqueheldoutseeds,novideo. DSRL uses deterministic learned mean noise; SFT uses its standard sampling. Outputs:`outputs/icon_dsrl_chunk20_large_eval_150` and `outputs/sft_chunk20_eval_150`. Supervisor validates bothrankloaderlogs identify the intended checkpoint and chunk20 before reporting complete.
- Config report:`outputs/icon_chunk20/config_validation.json` with30CPUchecks and resolvedYAMLs. Private learned-reward parent avoids Hydra's nested publicsearchpath restriction; shared parents/oldruns unchanged. Resource check found ~96/79GiB GPU memory free, ~440GiB RAM available,153GiBdiskfree. KeepRAMRaytemp; fullcheckpoint+replayestimated55–60GiB. ExistingICON8001healthyPID599189 reused; originalRobometer8000hadno listener. No reward-model server is stopped by this pipeline.
- Initial supervisor1590783 failed before GPU training because its Ray UNIXsocketpath exceeded107bytes. Artifacts archived under `outputs/icon_chunk20/attempt_001_ray_socket_path`; no active processes remained from that session. Launcher now uses `/dev/shm/c20-<pid>-<stage initial>` and was relaunched with supervisor1591506. Current stage/result must be read from live status rather than this launch-time note. No final training/evaluation result exists yet.
- Successful integration check finished at05:20:50Shanghai and automatically started the fresh400run, driver1600293.100actualSACrounds; actorgrad.272/.325,criticgrad.424/.097;criticloss.130/.0074,actorloss.021/.021,allfinite. Both rollout ranks log50networkhorizon/20execution. Four videoJSONs contain29validrewardchunks and11unscoredpaddingchunks, maximum20realactions/chunk, all `reward=progress-1+native_success` exactly withinfloat32tolerance. Allfour attempts happened to succeed; this is a tiny integration check, not capability evidence. Reports:`outputs/icon_chunk20/{smoke_validation,video_reward_validation,background_launch_verified}.json`. Fulltraining starts fromSFTagain, notthe100updatedsmokeheads.
- At05:22:47 fulltraining completed its first8-attempt collection and proceeded to the next iteration. Replay size1<25 correctly skips fullrunupdates during warm-up; do not mistake this for failure or absentupdates(the separate100-roundcheck already verified optimizers). Training/evaluation remain background work, not finished results.

The user chose **RLinf + RoboTwin + π0.5 + DSRL + Robometer**. The immediate pipeline is implemented and has passed short tests on both host GPUs. DSRL uses SAC to train a latent-noise steering policy while freezing the task-specific π0.5 SFT decoder. The existing PPO path remains available. The eventual research goal is to replace Robometer with the user's in-context reward model.

The user authorized building the proposed sequence and running/fixing its short integration checks. The user subsequently ran the 200-iteration DSRL experiment; the assistant verified completion at 19:06 Shanghai time and both checkpoints at iterations 100 and 200. The user received commands for the paired 150-seed SFT and learned-policy evaluations; check their actual outputs before reporting completion. Do not recreate custom environment setup/activation scripts, replace the checkout, update submodule pins, or broadly upgrade existing RLinf dependencies. The user explicitly authorized the targeted native OIDN repair below on October 28; the earlier preference to avoid OIDN updates was superseded for this repair. Both uv environments are named directly `rlinf` and `robometer`.

The current commands, exports, environments, architecture, server instructions, and pilot/evaluation recipes are in [README.md](README.md). **This section and README take precedence over historical sections below.**

### Current results and next steps — October 31 audit

- User requested an up-to-date README summary and suggestions. README now inventories all 11 completed 150-seed evaluations, their budgets/times, current comparison limits, and five prioritized next experiments. This audit launched no training, evaluation, simulator or model-inference jobs; do not start another run merely because an earlier launch recipe remains in the document.
- ICON400 training and automatic evaluation are complete. Reported loop times: training **5:20:25**, final evaluation **9:47**, finishing October30 at23:32 Shanghai time. Supervisor `outputs/icon_reward/experiment_status.json` is `complete`; training/supervisor PIDs are no longer active. The ICON reward service can remain running independently.
- Current results: SFT122/150(81.33%), zero-noise122/150; simulator DSRL200122/150, v2small120/150, v2large unfiltered129/150(86%), v2large filtered126/150(84%); original Robometer200121/150; historical endpoint/delta filtered40040/150(26.67%); dense8/single-stage filtered Robometer100114/150(76%); dense8/single-stage filtered ICON400124/150(82.67%); failed PPO1000/150. ICON is only+2successes versus SFT and−2 versus filtered simulator, from one training seed; no robust gain demonstrated. Robometer100 versus ICON400 has unequal budgets, and old26.67%is not a current matched control.
- **Robometer deployment caveat:** currentport8000listenerPID3365898 started2026-10-29 at19:33:59+08. Original `models/utils.py` repairmtime isOct30 10:56:59; dense100 savedconfig11:42:34, finaleval13:23:47. No service restart or loaded-source fingerprint is recorded before the pilot; unrecorded manual reload cannot be excluded. Label the100run as **dense8 single_stage, repaired live converter not verified**. Restart and fingerprint the repaired original server before any next comparison. Do not call the100result a verified fully repaired baseline.
- New durable evidence: `outputs/icl_debug/experiment_inventory_2026-10-31.json` (11results, config/loader proof, checkpoint availability, file hashes, deployment caveat); `icon_final400_audit_2026-10-31.json` (compact state/update audit); `comparison_protocol_2026-10-31.json` (comparability/uncertainty). Both rollout ranks explicitly loaded intended final checkpoints for ICON400, denseRobometer100 and simfiltered400. Evaluation env settings match except video output, with the same150scene seeds and mean-policy DSRL inference. Per-scene outcomes were not retained for paired analysis.
- ICON compact-state audit: both ranks7520updates;188online+188target+188shadow tensors finiteFP32 and matching replicas; all32actor+156critic Adam counters7520 and all376moments finite/nonzeroFP32. Formerly zero output biases changed. Training2608/3200successes81.5%, first100iterations77.5%, last10084.5%; no numerical collapse found. Full frozen decoder/replay were not reread. Wall-process durations differ from loop metrics (training5:21:32/eval11:18), so README labels its times as reported training-loop times.
- Next priorities: record per-scene evaluation outcomes and complete a verified-service Robometer400 control; score independent raw training prefixes for model/context quality; ablate learned progress against the native `-1+success` base; repeat promising matched recipes with at least3training seeds; investigate head-only intermediate rendering while preserving all3policy cameras at boundaries. These are suggestions, not launched experiments. Current full400Robometer config remains unrun.
- Cleanup removed old `outputs/dsrl_debug`, `oidn_debug`, `ppo_debug` audit reports and some test artifacts. The original simulator200 checkpoint is gone, although its completed evaluation and two-rank loader logs survive. Other formal checkpoint directories were present at the audit. README's stale local links and old "audit files remain" statements were corrected; historical preparation notes below are superseded by this section and the new inventory.

### Historical ICON integration and background launch — October 30

- The user completed step1 in **`.venv-icon`**, then explicitly authorized the assistant to fix steps2–6, collect the successful reference, and launch a large ICON experiment in the background. This supersedes the earlier manual-GPU-only preference for this requested work. Do not launch duplicate training; inspect `outputs/icon_reward/experiment_status.json`, `experiment.pid` and host processes first.
- Implemented server-side fixed context with optional `context_demo_path`, keeping the same `robometer.evals.eval_server` entrypoint and query-only HTTP interface as original Robometer. No demo fields were added to RL trajectories/replay/policy inputs. The ICON server owns one cached eight-frame reference; `/health` and `/model_info` expose its hash/config/identity. Supplied per-request context is rejected in fixed mode. Language paraphrases need not equal the reference instruction literally.
- Repaired the v2 logits conversion and multipart context key lists, preserved context during `use_frame_steps` expansion, and added strict query/progress/success length checks in the root HTTP client. Current RL requests use false: their growing prefix is scored in one forward pass. True explicitly expands separate prefixes. Context progress tokens are removed before query predictions.
- `scripts/collect_icon_context.py` uses native Pi0Eval and one native RoboTwin env directly, without Ray or optimizers. Seed0 timed out; training seed1 succeeded after127 real actions. The saved128 raw head-camera RGB frames include reset and true first-success image, with all terminal padding excluded. Fixed artifact:`data/context_demos/adjust_bottle/demo_000/{frames.npy,metadata.json}`, SHA256`69e02edeeec98efdbc424adf03449a74f38ecf9f98c157c0ce8fec862ae3e9ff`; eight indices`[0,18,36,54,73,91,109,127]`. Manifest records requested/actual seeds, native success, source SFT checkpoint, camera/renderer, code hashes and zero optimization. No held-out data used.
- ICON service launched detached on physicalGPU1,port8001, PID initially599189; authoritative PID:`outputs/icon_reward/server.pid`, log:`outputs/icon_reward/server.log`, launchargs:`launch.json`. Existing original Robometer server on8000 was idle and left untouched; it may still have older code imported and must be restarted before a corrected Robometer comparison run. ICON actual GPU loading succeeded despite inherited base-loading warnings; subsequent full ICON shards loaded, and no RBM missing-weight warning appeared.
- Validation:116rootCPUtests passed; actual GPU scoring returned correct query-only lengths and bounded progress. On the reference itself, endpoints~.009,.386,.803,.998 at action0,50,100,127; static-reset control~.020. This is protocol validation using the reference as query, not independent reward accuracy. Two-iteration GPU DSRL smoke exited0, collected16episodes(15native successes), performed four configured SAC rounds, finite actor/critic losses and nonzero actorgrad norms.128global/32micro batch used. All16labeled-video JSON sidecars were checked for exact native-success single-stage reward, real frame endpoints, eight query images and unscored post-terminal padding. No smoke checkpoint saved.
- Evidence:`outputs/icl_debug/icon_integration_validation.json`, `icon_cpu_tests.log`, `icon_dsrl_smoke.log`, `icon_offline_scoring.json`, `icon_dsrl_config_validation.json`. The latter confirms all actor/algorithm settings match simulator-filtered; ICON vs corrected Robometer raw config differences are exactly outputpath/experimentname/serverURL. Tests cover transport, token routing, numeric conversion, collection/seed isolation and response lengths.
- Large config:`configs/robotwin_adjust_bottle_pi05_dsrl_v2_large_icon.yaml`, existing corrected dense8 single-stage Robometer parent.400iterations,2env×4rounds=8attempts/iteration,3200attempts,7520SACrounds from readiness25,batch128/micro32,alpha.01,actorLR1e-4/critic3e-4,FP32heads,frozenπ0.5,actorseed1234/envseed0,horizon200/chunk50/flow5,filtertrue,freshheads/replay,noresume. Training videos retain30FPS progress labels and sidecars. Comparison simulator-filtered has actually completed **126/150,84%**, correct final400 checkpoint confirmed in both loader ranks. Current corrected Robometer dense8/single-stage is prepared but unmeasured; historical26.67% endpoint/delta run is not a matched corrected baseline.
- Background supervisor launched as PID initially610830; read `outputs/icon_reward/experiment.pid` and `experiment_status.json` for current stage/childPID. Artifact `run_large_experiment.py` runs fresh400 training then automatic150-seed native-success evaluation, with errors/status persisted. Trainstdout:`outputs/icon_dsrl_dense8_single_stage_large_s1234.log`; trainmetrics/video/TensorBoard:`outputs/icon_dsrl_dense8_single_stage_large_s1234/`; finalcheckpoint:`pi05_dsrl_icon_dense8_single_stage_large/checkpoints/global_step_400` beneath that directory. Finalevalstdout/directory:`outputs/icon_dsrl_dense8_single_stage_large_eval_150{.log,/}`. Full policy evaluation does not use ICON/context.
- Ray uses scoped `/dev/shm/icr-ray-icon-large-training` and `...-evaluation` to avoid rootFS95% spill warnings; preserve existing dual-GPU NCCL2.31.2 preload/P2Pdisable. Smoke scopedRay already exited normally. Measured dense rollout~43s, fullstep~45s with two SAC rounds;400training with20rounds estimated5–6hours plus finaleval. Do not report full training complete until status/400checkpoint/eval150 prove it. Reproduction/model restart/monitor commands are in README's ICON section. Experiment manifest saves pins, reference hash and relevant source hashes.

### Historical ICON repository review and initial integration plan

- User asks to review `sNiper-Qian/robometer-v2`, choose its submodule location and list TODOs. They supplied the actual reward checkpoint **`sniper-qian/ICON-4B`**. This request is review/planning; no GPU collection/training/evaluation was launched. User continues to run GPU jobs themselves.
- Read-only review used temporary clone `/tmp/icr-robometer-v2-review` and metadata `/tmp/icr-icon-metadata/`; durable metadata/schema audit copies are in `outputs/icl_debug/icon_repository_review_2026-10-30/`. SSH GitHub access works; anonymous HTTPS did not. No permanent submodule or environment was added and no model weights were downloaded.
- **Correct branch is `full_training`, code pin `b68c27ef7df11cdebdfef995c78644c3cb33cdf3`.** The default `master` (`a59a217`) does not match ICON: strict checkpoint-config construction fails on unsupported fields, and its collator omits context progress tokens. Other reviewed branches (`dtw`, `reset_progress_token`, `small_model`) also do not match all fields. `full_training` has zero unsupported ICON config fields across model/data/loss/PEFT/training/logging/custom-eval. Runtime model loading remains untested.
- Model is public/ungated; reviewed HF revision **`e75b5fca997ab18f620ecb1931e5d9acba7c1f9b`**. Its config: Qwen3-VL-4B, BF16, full weights (`use_peft:false`), ten progress bins, multi-image/per-frame progress tokens, `conditioning_mode:language_context`, `context_frames:8`, `add_context_progress_tokens:true`, `include_language_with_context:true`, zero language/context dropout, same-source/view training requirement. Saved `use_unsloth:true` needs a deliberate inference override if using the existing standard Transformers pattern. Preserve trained conditioning/preprocessing settings; do not filter unknown keys to force master to load. Metadata includes training-only DTW cache paths; deploying reward inference does not imply reconstructing that training dataset/cache.
- Proposed placement: `third_party/robometer_v2` as an additional pinned submodule; separate uv environment `robometer_v2/` (add it to root ignore when created), proposed service port8001 and checkpoint directories `checkpoints/ICON-4B`/`ICON-4B-inference`. Both forks use package name/import `robometer`; never install both in one env or combine both source roots in PYTHONPATH. Preserve original baseline and RLinf env.
- `full_training` still has three integration issues: (1) raw-logits-sum==1 conversion bug in `robometer/models/utils.py`—port the existing explicit logits/probabilities repair and tests; (2) `context_trajectory` omitted in multipart key lists in `evals/eval_utils.py` and `evals/eval_server.py`; (3) `use_frame_steps:true` drops context while expanding prefixes, so retain false. Its context progress-token counting/exclusion is implemented: collator `context_progress_token_counts`, server forwarding, RBM extraction skip context tokens. Responses should cover query frames only. Add an exact returned-T check in our client.
- Collection TODO: one independently successful task-specific SFT rollout on training/dedicated support seeds, retrying until native success; no optimization. Raw head-camera RGB uint8 THWC, reset through true first-success frame, no labels/cropping/padding. Keep native raster/default renderer and disabled denoiser as benchmark. Store `data/context_demos/adjust_bottle/demo_000/{frames.npy,metadata.json}`; manifest includes requested **and actual** scene seeds (RoboTwin unstable-scene retries can advance the seed), instruction, simulator success, real frame/action count, SFT/code revisions, camera settings, selected eight reference indices and artifact hash. Do not reuse held-out150 evaluation clips. Never feed annotated MP4 pixels to ICON.
- Generic `CollectEpisode` is not a finished raw context exporter: chunk recording ignores valid-action masks, evaluation does not publish the training mask, default eval ignores terminations, and reset infos omit actual retry seed. Add a small dedicated mask-aware collection/export path or fix these explicitly; reuse `build_reward_chunk_metadata` for true terminal images/valid frames. Ensure the selected successful actual scene belongs to the permitted support/train split. Requested seed or video filename alone is not adequate provenance.
- Bridge TODO: keep fixed reference separate from per-env growing query histories. Send one `context_trajectory` plus one `trajectory`; explicitly supply eight context frames (inference collator does not subsample them automatically), using `full_training`'s deterministic endpoint-spanning nearest-rounded `evenly_spaced_subsample_indices`, and preserve the current up-to-eight query-prefix sampling. Add a distinct backend/config/output, demo manifest/hash loader and checkpoint/context-mode checks; reuse single_stage native-success base and terminal padding filter. Missing/disabled context or wrong model identity must fail before training. Context counts/tokens must not inflate returned query T. Keep equal-query-shape request grouping because heads stack query tensors directly.
- Validation TODO: focused CPU serialization/token-count/shape tests, training-only offline successful/failed-prefix and wrong-context scoring, then user-run GPU inference and two-iteration DSRL update check, fresh short pilot and larger run. Start new controller heads/replay because the estimator changed; keep frozen π0.5 and reward model. Record fixed demo/model/code revisions and inspect score videos. Sixteen total images per maximum request (8 context+8 query) require new latency/VRAM profiling; old Robometer time estimates do not apply.
- Final policy-only150 evaluation uses the same held-out benchmark and simulator success; the ICON service/reference is not required for policy rollout. For offline reward-model evaluation the reference is required. README's **Planned ICON reward integration** section is the durable ordered TODO; there is not yet a runnable ICON training config/collection command.

### Prepared simulator filtering comparison

- User wants to measure simulator-reward DSRL after post-terminal replay filtering and edit later hyperparameters themselves. Expanded `configs/robotwin_adjust_bottle_pi05_dsrl_v2_large_filtered.yaml` into a commented per-experiment override file: budget/save/eval, two-env collection, horizons/chunking, batches, actor/critic LRs, alpha, replay readiness and filter are explicit. Do not edit shared parents to tune only this run.
- Prepared values deliberately match the previous86% unfiltered simulator recipe:400iterations,2env×4rounds=8attempts/iteration,3200attempts,total7520SACrounds,batch128/micro32,20rounds,minreplay/actor25,FP32heads,alpha.01,LR1e-4/3e-4,200/50/fiveflowsteps,endpoint-only capture,no training video/eval. Native reward0/+1; `reward.use_reward_model:false`. No Robometer single-stage penalty applies. Start fresh from SFT/emptyreplay.
- Filter keeps real zero-reward chunks and first success/terminalchunk +1; excludes only later chunks from minibatch sampling. Raw checkpoint/replay padding remains stored. Only filtertrue and outputnames differ from unfiltered-large fully resolved config.
- Output/checkpoint:`outputs/dsrl_sim_v2_large_filtered_s1234/pi05_dsrl_v2_large_filtered/checkpoints/global_step_400`; full150evaloutput:`outputs/dsrl_sim_v2_large_filtered_eval_150`. Both commands and field-edit table are at the start of READMEsection1. No reward server needed. No GPU/job was launched.
- CPU Hydra train/eval composition and FSDP validation passed; report:`outputs/dsrl_debug/simulator_large_filtered_editable_validation_2026-10-30.json`, resolvedconfig:`simulator_large_filtered_editable_resolved_2026-10-30.yaml`. This is config preparation, not a measured filtered SR result.

### Current Robometer defaults — dense8 and simulator-gated single-stage reward

- User requested keeping growing histories but sampling more actual action frames, plus the paper-inspired single-stage reward without any model success threshold. This supersedes the endpoint/delta recovery instructions in the historical October30 audit below.
- All Robometer DSRL defaults now use `reward.model.reward_mode: single_stage`, `max_frames:8`, `env.train.skip_intermediate_renders:false`, videos30FPS. Maintain every real post-action head image in temporary per-episode history; uniformly select up to8 across the entire prefix including initial/current. No repeated terminal padding or extra duplicate endpoints; no separate reset-only HTTP request. Dense images are not stored in SAC replay or actor observations.
- Scalar reward per50-action chunk at slot49: `progress - 1` while unfinished or timeout; `progress` only on explicit simulator success. Model success probabilities never decide termination or base reward. Simulator sparse mixing weight0/modelweight1; native success determines the -1/0 base inside the adapter. Existing `progress_delta` and `progress` stay explicit opt-ins; PPO config remains explicitly delta. Do not claim this reproduces the original paper's chunking or model-driven success detection.
- EnvWorker's `build_reward_chunk_metadata` transports raw `[B,T,H,W,3]` frames, `[B,T]` action validity and explicit native success before LeRobot rewrites observations. Validity is available even with LeRobot disabled. True terminal RGB replaces the last VALID frame on auto-reset; non-done rows are untouched. Recorder labels the actual final valid frame (e.g7,notpadded50), marks later padding unscored, and writes native outcome/actual frame indices into score JSON.
- Current manual smoke/recovery/full150 commands are in README section3. Recommended recovery100iterations800attempts1520SACupdates, batch128/micro32,20rounds,minreplay25,alpha.01,existingLRs;20episode monitoring every25,onlyfinal100saved,noresume. Newoutput `outputs/robometer_dsrl_dense8_single_stage_s1234/pi05_dsrl_dense8_single_stage/checkpoints/global_step_100`, evaloutput `outputs/robometer_dsrl_dense8_single_stage_eval_150`.
- Full400option newoutput `outputs/robometer_dsrl_dense8_single_stage_large_s1234/pi05_dsrl_dense8_single_stage_large/checkpoints/global_step_400`. Preserve prior endpoint/delta experiment outputs. Dense rendering runtime has NOT been GPU-profiled; old35–50min/2hestimates no longer apply. Run the two-iteration integration check first. No GPU job, production server or simulator was launched by the assistant.
- CPU tests and Hydra/FSDP config checks cover new defaults plus legacy delta. Config evidence:`outputs/dsrl_debug/robometer_dense8_single_stage_config_validation_2026-10-30.json`; 38focusedCPUtests passed(27reward/HTTP/integration+11video/dense-metadata/routing); Ruff and whitespace checks passed. Final report:`outputs/dsrl_debug/robometer_dense8_single_stage_validation_2026-10-30.json`.

### Historical endpoint/delta audit and superseded next step, October 30

- Filtered Robometer400 finished in 2:04:08 with 3,200 episode attempts, 7,520 SAC updates and 40/150 held-out successes (26.67%). Both rollout workers loaded the correct final400 checkpoint. Eval environment/model settings match simulator-large except checkpoint; deterministic mean noise, FP32 heads, same 150 seeds. Durable loader evidence and fixed-observation CPU head probe are in `outputs/dsrl_debug/robometer_filtered_{eval_loader_rank0.log,eval_loader_rank1.log,head_probe_2026-10-30.json}`. Mean absolute deterministic latent .37825 versus simulator .13536, without hard tanh saturation; no decoded physical actions measured.
- Audit artifacts: `outputs/dsrl_debug/robometer_filtered_analysis_2026-10-30.{json,csv,png}`. Training SR in 50-iteration windows:78.25,78.5,73,75.25,63.25,58.75,57,49.75%. All scalars/rewards finite. Replay10798 valid/12800 raw,2002 padding all zero and excluded; first terminal rewards retained. Filter is functioning. The 86% simulator run was unfiltered, so it is not a matched filtering control.
- Definite server utility bug: `third_party/robometer/robometer/models/utils.py::convert_bins_to_continuous` treated raw signed logits as probabilities if their sum happened to equal1. CPU proof returned progress-5.444 instead of ~0. Replay has12 first-chunk rewards4.04–5.82,60.94% of squared raw reward. The location fits single-reset-frame baseline corruption; old logits/p0/pt were not retained, so historical causal connection remains inferred. Conversion is repaired with explicit logits/probability handling; HTTP bridge now rejects progress outside[0,1]. Restart the already running reward server to import the repair; no environment upgrade needed.
- Reward alignment also poor independently of outliers: last100 episode gain AUC.5077, or .5104 after excluding spike episodes. Failed episodes can receive similar/higher learned return. All1065 timeout failures still bootstrap, whereas2135 true successes terminate value targets; this potential bias is unchanged/shared with simulator baseline. Do not claim conversion fix alone restores SR.
- Exact online progress/delta/reward now travels from reward worker to the video recorder without changing SAC reward tensors. Save-time footer and per-MP4 JSON label reset/current progress and chunk indices; intermediate frames show last scored progress, later padding unscored. No additional HTTP calls or modified model pixels. Learned-reward configs enable training video at2FPS using endpoint frames; full motion requires skip_intermediate_renders=false andfps30 and costs extra rendering. Old training/eval had no videos and cannot gain exact historical absolute scores retrospectively.
- Next manual test: restart corrected server, then `--config-name robotwin_adjust_bottle_pi05_dsrl_robometer_recovery`. Fresh SFT/empty replay;100iterations800attempts1520SACupdates, batch128/micro32,20SAC rounds,min replay25,alpha.01,existingLRs unchanged. Monitor20episodes every25iterations, save final100only. Output `outputs/robometer_dsrl_recovery_s1234/pi05_dsrl_robometer_recovery/checkpoints/global_step_100`; labeled clips/JSON in output `video/train/`; full150eval command in README section3. Estimated35–50min including monitoring, unprofiled. Do not resume contaminated Robometer400/replay. If reward remains misaligned after repair, inspect scored success/failure clips, use matched filtered simulator control, then consider explicitly labeled hybrid(1sim+.1model) rather than assuming longer pure learned-reward training fixes it.
- This turn used CPU/files and a tiny localhost test fixture only; no production Robometer process, GPU job, simulation or training was launched. User runs integration/training/evaluation manually. Validation:62focusedCPUtests(45conversion,10HTTP,7video/routing/outcomes),20actualRobometer-env assertions,MP4encode/decode,Hydra/FSDPconfigandRuffchecks passed. Report:`outputs/dsrl_debug/robometer_repair_validation_2026-10-30.json`. JSON sidecars also retain native termination/truncation and available simulator success/return.

### Historical experiment audit, October 29

Large simulator-reward DSRL checkpoint400 scored **129/150 (86.0%)**, versus SFT/zero-noise122/150. The full400-iteration/3,200-attempt run completed in1:50:31 with7,520 verified updates. See “Larger DSRL result and recommended next move” below; the original simulator200 audit in the next bullets is historical. No new GPU job was launched by the analysis.

- The user completed simulator-reward DSRL training at `outputs/dsrl_sim_benchmark_s1234` on October 28, then evaluated the final checkpoint on October 29. The resolved evaluation config and both rollout-rank loader logs confirm `outputs/dsrl_sim_benchmark_s1234/pi05_dsrl/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt` was loaded with `openpi.task: dsrl`; it was neither original SFT nor the Robometer checkpoint. Proof: `outputs/dsrl_sim_raster_eval_150.log` lines 313–320. The SFT path printed for norm stats is expected shared normalization, independent of learned weights.
- The original simulator200 evaluation result is **122/150 (81.33%)**, with all 150 videos present and evaluation complete. All four formal evaluations have identical saved environment settings after excluding video output directories: same task, configured held-out seed source, cameras, normalization, horizon 200, chunk 50, five denoising steps, and raster/default shader with no denoiser. Simulator DSRL and SFT share the aggregate score; per-seed success labels were not retained, so do not assert identical successful seeds.
- CPU audit passed: all 855 inference tensors finite; all 667 frozen FP32 π0.5 tensors exactly match SFT; 188 learned BF16 DSRL tensors exactly match the saved training replica; online and target replicas match across ranks. Both ranks' update counters and all 188 optimizer parameter counters are 3,820, with finite/nonzero moments. Previously zero-initialized mean/log-std biases changed. Simulator and Robometer DSRL differ in 151/188 trainable tensors. Replay has 3,200 chunk slots and sparse simulator rewards, with model reward disabled. Reports: `outputs/dsrl_debug/simulator_final200_audit.json` and `simulator_final200_vs_robometer.json`.
- Deeper DSRL audit: simulator checkpoints 100→200 advanced all 188 Adam counters 1,820→3,820 and changed 27/32 actor tensors plus 123/156 critic tensors. Actual `Pi0DSRL.sac_forward` on 20 identical replay observations changed all 640 independent deterministic latent components (mean absolute delta 0.02707, maximum 0.09595). A recording stub confirmed actual `predict_action_batch` passes this latent to the decoder and replay; it did not run the physical-action decoder. Fresh Robometer final-checkpoint audit also passed. Reports: `outputs/dsrl_debug/{simulator_update_evidence_2026-10-29,functional_update_audit_2026-10-29,robometer_final200_audit}.json`; reusable diagnostic: `scripts/audit_dsrl_functional.py`. **Optimization issues identified by the audit (addressed by the v2 source changes below):** `fsdp_sac_policy_worker.py` clips the whole model at both optimizer phases while other-group gradients persist; actor backward also computes Q-head gradients because only critic encoders are detached. A CPU backward-only probe reproduced 140 Q-head gradient changes during the actor phase. Separately, heads and plain-Adam moments are BF16 with no FP32 master weights despite the base model's FP32 storage; small updates can round away. The later v2 change clears inactive gradients and adds explicit FP32 head storage/computation; GPU integration remains to be checked by the user. Their success-rate impact is unmeasured; the synthetic CPU probe is not a historical distributed training reproduction. This audit made no production training/config/checkpoint changes, optimizer steps, GPU calls, or Ray sessions.
- Formal completed results: SFT **122/150, 81.33%**; DSRL + simulator **122/150, 81.33%**; DSRL + Robometer **121/150, 80.67%**; PPO + simulator (`outputs/ppo_sim_raster_s1234`, checkpoint 100) **0/150**. Both DSRL runs used 200 iterations, 800 episode attempts and 3,820 SAC updates. Completed PPO used 100 iterations, 800 attempts and 1,000 optimizer steps. Training-loop times were approximately 32:49 (simulator DSRL), 42:17 (Robometer DSRL), and 2:34:55 (PPO); full evaluations took 10–13 minutes. No improvement over SFT is demonstrated by the DSRL runs so far.
- An additional scaled official-based PPO run exists at `third_party/rlinf/logs/20261028-15:00:41-robotwin_adjust_bottle_ppo_openpi_pi05`. It recorded **11/1,000 planned iterations**, eight environments × eight rounds = 64 episode attempts/iteration, global batch 256, microbatch eight, one PPO epoch, LR 5e-6, suffix critic (`value_after_vlm: false`), no critic warmup. Checkpoint 10 exists and scored **7/8 (87.5%)** in monitoring; there is no full 150-seed evaluation for this checkpoint. The log ends during a subsequent rollout around 15:32. This is separate from the 0/150 PPO checkpoint 100 and does not validate the proposed prefix-critic/lower-LR recipe. No active RLinf training/evaluation driver remained when inspected on October 29; do not call this run still running or complete.
- Other attempts: `ppo_adjust_bottle` has partial restarts with nine and six completed iterations and checkpoint five; the official PPO attempts at 14:08/14:18/14:46 on October 28 failed at startup (NCCL, then camera buffer allocation). `robometer_dsrl_benchmark_s1234` stopped after four warmup iterations with no optimizer update and was replaced by `_fixed`. `ppo_verified_eval_150` has partial videos but no completed result. Historical short SFT/PPO/DSRL save/reload/resume and rendering checks remain integration evidence only. No formal PPO + Robometer training/evaluation or prefix-critic/lower-LR PPO pilot has been completed.
- Complete machine-readable inventory and evidence: `outputs/experiment_summary_2026-10-29.json`. Root README now has the result table. This audit used CPU/file/process inspection only: no GPU inference, Ray, training, or evaluation was launched, and no model/checkpoint/config was modified.

### DSRL v2 preparation, October 29 — next unfinished step

- User requested focusing only on DSRL and preparing revised hyperparameters after the flat result. No GPU, Ray, simulator, long training, or evaluation job was started. Existing checkpoints, environments, and legacy recipe files are preserved.
- Fixed DSRL-only optimizer gradient ownership in `fsdp_sac_policy_worker.py`: clear stale actor gradients before critic training, clear critic gradients before actor training and after actor backward, then retain native FSDP clipping. Differentiation through Q to sampled actor actions is preserved; other SAC methods retain their old behavior.
- Added `openpi.dsrl_head_precision: fp32` through `Pi0DSRLConfig` and the model factory. All six DSRL modules use FP32 storage and input precision, giving plain Adam FP32 moments. Legacy default remains BF16; old tensor shapes load by casting. v2 explicitly assigns the six named head wrappers to the FP32 FSDP policy, while frozen decoder precision remains unchanged.
- Added `openpi.dsrl_eval_noise_mode: policy|zero|sample`, affecting evaluation only. `policy` retains deterministic `tanh(mean)`; `zero` bypasses the controller with zero initial noise; `sample` evaluates the stochastic learned controller. Training always samples regardless of this evaluation knob.
- New public configs: `robotwin_adjust_bottle_pi05_dsrl_v2`, `_v2_eval`, `_v2_repair_only`. Shared v2 overrides live in `configs/_robotwin_adjust_bottle_pi05_dsrl_v2.yaml`. Main proposed change is fixed alpha 0.01, replacing automatic alpha initially 1.0 (old final about 0.437). This is a task-specific hypothesis: old entropy was near the bounded 32-D maximum; official LIBERO step penalties have a different scale from this task's sparse +1. Loss scalar magnitude does not prove gradient dominance.
- Keep the experiment otherwise matched: 200 iterations, two environments × two rounds, 800 attempts, horizon 200, action chunk 50, five denoising steps, batch 16/microbatch four, actor LR 1e-4/critic 3e-4, gamma 0.999/tau 0.005, ten-batch replay readiness, 20 SAC rounds per eligible iteration = 3,820 rounds total. Save only final checkpoint 200 to reduce duplicate frozen-weight storage. New output: `outputs/dsrl_sim_v2_alpha001_s1234/pi05_dsrl_v2/checkpoints/global_step_200`. Start fresh from SFT/replay/optimizers; do not resume an old BF16 optimizer.
- **Next user actions:** run the two-iteration smoke from README section 1 (override replay readiness to one and updates to two, enabling four real SAC rounds, no checkpoint/video); run the 150-seed zero-noise SFT decoder control; run main v2 training; run v2's learned-policy 150-seed evaluation. Estimate 35–60 minutes training plus 10–15 minutes for each full evaluation; FP32 timing is unprofiled. Main and zero evaluations disable videos but retain the original capture schedule. Compare with both SFT 122/150 and the zero-noise result. If flat, test sampled learned-policy evaluation before more training. Repair-only arm retains automatic alpha 1.0 with the same code/precision fixes for attributing any effect to entropy; it has not been run.
- **Validation completed:** 30 focused CPU tests across existing `test_models.py` and `test_fsdp.py` passed, including four regressions that failed before the clipping fix. Hydra composed main/control/eval/zero/smoke cases; FSDP config validation passed. The actual FSDP wrap plan on a meta-device full model verifies FP32 storage/computation/reduction for all six heads without GPU allocation. Reports: `outputs/dsrl_debug/v2_config_validation.json`, `v2_fsdp_wrap_validation.json`. Ruff lint/format and whitespace checks passed for changed upstream files. CPU tests exercise tiny fixture optimizer steps only, never a trained checkpoint. Distributed FSDP and simulator behavior remain untested for v2 until the user runs the smoke.
- Upstream overlap search found a related BF16 in-place optimizer report, https://github.com/RLinf/RLinf/issues/1649 (PPO scope, not this DSRL patch); no indexed DSRL-specific duplicate fix was found. GitHub CLI is not installed.

### DSRL v2 completed; next larger-budget test prepared

- The user completed the zero-noise control: 122/150 (81.33%), using SFT weights and `dsrl_eval_noise_mode: zero`. The short v2 GPU smoke passed with finite losses, nonzero gradients, alpha 0.01 and all six head FSDP policies FP32.
- Full v2 then completed 200 iterations in 35:36. Both compact saved ranks have `update_step=3820`; all 188 Adam counters are 3820; all 4,857,994 online head parameters, target heads and Adam moments are FP32/finite, all 376 moment tensors nonzero, online ranks equal. Global batch 16/microbatch four and 20 SAC rounds were retained. Training success averaged 79.5%; final held-out evaluation scored 120/150 (80%), with saved model path pointing to v2 checkpoint 200 and noise mode `policy`. No improvement over SFT/zero control is demonstrated.
- User requested increasing training length, batch and collected episodes. Prepared `configs/robotwin_adjust_bottle_pi05_dsrl_v2_large.yaml`, inheriting the private v2 defaults (public configs cannot inherit other public configs that set Hydra searchpath). Fresh SFT initialization, fixed alpha 0.01 and actor/critic LR 1e-4/3e-4 stay unchanged.
- Larger proposal: 400 outer iterations, two environments × four sequential collection rounds, eight episode attempts/iteration, 3,200 total attempts, horizon 200/chunk 50/five denoising steps, 12,800 stored chunk slots including padding. Global batch 128, 64/rank, microbatch 32/rank, two accumulation microbatches. Keep 20 SAC rounds; min replay and actor threshold 25 starts learning at iteration 25 (200 attempts collected), so 376 eligible iterations produce 7,520 SAC rounds and 962,560 nominal replay sample draws. This is about four times the replay sample reuse per collected transition versus v2; do not simultaneously increase update rounds to 40.
- Replay cache and sampling window 512 trajectory-batch slots/rank cover all 400 iterations; replay remains CPU-resident, around 17.9GB total at completion. Only final checkpoint 400 is saved, around 30GiB including replay and full frozen decoder. Workspace free disk was 148.7GiB when preparing this config. There are 1,000 distinct training reset seeds and 150 disjoint evaluation seeds; larger attempts revisit the training set rather than add 3,200 unique resets.
- Estimated run time 2–4 hours; v2 steady rollout 8.00s, learner 2.14s, sync 0.60s per iteration. New rounds double rollout time; new batch size is not GPU-profiled. User should run the two-iteration scaled-batch smoke first, then full run and 150-seed evaluation using commands in README section 1. Extrapolate after iteration 50. Main output `outputs/dsrl_sim_v2_large_s1234/pi05_dsrl_v2_large/checkpoints/global_step_400`; eval overrides the existing v2 eval config with this model path and a distinct output directory. No larger GPU job was started by the assistant.
- CPU composition and FSDP config validation passed for main/scaled-smoke/evaluation cases. Reports: `outputs/dsrl_debug/large_run_config_validation.json` and `dsrl_v2_large_resolved.yaml`. Existing code is unchanged for this larger proposal. If held-out success remains flat, check fixed-observation decoder action sensitivity to latent noise before spending more on training.

### Larger DSRL result and recommended next move

- User completed the large simulator-reward run and its full evaluation. Training: 400 iterations, 3,200 episode attempts, 12,800 stored chunk slots, 2,645 successes (82.65625%), 1:50:31 loop time. Both final400 ranks have update_step7520, all188 Adam counters7520, all4,857,994 trainable parameters/targets/moments FP32 and finite, all376 moment tensors nonzero, matching online/target replicas and FP32 EMA shadows. All188 trainable tensors differ from small-v2 final200; controller is not stuck.
- Held-out evaluation: **129/150=86.0%**, vs SFT and zero-noise122/150=81.33% (+7 successes/+4.67percentagepoints), smallv2 120/150=80% (+9/+6pp). Both rollout workers loaded large checkpoint400; saved config uses DSRL policy/FP32 heads, same150seeds, raster, horizon200/chunk50/fiveflowsteps. No videos. Source `outputs/dsrl_sim_v2_large_eval_150/{metrics.log,tensorboard/config.yaml}` and `.log` loader lines315–316.
- Training success 50-iteration windows:78,82,77.5,83.5,84.25,85.5,87.5,83percent. First200iterations80.25%,last20085.0625%. Criticloss.13666→.001070,Qpi.1843→.6777,entropy21.6486→21.2774,alpha.01fixed. All scalar logs finite. Loop83.23%rollout,14.02%learner,2.62%sync. Larger batch/experience helped this one run, but changes were combined and repeatability/statistical significance are not established.
- Recommended next: preserve checkpoint/current hyperparameters; add per-seed paired evaluation outcome capture, explicitly seed model/learner/rollout RNGs, then repeat the same large recipe independently before claiming robust improvement. Afterwards run matched Robometer-large at the same3,200-attempt budget. No further architecture/action-magnitude/denoising changes justified now. Current actor.seed only visibly controls replay and critic subsampling; do not describe changing it alone as full model/sampling seeding.
- Per-seed recording hook: RoboTwinEnv._handle_auto_reset BEFORE update_reset_state_ids; completed episode metrics and old requested seed are available. Actual retry-adjusted simulator seed is `self.venv.envs[idx].task.ep_num`; SubEnv.reset passes trial_seed as now_ep_num. Use opt-in separate JSONL files per worker, not numeric episode metric keys. Rerun baseline and learned evaluation with recording to recover paired successes/failures; old aggregate logs cannot recover individual outcomes.
- Additional audit:2,549/12,800 replay chunks(19.914%)occur after success already terminated the episode. All have identicalcurrent/nextjointstate,zeroreward,terminationtrue. CurrentSACsamples them and actor updates include terminal observations. They are absorbing-state padding; performance impact unmeasured. Treat filtering them as a separate learning ablation rather than silently mixing it into an unchanged replication.
- Saved `outputs/dsrl_debug/large_run_analysis_2026-10-29.{json,png}`. This analysis used CPU/files only; no GPU/Ray/training/evaluation was launched, and no production code or hyperparameters changed. README result table and next-step notes updated.

### Post-terminal replay fix and large Robometer preparation — next unfinished step

- User explicitly requested fixing post-success replay padding and preparing the large Robometer config/commands. Implemented opt-in `algorithm.replay_buffer.filter_terminal_padding`. It is false in legacy local defaults to preserve historical recipes, true in the two new filtered large recipes. No GPU/Ray/model-server/training job was started; old checkpoint weights and raw replay files remain unchanged.
- `terminal_padding_sample_indices` in replay/buffer.py derives eligibility through each rollout epoch's first done, retaining that done-containing transition and its reward. Handles native epoch-start boundary rows and legacy post-action flags with explicit chunks_per_rollout_epoch=4; rejects inconsistent geometry. Worker validates DSRL, auto_reset=false, ignore_terminations=false and rollout divisibility. Auto-reset cannot safely use this epoch-based filter.
- Raw rectangular archives/index num_samples/cache offsets/total_samples stay unchanged. Sampling windows weight by eligible counts, map sampled eligible indices back to raw slots for both cache-hit/miss paths, and preserve requested batch size with replacement. New and old checkpoint loading reconstructs eligible maps. Empty eligible windows raise clearly. New stats `valid_samples` and `terminal_padding_samples` are per-rank averages, summing to raw total_samples. No fake terminal-state observation is sampled by actor, critic or alpha updates when enabled.
- Real800-file CPU audit:12800raw→10251eligible,2549excluded;all2645rewardedchunks,3200episode starts and3200firstterminalchunks retained;independent scan matches. Rankvalid counts4974/5277. Audit `outputs/dsrl_debug/terminal_padding_filter_audit.json`; reusable `scripts/audit_dsrl_replay_filter.py`. Existing Robometer reward context already skips HTTP requests for invalid padding while scoring the real terminal chunk; its reward values/placement are unchanged.
- Configs: private shared `_robotwin_adjust_bottle_pi05_dsrl_v2_large.yaml` copies the existing large defaults without public Hydra searchpath; existing unfiltered large file remains unchanged. New simulator control `robotwin_adjust_bottle_pi05_dsrl_v2_large_filtered`; new learned reward primary `robotwin_adjust_bottle_pi05_dsrl_v2_large_robometer`. Both use400iterations,3200attempts,batch128/micro32,20SACupdates,minbuffer/actor25,7520expectedupdates,FP32heads,alpha.01,actor/criticLR1e-4/3e-4,raster,horizon200/chunk50/5flow,cache/window512,final400only,freshSFT/noresume. Baseline86%usedunfilteredreplay,sofairreward-sourcecomparisonneedsfilteredSimcontrol.
- Large Robometer uses reward placement0 for the HTTP bridge; separate GPU1 server at127.0.0.1:8000. env_reward_weight0/modelweight1,progress_delta×1,maxframes8,episodecontexttrue,50-slot last-slot scalar+SACsum. Learns only Robometer; native simulator success/return remain monitoring metrics. Raw signed progress differences do not have the same numerical scale as sparse+1success.
- Outputs: `outputs/robometer_dsrl_v2_large_filtered_s1234/pi05_dsrl_v2_large_robometer/checkpoints/global_step_400`. Simulator filtered control uses `outputs/dsrl_sim_v2_large_filtered_s1234/pi05_dsrl_v2_large_filtered/checkpoints/global_step_400`. Eval uses existing FP32 v2 eval config with explicit model/output overrides. All server/smoke/full/eval commands are in README section3.
- Real Robometer server `/health` was connection-refused at preparation time; user must start it in separate robometer env/GPU1, without inherited LD_PRELOAD/NCCL_P2P_DISABLE. Next user action: startserver,checkhealth,runlargeRobometer2-iterationsmoke withminbuffer/actor1,update2,save-1 (4SACrounds),thenfull400runandeval150. Expectroughly2–3hoursplus10–15mineval,unprofilednewrecipe;finalcheckpoint~30GiB.
- Validation passed:38selected replay/SAC CPU tests,8HTTP reward tests,200independent randomized native/legacy eligibility cases,real800fileaudit,main/smoke/eval Hydra/FSDP validation,extensionregistry,Rufflint/formatandwhitespace. Initial sandbox HTTP test attempt could not bind sockets; permitted host retry passed all8. CPU tests use synthetic fixtures only; no actual policy optimizer step occurred. Reports: `outputs/dsrl_debug/robometer_large_filtered_validation.json`, `{simulator,robometer}_large_filtered_resolved.yaml`. Temporary pytest files from this turn removed after completion.

### Latest OIDN repair

- The user tried the official PPO config. First it failed during FSDP initial weight broadcast, consistent with the known NCCL workaround being absent; the subsequent run passed that stage and failed during the initial simulator camera readback with `cannot create buffer`. It requested 256 train and 128 eval environments and restored native ray tracing/OIDN. OIDN explicitly reported `unsupported device type: CUDA`. Later `ray.kill` messages were cleanup consequences. Rendering resource exhaustion is a plausible cause of the buffer failure, not an explicitly confirmed CUDA OOM. The assistant provided a CPU-composed, two-iteration hardware-scaled startup command; no new PPO training was started by the assistant.
- User then explicitly requested repairing OIDN. SAPIEN 3.0.1 bundled OIDN 2.0.1, which predates Blackwell support; official OIDN 2.3.3 added support. The assistant downloaded the official x86-64 Linux 2.3.3 bundle, reproduced old CUDA device initialization failure (error 5, unsupported CUDA), and confirmed the new bundle initialized CUDA on the same GPU.
- Installed one complete matching native library set at `rlinf/lib/python3.11/site-packages/sapien/oidn_library/lib`. This nested `lib` layout preserves the release libraries' `$ORIGIN/../lib` dependency paths. Updated only the two absolute Linux core/API filenames in SAPIEN's `_oidn_tricks.py`; this loader otherwise forces old 2.0.1 regardless of ordinary library search paths. Torch, NCCL, SAPIEN's Python version, driver, production source/configs and experiments were unchanged. The installed-package patch must be reapplied after a SAPIEN reinstall.
- Backup: `data/oidn/backups/20261028T063837Z/{oidn_library,_oidn_tricks.py}`. Original hashes were verified after repair. Installation source URL, archive hash, new/old hashes and backup paths: `data/oidn/installation.json`. With RLinf stopped, rollback consists of moving the new `oidn_library` aside and restoring the old directory and loader from the backup. Do not remove this backup during cache cleanup.
- `scripts/check_sapien_oidn.py` is a bounded manual diagnostic: no Ray, policy, optimizer or training. It checks direct OIDN CUDA initialization, real SAPIEN ray tracing and denoising, camera readback, finite/nonblank output, physical PCI/device identity and loaded native libraries. On each physical GPU, three 320×240 cameras, 32 spp, depth eight, ten frames and two scene lifecycles passed (60 captures per GPU); an additional GPU 1 check passed nine captures. All 129 verified captures had no OIDN error log, and only 2.3.3 OIDN API/core/CUDA libraries were mapped. Summary: `outputs/oidn_debug/verification.json`; PNGs, per-GPU JSONs and logs are retained. Ruff and whitespace checks passed.
- An initial diagnostic selecting Vulkan GPU 1 without selecting the current CUDA device failed and timed out. Native SAPIEN's OIDN initializer uses `cudaGetDevice`, so explicit Vulkan selection alone can mismatch GPUs. The diagnostic now calls the existing CUDA runtime's `cudaSetDevice` before rendering. Both the RLinf-style `CUDA_VISIBLE_DEVICES=1` / local GPU 0 case and the unmasked physical GPU 1 case then passed. The failed control remains recorded in the verification report; its process exited with timeout code 124 and no test process was left running.
- Current raster/no-denoiser training and evaluation configs are unchanged. The repair validates OIDN/SAPIEN CUDA integration, not long-run RoboTwin ray-tracing stability or capacity for 256 concurrent environments. Keep the existing raster baseline protocol unless deliberately rerunning all policy comparisons with matching ray-traced images. An opt-in ray-tracing launch needs a fresh process, `SAPIEN_DISABLE_RAY_TRACING` unset and `render_shader: rt` / `render_denoiser: oidn` in both train/eval task configs. No Ray session was created for this repair.

### Latest evaluation and PPO diagnosis

- Completed matching raster, 150-seed held-out evaluations: SFT **122/150 (81.33%)**, DSRL + simulator **122/150 (81.33%)**, DSRL + Robometer **121/150 (80.67%)**, final simulator PPO **0/150**. Metrics: `outputs/sft_raster_eval_150/metrics.log`, `outputs/dsrl_sim_raster_eval_150/metrics.log`, `outputs/robometer_dsrl_raster_eval_150/metrics.log`, and `outputs/ppo_sim_raster_eval_150/metrics.log`. Both final DSRL checkpoints and PPO checkpoint 100 were loaded correctly. The 12.5% previously discussed was one training batch, not this full evaluation.
- `outputs/ppo_sim_raster_s1234` completed 100 collection iterations and 1,000 optimizer steps. Training success declined from 77.5% over iterations 1–10 to 2.5% over 91–100. First-iteration optimization already reported large ratios and substantial clipping. The prepared PPO recipe remains unchanged; do not repeat its long commands as a recommended configuration.
- User requested checking probability ratios before an update. New `scripts/check_ppo_logprob_parity.py` uses real retained replay images/states, the actual saved precision/FSDP configuration, all five denoising indices, CUDA PPO reductions, and full initial actor-to-rollout weight synchronization. With unchanged SFT weights, all **40 chunks** had ratio **exactly 1.0**, zero elementwise log-probability differences, zero clipping, and unchanged parameter hashes. Report: `outputs/ppo_debug/logprob_parity_sft_synced/summary.json`; per-rank evidence and log are retained alongside it. No optimizer, backward, training, Ray, or simulator was started. Task prompts were reconstructed identically from the canonical task description because replay retains only tensors. This is a forward parity test, not a test of actual patch transport after an optimizer step.
- The final PPO checkpoint 100 also passed a separate forward-only check: **10 chunks**, all five denoising indices on two ranks, all ratios **exactly 1.0**, zero log-probability differences and unchanged hashes. Maximum critic-value gap was **0.02191534**. Report: `outputs/ppo_debug/logprob_parity_final_synced/summary.json`; log and per-rank data are retained. Both diagnostics exited successfully. The existing large training ratios therefore are not reproduced by same-weight forward replay; update dynamics and actual post-update weight transport remain outside this check.
- The independently loaded, unsynchronized model control is retained under `outputs/ppo_debug/logprob_parity_sft_unsynced_control`. It does not model actual PPO initialization: the real initial sync sends all state, including frozen FP32 precision islands. Its differing ratios must not be reported as evidence of a production probability mismatch. Existing frozen-vision eval behavior predates the failed run; no production model fix was introduced during this diagnostic.
- Separate confirmed critic issue: with `value_after_vlm: false`, rollout averages values across denoising steps while actor replay uses the selected step. The synchronized SFT check measured maximum absolute value difference **0.14852224**, despite identical log probabilities. The official π0.5 introduction recommends `value_after_vlm: true`. That architecture change requires restarting from original SFT with a fresh optimizer; it has not yet been applied. A less aggressive pilot was proposed (32 episode attempts, one PPO epoch, global batch 128, policy LR 1e-6, prefix critic), but remains unverified. No new training has been started.
- OpenPI's generic `recompute_logprobs` switch is unsupported by this checkout's validator. `clip_ratio_c` is not forwarded for OpenPI, and `kl_beta` does not create a reference-policy anchor in this embodied actor. Do not present these as working fixes. Current production configs still describe the failed recipe.

### Latest output cleanup

- The user explicitly requested removal of unused caches and checkpoints under `outputs/`. Active-process arguments and open files were checked before deletion; no training/evaluation jobs or open output files were found.
- Removed four unused checkpoints: assistant PPO smoke `ppo_verified/.../global_step_2` (28.204 GiB), simulator DSRL smoke `dsrl_sim_verified/.../global_step_2` (12.653 GiB), learned-reward DSRL smoke `dsrl_robometer_verified/.../global_step_2` (12.653 GiB), and the formal DSRL intermediate `robometer_dsrl_benchmark_s1234_fixed/.../global_step_100` (14.695 GiB). Five empty failed-save directories were also removed. Total reclaimed: 68.205 GiB; approximately 87.9 GiB free afterward.
- Retained the complete final DSRL checkpoint 200, including both optimizer states and 200 replay records per rank, and the latest user PPO checkpoint `ppo_adjust_bottle/pi05_ppo/checkpoints/global_step_5`. Evaluation results, videos, TensorBoard metrics, audit JSON files and logs were preserved. No ordinary cache files or partial downloads were found under `outputs/`.
- Manifest: `outputs/cleanup_manifest_2026-10-27.json`. The DSRL evaluation config and live README commands now point to the retained final learned-reward checkpoint 200. Historical smoke-check paths below describe past validations; their weights are no longer present.

### Latest PPO configuration preparation

- The user requested matched PPO recipes for simulator rewards and Robometer rewards, with manual launch commands. No PPO GPU training was launched during this preparation, and no environments, packages, submodule revisions, or existing experiments were changed.
- Shared recipe: `configs/_robotwin_adjust_bottle_pi05_ppo.yaml`. Simulator primary: `configs/robotwin_adjust_bottle_pi05_ppo.yaml`. Learned-reward primary: `configs/robotwin_adjust_bottle_pi05_ppo_robometer.yaml`. Both public primaries provide Hydra's upstream search path; they share model, optimizer, environment, and sampling settings.
- Each recipe starts independently from the original SFT policy: 100 outer iterations, two environment workers with one environment each, four collection rounds, eight episode attempts per iteration and 800 in total. Horizon 200, chunk size 50, 32 scheduled chunk slots per iteration, global batch 16, microbatch one, eight accumulation steps per GPU, and five PPO epochs give ten nominal synchronized optimizer steps per iteration. Existing PPO learning rates, offload and FSDP precision settings are preserved. PPO trains the action expert/projections and value head while keeping the VLM frozen.
- Training and evaluation use `render_shader: default`, `render_denoiser: none`, `single_thread_render: true`. Training skips intermediate renders, disables videos, and explicitly keeps `auto_reset: false` and `ignore_terminations: false` so PPO masks terminal padding. PPO's chunk aggregation already sums all 50 reward slots; no SAC aggregation settings are used.
- Simulator run: `outputs/ppo_sim_raster_s1234`, experiment `pi05_ppo`, simulator weight one/model weight zero. Robometer run: `outputs/ppo_robometer_raster_s1234`, experiment `pi05_ppo_robometer`, simulator weight zero/model weight one. The latter reuses the HTTP progress-delta adapter, episode context, and separate GPU-1 reward service; retain `RLINF_EXT_MODULE=icl_reward`.
- Automatic held-out evaluation is disabled and only the final iteration-100 checkpoint is saved. The full PPO evaluation config now defaults to `outputs/ppo_sim_raster_s1234/pi05_ppo/checkpoints/global_step_100`; override this with the learned-reward checkpoint for the second evaluation. It uses the same 150 held-out task seeds, renderer, three cameras, action horizon and five denoising steps, and saves unannotated videos at 20 fps. The latest user PPO checkpoint 5 remains; the old assistant smoke checkpoint 2 was retired in the cleanup above.
- Validation passed without starting Ray or GPU jobs: Hydra composition/resolution of both training recipes, the two-iteration smoke overrides, and both evaluation commands; actual RLinf FSDP validation; matching training settings except output paths and rewards; 150 unique evaluation seeds disjoint from training; and a synthetic CPU flow through actual episode tracking, Robometer reward placement and PPO aggregation checking positive/negative deltas, terminal reward and padding mask. This does not constitute a GPU integration test of PPO + Robometer. README provides a two-iteration learned-reward smoke command with checkpoint saving disabled.
- Disk space was only about 20 GiB when preparing these configs. An existing PPO checkpoint occupies about 29 GiB; allow at least 60 GiB of additional space for the two final checkpoints plus video headroom before launching both full experiments. No existing checkpoints were deleted.

### Latest runtime repair and cleanup

- The user started the 200-iteration experiment at 17:18 Shanghai time. It completed four replay-warmup iterations and then froze in `camera.get_picture("Color")`; the native stack was waiting inside `libnvidia-glcore.so.580.178.04`. No SAC updates had completed. The user stopped that run before repair; its files are preserved under `outputs/robometer_dsrl_benchmark_s1234` and the corresponding `.log`.
- The initial thread-only fix completed 20 training iterations and 220 SAC updates, but a subsequent full-motion evaluation still froze on GPU 1. Therefore thread handling alone is insufficient for this workstation's ray-tracing path.
- The final fix uses `task_config.single_thread_render: true` and `render_shader: default` in both DSRL training and standalone evaluation. `VectorEnv` disables native ray tracing before seed-probe cache access. Scene creation, reset, camera reads, cache clearing, and teardown use the existing persistent single-worker executor. This requires one environment per worker; the tested two-GPU total remains two. Default behavior for other configs is retained. OIDN stays disabled and cache-clearing frequency stays one; no package or driver upgrade was performed.
- Final raster regression passed: 20 collection iterations, 80 episode attempts, 220 SAC updates, 309 seconds for the training loop, all 762 scalar observations finite. It started with the disable flag unset in the parent process, verifying that config propagation disables ray tracing early enough. No large test checkpoint was saved. Audit `outputs/dsrl_debug/render_raster_verification.json`; log `outputs/dsrl_debug/render_raster_training_check.log`.
- Full-motion raster evaluation completed four episodes with three successes, including auto-reset, and saved four videos at 20 fps under `outputs/render_raster_video_check/video/eval`. Log `outputs/dsrl_debug/render_raster_video_check.log`. The earlier 2/2 ray-traced evaluations and these raster scores have different rendering; use matching raster settings for all new SFT and trained-policy comparisons.
- Five CPU contract tests pass in `tests/test_robotwin_rendering.py`; they cover resource-thread ownership across callers, teardown/recreation, reentrancy, unchanged default dispatch, and early shader validation/configuration.
- At the user's request, inactive Ray sessions, assistant temporary scripts/probes/stack dumps, abandoned download pieces, obsolete setup scratch, and pip/uv download caches were removed. About 14 GiB of physical disk space was recovered; installed environments, source, model checkpoints, assets, runtime NCCL library, and user experiments were preserved. User-specific pip/uv cache entries can be downloaded again if needed. Future diagnostics should also remove inactive Ray sessions after finishing.
- After final verification, all five inactive Ray sessions from the stalled run and repair were removed; `/tmp/ray` is empty. Render-test fixtures and partial failed-test outputs were removed. Final logs, the JSON audit, training metrics, and four evaluation videos are retained. Robometer health returned HTTP 200, no training/evaluation process remained, and approximately 94.6 GiB of disk space was available at the final check.
- Subsequent user run: `outputs/robometer_dsrl_benchmark_s1234_fixed` completed all 200 iterations at 19:06, with checkpoints 100 and 200 under `pi05_dsrl_robometer/checkpoints`. The user received paired SFT/DSRL 150-seed evaluation commands; do not assume those evaluations completed without checking outputs. The reward server may have been stopped for evaluation; verify `/health` before learned-reward training. Raster images differ from ray tracing, so report the renderer in performance results. Switching back to ray tracing requires a fresh process and unsetting `SAPIEN_DISABLE_RAY_TRACING`.

### Installed environments and machine

- `/home/yifei/icr/rlinf`: Python 3.11.15, Torch 2.7.1+cu128, runtime Transformers fork 4.53.2, SAPIEN 3.0.1, MPLib 0.2.1, Ray 2.59.0. Preserve existing installed patches.
- `/home/yifei/icr/robometer`: Python 3.10.20, Torch 2.8.0+cu128, Transformers 4.57.6, PEFT 0.17.1. Inference requirements: `requirements/robometer-inference.txt`; installed freeze: `data/robometer/requirements-inference.lock.txt`. Training-only Robometer dependencies are intentionally omitted. Editable upstream installation uses `--no-deps`.
- Parent package `icl-reward` is installed editable, without dependencies, in `rlinf`. Export `RLINF_EXT_MODULE=icl_reward` to register the HTTP reward model in every Ray process.
- Two RTX PRO 6000 Blackwell Max-Q GPUs, about 96 GiB each, driver 580.178.04; 48 CPU threads, about 503 GiB RAM. Other users' GPU jobs were left untouched. GPU 0 had an unrelated approximately 33 GiB process and GPU 1 another approximately 3.5 GiB process during checks.
- Required host-specific exports: `ROBOT_PLATFORM=ALOHA`, explicit NVIDIA Vulkan ICD, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, and job-scoped `NCCL_P2P_DISABLE=1 LD_PRELOAD=/home/yifei/icr/data/nccl/2.31.2/nvidia/nccl/lib/libnccl.so.2`. Installed Torch's NCCL 2.26.2 crashes on this machine; the verified newer library is extracted separately. Localhost must bypass HTTP proxies. Copy the complete exports from README.
- Native GPU/process/network inspection and GPU jobs require host execution outside the sandbox. Do not run machine-wide `ray stop` or broad `pkill` on this shared machine.

### Assets and pinned sources

- RLinf `8fc378bceb3e761442e9bd0936a682455b4d35b8`, RoboTwin RLinf_support `dca9ec682688821c42944eacd7f5dd9bfb15396f`, Robometer `352d160389daa964788de1ec933d1925f3a6de4f`. Local source changes are uncommitted; preserve them.
- Simulator objects/embodiments are already downloaded, verified, and extracted. Random backgrounds are disabled; do not restart the full asset script.
- SFT: `checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle`, verified HF revision `fa8df6ed103db0f5549c122f3a17c00ba6426c98`. Original duplicate remains under `checkpoints/pi05-adjust-bottle-sft`. Normalization: `physical-intelligence/robotwin/norm_stats.json`.
- Robometer original checkpoint: `checkpoints/Robometer-4B`, HF revision `beef63bc914c5c189329d49c6d712d96d632aa34`; Qwen/Qwen3-VL-4B-Instruct base revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. All downloaded files were hash verified; `data/robometer/download_verification.json` records evidence.
- `checkpoints/Robometer-4B-inference` is the local inference view using the original weights, local pinned Qwen base, and `use_unsloth=false`; original downloaded configuration is preserved.

### Implemented DSRL path

- Shared recipe: `configs/_robotwin_adjust_bottle_pi05_dsrl.yaml`; simulator primary: `configs/robotwin_adjust_bottle_pi05_dsrl.yaml`; learned-reward primary: `configs/robotwin_adjust_bottle_pi05_dsrl_robometer.yaml`; standalone evaluation: `configs/robotwin_adjust_bottle_pi05_dsrl_eval.yaml`.
- Native upstream DSRL/SAC uses actual RoboTwin state dimension 14, latent noise dimension 32, 50 executed actions per chunk, three base-policy cameras, ten Q heads. Default pilot: 20 collection iterations, two environments, two collection rounds, global batch 16, microbatch 4, two SAC updates per eligible iteration, evaluation/save every five iterations. Replay warmup counts received trajectory batches, not episodes.
- Reward alignment fix: `algorithm.sac_reward_aggregation: sum` includes RoboTwin's last-slot success and sums learned reward exactly once. Upstream default `first` remains for existing LIBERO configurations.
- Distributed correctness fix: Gaussian noise sampling now traverses the module's normal forward call with tensor outputs, so FSDP hooks synchronize steering gradients. Calling `.sample()` directly had bypassed these hooks; pre-fix diagnostic checkpoints were retired after fresh checks passed.
- SAC target EMA shadows cover only trainable parameters, retain FP32 precision, and save/restore together with `update_step`. Opt-in compact checkpoints store one full self-contained inference policy plus per-rank trainable online/target heads, Adam state, schedulers, RNG, alpha, and replay. This requires the configured two-rank `no_shard` strategy and is approximately 12.65 GiB per checkpoint. Replay sampling RNG is reinitialized by upstream replay loading, so resume is not promised to be bitwise identical.
- RoboTwin `render_denoiser` is configurable and set to `none` in new DSRL configs; other configs retain upstream `oidn`. This removes the OIDN invalid-handle errors in these checks; it does not prove every prior intermittent rendering hang is resolved.
- Existing PPO fixes and user edits are retained. Parent PPO configs are unchanged.

### Robometer reward contract

- `src/icl_reward` contains the native multipart NumPy HTTP client, reward model extension, and causal offline video scorer. RLinf does not import the Robometer model or Torch 2.8.
- Official server endpoint `/evaluate_batch_npy` receives uint8 RGB head-camera prefixes and task instructions. Reset frames establish a baseline. Per-step reward is `progress(current_prefix) - progress(previous_prefix)`, including negative differences; at most eight real prefix frames are sampled.
- Prefix batches are grouped by exact frame shape because native Robometer stacks equal-length per-frame outputs. No synthetic padded frames are added.
- Stable environment IDs, independently reset histories, true terminal observations, initial-frame snapshots, and valid-transition flags are carried by opt-in `RewardEpisodeContext` in EnvWorker. Post-terminal padding scores zero without HTTP calls. HTTP failures raise instead of silently returning rewards.
- A scalar reward occupies only slot 49 of each 50-action chunk. Learned configuration uses `env_reward_weight: 0.0`, `reward_weight: 1.0`; simulator success is solely a diagnostic/evaluation metric.
- Reward diagnostics sum chunk slots before averaging and exclude prefix samples from episode counts. Training rewards themselves are unaffected by that metric fix.
- Robometer has one small optional-Unsloth import patch. Transformers 4.57.6 fixes an actual processor-loading bug encountered with 4.57.2. No RLinf package upgrade was made.

### Verified experiments and evidence

1. Fresh simulator DSRL: `outputs/dsrl_sim_verified/pi05_dsrl/checkpoints/global_step_2`. Eight successful training attempts over two iterations, first replay warmup followed by two SAC optimizer updates; two monitoring episodes succeeded. Audit: `outputs/dsrl_debug/simulator_fixed_audit.json`; log: `outputs/dsrl_debug/simulator_smoke_fixed.log`.
2. Offline real Robometer: final progress 0.684 for a successful video versus 0.294 for a simulator-verified hold-position failure; warm requests approximately 40–70 ms, first approximately 0.39 s. Results: `outputs/robometer_offline/{success_scores,hold_control_scores,batch_contract}.json`. These two trajectories do not establish calibration or policy improvement. Only unannotated RGB videos should be scored.
3. Combined learned-reward DSRL: `outputs/dsrl_robometer_verified/pi05_dsrl_robometer/checkpoints/global_step_2`. Two collection iterations and two SAC optimizer updates completed; two monitoring episodes succeeded. Log: `outputs/dsrl_debug/robometer_smoke_fixed.log`; audit: `outputs/dsrl_debug/robometer_fixed_audit.json`.
4. Both fresh checkpoints: all 855 saved tensors finite; all 667 frozen π0.5 tensors exactly unchanged from SFT; all 188 online and 188 target tensors exactly match across ranks; every optimizer step counter is two with nonzero finite moments. Combined replay contains 32 chunk transitions, 24 nonzero fractional learned rewards, and no nonzero values before slot 49. Config audit confirms simulator reward weight zero.
5. Combined checkpoint reload: `outputs/dsrl_robometer_verified_reload`, log `outputs/dsrl_debug/robometer_checkpoint_reload.log`. Native `openpi[dsrl]` loaded learned steering plus the frozen decoder. Both episodes succeeded. Videos: `video/eval/seed_0/0.mp4` and `seed_1/0.mp4`, 202 frames at 20 fps, 10.1 seconds each, no text overlay.
6. Fresh optimizer/replay resume also passed: `outputs/dsrl_robometer_resume_check`, log `outputs/dsrl_debug/robometer_resume_check.log`. Loaded the learned checkpoint at step two, completed iteration three and two more SAC updates, with finite losses and 3/4 successful training episodes. The corrected diagnostics report four episodes and mean valid chunk reward 0.16335. No additional checkpoint was saved. Upstream replay sampling RNG restarts on resume.
7. Thirty-seven focused tests passed: 15 SAC/reward/checkpoint tests, seven model tests (including two existing PPO cases), two episode-context tests, eight actual-localhost HTTP contract tests, five metric tests including two new episode-count regressions. HTTP tests need host localhost socket access; a sandbox-only run failed at socket creation, then all eight passed outside the sandbox. Ruff and whitespace checks pass on modified files.
8. `scripts/audit_dsrl_checkpoint.py` is a reusable CPU read-only audit. Add `--require-robometer-reward` for learned-checkpoint replay/config checks.

Short simulator and learned runs used `runner.max_steps=2 runner.save_interval=2 runner.val_check_interval=2` and separate output directories. Old assistant-created pre-fix 63 GiB and 13 GiB checkpoints were removed after replacement audits passed; logs and raw scoring videos were preserved. User PPO outputs, original SFT assets, and all current checkpoints are retained. Approximately 106 GiB disk space remained after cleanup.

### Server and next action

- The official Robometer server is available on `http://127.0.0.1:8000`, GPU 1. The user restarted it as PID 1672329 and its current logs print in that terminal; `outputs/robometer/server.log` is from the earlier service. Check `/health` before launching another server. It uses the separate `robometer` interpreter and no RLinf NCCL preload. README contains the foreground start command.
- No long training job should remain after the repair. The next experiment is the 200-iteration learned-reward run from README, followed by the same 150 evaluation seeds for the learned policy, simulator-reward DSRL baseline, and SFT baseline. Short smoke success does not establish DSRL improvement.
- The standalone DSRL evaluation defaults to `outputs/robometer_dsrl_benchmark_s1234_fixed/pi05_dsrl_robometer/checkpoints/global_step_200`, the retained final Robometer-trained policy. Evaluation saves full-motion videos and does not update the policy or require the reward server. Keep `openpi.task=dsrl` and choose a fresh log path when repeating an evaluation.
- For a new ICL reward backend, preserve this episode/reward contract and replace the scoring service after matched baseline results are collected.

### Historical PPO result retained

The earlier PPO smoke checkpoint was `outputs/ppo_verified/pi05_ppo/checkpoints/global_step_2`; its weights were retired during cleanup. Audit `outputs/ppo_debug/checkpoint_verification.json` and successful two-episode reload `outputs/ppo_verified_reload` remain. The user's latest earlier PPO checkpoint `outputs/ppo_adjust_bottle/pi05_ppo/checkpoints/global_step_5` is retained. Full PPO evaluation config: `configs/robotwin_adjust_bottle_pi05_full_eval.yaml`. Check actual user outputs before claiming completion of either newly prepared PPO experiment.

Sections below retain initial architecture and historical setup/research notes. Treat old unimplemented-status statements as superseded by the current evidence above.

## 2. Agreed architecture

Use a small parent repository with these dependencies:

| Component | Repository / source | Responsibility |
|---|---|---|
| RL framework | https://github.com/RLinf/RLinf | DSRL/SAC noise steering of frozen π0.5, rollout workers, replay, checkpointing, evaluation |
| Simulator | https://github.com/RoboTwin-Platform/RoboTwin | RoboTwin tasks and assets; use the `RLinf_support` integration |
| Baseline reward | https://github.com/robometer/robometer | Robometer inference in a separate environment |
| New reward model | Parent-owned code | ICL reward model and support-example handling |

Let the pinned RLinf setup own its compatible OpenPI implementation and dependencies. Do not independently install an arbitrary upstream OpenPI checkout into the same runtime.

The intended structure is:

```text
icl-reward/
  README.md
  pyproject.toml
  .gitmodules
  .gitignore
  third_party/
    rlinf/                    # pinned submodule
    robotwin/                 # pinned RLinf_support revision
    robometer/                # pinned submodule
  configs/
    robotwin_pi05_ppo.yaml
    robotwin_pi05_eval.yaml
    reward/                   # later: environment, robometer, icl
    splits/                   # later: train, eval, support manifests
  src/icl_reward/
    rewards/
      interface.py
      robometer.py
      icl.py
      shaping.py
    context.py
    integrations/
      rlinf_reward.py
  scripts/                    # thin launchers only when useful
  tests/                      # later: reward alignment/reset tests
  checkpoints/                # ignored by Git
  data/                       # ignored by Git
  outputs/                    # ignored by Git
```

Only the initial directories/submodules may exist. The reward package, scripts, and tests above are a design, not completed implementations. No custom reward code is required for the initial simulator-reward run.

Parent code owns experiment configurations and reward research. RLinf owns policy training and rollout infrastructure. If a small upstream integration change is needed later, commit it in a fork and pin that commit instead of leaving untracked modifications inside a submodule.

## 3. Completed step and dependency baseline

- [x] User reports completion of step 1: create parent repository and add submodules.
- [ ] Locally inspect the exact state and confirm which parts of that step are present.

The previously suggested step 1 used:

```text
third_party/rlinf     -> RLinf/RLinf
third_party/robotwin  -> RoboTwin-Platform/RoboTwin, RLinf_support
third_party/robometer -> robometer/robometer
```

The RLinf revision inspected for the instructions is:

```text
8fc378bceb3e761442e9bd0936a682455b4d35b8
```

Verify the local revision before relying on exact paths or configuration fields. Other submodule SHAs were not confirmed on the user's machine. Keep the versions already recorded in the parent repository; reconcile any differences intentionally.

The parent `.gitignore` was intended to exclude `/checkpoints/`, `/data/`, `/outputs/`, `__pycache__/`, `*.pyc`, and `.venv*/`.

## 4. First experiment: fixed choices

| Item | Initial choice |
|---|---|
| Task | `adjust_bottle` |
| Policy | RoboTwin-finetuned π0.5 SFT checkpoint |
| Checkpoint | `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` |
| Algorithm | RLinf DSRL/SAC; RoboTwin configuration and reward alignment need implementation |
| Training reward | Simulator-derived sparse success reward |
| Embodiment | `aloha-agilex` |
| Scene protocol | Clean, with the recipe's domain randomization disabled |
| Policy cameras | Head plus both wrist cameras |
| Action dimension | 14 |
| Executed action chunk | 50 actions |
| Episode / rollout limit | 200 action steps |
| OpenPI configuration | `pi05_aloha_robotwin` |
| Training task mode | `actor.model.openpi.task: dsrl` |
| Standalone evaluation mode | `rollout.model.openpi.task: eval` |

Preserve the checkpoint's transforms, camera ordering, embodiment, action conventions, and normalization. The model's network action horizon and the number of executed actions before replanning are separate concepts; do not conflate them when importing other checkpoints later.

Upstream configurations at the inspected revision:

```text
examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml
evaluations/robotwin/robotwin_adjust_bottle_openpi_pi05_eval.yaml
examples/embodiment/config/env/robotwin_adjust_bottle.yaml
```

The checkpoint includes the normalization statistics at:

```text
physical-intelligence/robotwin/norm_stats.json
```

At the inspected revision, RLinf's loader handles the recommended SFT checkpoint, including its legacy tensor names and sharded safetensors. No separate manual weight conversion was required by the inspected loader. Recheck if using a different revision or checkpoint.

**Do not download the demonstration dataset for this initial experiment.** Online RL starts from the SFT checkpoint and collects new trajectories in simulation. Simulator assets are required. Demonstrations become relevant for further SFT, offline experiments, or ICL support sets.

## 5. Local preflight — next action

Inspection completed on 2026-10-26:

- Project root: `/home/yifei/icr`; parent branch `main`; all three submodules clean.
- RLinf: `8fc378bceb3e761442e9bd0936a682455b4d35b8`.
- RoboTwin `RLinf_support`: `dca9ec682688821c42944eacd7f5dd9bfb15396f`.
- Robometer: `352d160389daa964788de1ec933d1925f3a6de4f`.
- Ubuntu 24.04.4; Threadripper PRO 9965WX (24 cores / 48 threads); 503 GiB RAM.
- About 256 GiB free disk at inspection time.
- The user confirmed two RTX PRO 6000 Blackwell Max-Q GPUs, each reporting 97,887 MiB, driver 580.178.04. Agent sandbox lacks `/dev/nvidia*`; its failed `nvidia-smi` is not a host driver failure.
- uv 0.11.14; uv-managed Python 3.11.15 already available. Active shell was Conda base Python 3.13.13.
- CUDA 12.8 and 13.2 installed; `/usr/local/cuda-12.8/bin/nvcc` works. Compiler was missing from PATH. Docker and CMake were unavailable on PATH.
- No RLinf venv, downloaded simulator assets, SFT weights, or parent experiment configs exist yet. The existing `assets/` contains tracked placeholder files only.

The next user action is repairing the installed Torch CUDA build, then checking GPU execution. No installation has been executed by the assistant. Historical preflight commands follow for future rechecks.

From the actual parent repository, inspect:

```bash
pwd
git status --short
git submodule status --recursive
git -C third_party/rlinf rev-parse HEAD
git -C third_party/robotwin rev-parse HEAD
git -C third_party/robometer rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h .
free -h
```

Also inspect the OS, Python environments, Docker availability, and GPU runtime support as appropriate. Record GPU count and VRAM before choosing worker placement or claiming that full-policy PPO will fit.

The stock recipe uses eight GPUs. Fewer GPUs may satisfy batch divisibility while failing memory requirements because parameter/optimizer sharding is reduced. Do not infer that a one-GPU inference run guarantees one-GPU training feasibility.

Do not modify host GPU drivers or system CUDA merely to follow this handoff. Diagnose actual dependency/runtime errors first.

## 6. Set up one compatible local runtime

### Route A: local Docker (previously proposed)

Run on the host from the parent repository:

```bash
export PROJECT_ROOT="$(pwd)"
docker run -it \
  --name icl-reward \
  --gpus all \
  --shm-size 32g \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v "$PROJECT_ROOT:/workspace/icl-reward" \
  -w /workspace/icl-reward \
  rlinf/rlinf:agentic-rlinf0.4-robotwin \
  bash
```

If the container already exists, reuse it instead of recreating it. Inside it:

```bash
source switch_env openpi
export PROJECT_ROOT=/workspace/icl-reward
```

The image is the one documented by RLinf. Compatibility between this image, the inspected source pin, and the user's hardware has not been runtime-tested. Check imports and the pinned requirements if a mismatch appears; do not blindly upgrade all packages.

### Route B: native local uv environment — selected

The user completed the official installation into `/home/yifei/icr/rlinf`. Activate it with `source /home/yifei/icr/rlinf/bin/activate`. No custom setup/activation helper is used. Robometer requires Python 3.10 / Torch 2.8.0 and will use a separate environment later.

The current installed Torch wheel is `2.7.1+cu126`; use the current-continuation instructions above to select CUDA 12.8 explicitly. The selected upstream `adjust_bottle` recipe uses MPLib and RGB observations (`pointcloud: false`), so optional CuRobo/PyTorch3D runtime paths are not needed for this first task. FlashAttention is not called by the selected local π0.5 implementation.

The failed custom-launcher proposal is historical. Its ignored preparation artifacts may remain under `outputs/setup/rlinf_uv`, but they do not describe the user's installed environment. The assistant has run no package installation or GPU workload.

### Common paths after activating the chosen runtime

```bash
export REPO_PATH="$PROJECT_ROOT/third_party/rlinf"
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export ROBOTWIN_PATH="$PROJECT_ROOT/third_party/robotwin"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$REPO_PATH:$ROBOTWIN_PATH:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export POLICY_DIR="$PROJECT_ROOT/checkpoints/pi05-adjust-bottle-sft"
export NORM_STATS="$POLICY_DIR/physical-intelligence/robotwin/norm_stats.json"
```

These shell variables must be available in each new working shell. Host and container absolute paths are different; do not copy `/workspace/icl-reward` into a native installation unless that is the actual local path.

Check the active implementation and CUDA:

```bash
python -c "import torch, rlinf; print(rlinf.__file__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## 7. Download assets and policy, then prepare configurations

```bash
cd "$ROBOTWIN_PATH"
bash script/_download_assets.sh

hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
  --local-dir "$POLICY_DIR"

test -f "$NORM_STATS" && echo "Normalization statistics found"
```

Inspect existing assets/downloads before repeating expensive transfers. Record the checkpoint revision used. The asset script also configures embodiment paths; run it in the runtime/path where the simulator will operate.

**`env.train.assets_path` and `env.eval.assets_path` must be the RoboTwin repository directory containing `assets/`, not the `assets/` directory itself.**

Create parent-owned configuration copies if they do not already exist:

```bash
cp "$EMBODIED_PATH/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml" \
   "$PROJECT_ROOT/configs/robotwin_pi05_ppo.yaml"
cp "$REPO_PATH/evaluations/robotwin/robotwin_adjust_bottle_openpi_pi05_eval.yaml" \
   "$PROJECT_ROOT/configs/robotwin_pi05_eval.yaml"
```

Do not overwrite a user-edited configuration. The copied files retain their Hydra search path into the upstream configuration directory via `EMBODIED_PATH`.

Adjust `cluster.component_placement` to the hardware selected for the experiment. Stock training:

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor, env, rollout: 0-7
```

Stock standalone evaluation:

```yaml
cluster:
  num_nodes: 1
  component_placement:
    env, rollout: 0-7
```

For one, two, four, or eight GPUs, the placement values are respectively `0`, `0-1`, `0-3`, or `0-7`. This is a worker-placement mapping, not a memory guarantee. Recheck batch arithmetic for other worker counts.

## 8. Run the unchanged SFT policy

With the runtime activated and paths exported:

```bash
cd "$REPO_PATH"
python evaluations/eval_embodied_agent.py \
  --config-path "$PROJECT_ROOT/configs" \
  --config-name robotwin_pi05_eval \
  rollout.model.model_path="$POLICY_DIR" \
  ++rollout.model.openpi_data.norm_stats_path="$NORM_STATS" \
  env.eval.assets_path="$ROBOTWIN_PATH" \
  env.eval.total_num_envs=8 \
  runner.logger.log_path="$PROJECT_ROOT/outputs/sft_smoke" \
  runner.logger.experiment_name=pi05_sft
```

Acceptance criteria:

- Model and normalization load successfully.
- Simulator launches and renders plausible head/wrist observations.
- Both arms move plausibly under the task instruction.
- Evaluation completes with simulator success metrics and viewable videos.

If policy behavior is wrong, resolve checkpoint/transforms/normalization/camera issues before adding PPO or Robometer. Eight episodes are only a functional check.

## 9. Run two PPO updates with simulator rewards

Use the Python entry point directly. **At the inspected revision, `run_embodiment.sh` does not forward arbitrary trailing Hydra overrides.** Its second argument is the robot platform. Do not append path/batch overrides to that shell wrapper and assume they are applied.

```bash
python "$EMBODIED_PATH/train_embodied_agent.py" \
  --config-path "$PROJECT_ROOT/configs" \
  --config-name robotwin_pi05_ppo \
  actor.model.model_path="$POLICY_DIR" \
  ++actor.model.openpi_data.norm_stats_path="$NORM_STATS" \
  env.train.assets_path="$ROBOTWIN_PATH" \
  env.eval.assets_path="$ROBOTWIN_PATH" \
  env.train.total_num_envs=8 \
  env.train.rollout_epoch=1 \
  env.train.video_cfg.save_video=false \
  env.eval.total_num_envs=8 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=32 \
  algorithm.update_epoch=1 \
  reward.use_reward_model=false \
  runner.max_steps=2 \
  runner.val_check_interval=2 \
  runner.save_interval=2 \
  runner.logger.log_path="$PROJECT_ROOT/outputs/smoke" \
  runner.logger.experiment_name=pi05_sim
```

Retain the inherited environment settings `use_custom_reward: true` and `use_rel_reward: true`. In the inspected code these compute changes in a simulator-success signal. This is a **sparse success-reward baseline**, not dense visual reward. Setting `use_custom_reward: false` alone does not establish a correct dense per-step reward path; the chunk wrapper's assignment logic must also be considered.

Smoke-run arithmetic: 8 environments × (200 action steps / 50 actions per chunk) × 1 rollout epoch = 32 chunk transitions per iteration. With eight actor ranks and microbatch 1, global batch 32 gives four accumulation steps. There is one PPO update epoch and two training iterations. Other settings must still satisfy RLinf's validator and the machine's memory limits.

Acceptance criteria:

- Rollout collection and both optimization iterations finish.
- Training losses/gradients are finite and policy updates occur.
- Evaluation finishes.
- A checkpoint is written, including full model weights and resumable training state.

Expected checkpoint directory:

```text
outputs/smoke/pi05_sim/checkpoints/global_step_2/
  actor/
    dcp_checkpoint/
    model_state_dict/full_weights.pt
```

## 10. Reload, evaluate, and optionally test resume

```bash
export RL_CHECKPOINT="$PROJECT_ROOT/outputs/smoke/pi05_sim/checkpoints/global_step_2"
python "$REPO_PATH/evaluations/eval_embodied_agent.py" \
  --config-path "$PROJECT_ROOT/configs" \
  --config-name robotwin_pi05_eval \
  rollout.model.model_path="$RL_CHECKPOINT" \
  ++rollout.model.openpi_data.norm_stats_path="$NORM_STATS" \
  env.eval.assets_path="$ROBOTWIN_PATH" \
  env.eval.total_num_envs=8 \
  runner.logger.log_path="$PROJECT_ROOT/outputs/rl_smoke" \
  runner.logger.experiment_name=pi05_rl
```

Supply the **original SFT normalization path** explicitly when loading RL weights. The weights alone do not include those statistics. At the inspected revision, the OpenPI model loader accepts the outer global-step directory and resolves the full weights. Prefer this loading route over a separate strict `runner.ckpt_path` override when using the standalone evaluation model.

Successful reload/evaluation completes the initial end-to-end milestone. Two updates do not demonstrate meaningful policy improvement.

For a resume check, repeat the same smoke training command, replace `runner.max_steps=2` with `runner.max_steps=4`, and add:

```text
runner.resume_dir=<absolute path to outputs/smoke/pi05_sim/checkpoints/global_step_2>
```

Resume uses the outer global-step directory and restores training state. Loading only `full_weights.pt` as initialization is not equivalent to restoring optimizer state.

## 11. Scale the simulator-reward baseline

After all smoke milestones pass, choose a larger configuration appropriate to measured memory and throughput. For the stock eight-GPU recipe, the original settings are:

| Setting | Stock value |
|---|---:|
| Training environments | 256 |
| Training rollout epochs | 4 |
| Evaluation environments | 128 |
| Actor microbatch | 32 |
| Actor global batch | 2048 |
| PPO update epochs | 5 |

An initial budget discussed was 100 training iterations with evaluation/checkpointing every 10. It is a proposed budget, not a demonstrated convergence requirement. Start the substantive baseline from the same original SFT checkpoint and use a new output directory. Do not accidentally resume the smoke run when intending a fresh baseline.

Evaluate SFT and RL checkpoints on identical seed lists, scene settings, camera inputs, action chunks, and episode limits. Log success rate, environment interactions, wall-clock time, and the exact resolved configuration/revisions. Eight evaluation episodes are insufficient for a research result. The inspected `adjust_bottle` evaluation seed file contains 150 seeds; the stock 128 fixed parallel environments cover a subset. Choose and report explicit coverage rather than claiming the full benchmark was evaluated.

## 12. Robometer and ICL integration — later work

This work is not implemented yet. The first simulator-reward milestone does not depend on it.

1. Run Robometer in a separate environment using its HTTP inference service.
2. Collect and save a small set of successful and failed RoboTwin rollouts with instructions and simulator outcomes.
3. Score the same saved trajectories and inspect reward curves before online training.
4. Implement the parent-owned Robometer client and RLinf reward adapter.
5. Repeat the two-update/save/reload sequence using only learned rewards.
6. Run a comparison with the same initial SFT policy and interaction budget.
7. Add the ICL backend behind the same scoring interface.

Proposed project-level scoring contract (not an upstream API):

```python
score_prefixes(instruction, frames, end_indices, context=None)
```

Return scores aligned with the requested prefix endpoints, with explicit score semantics and optional success probabilities. The ICL backend uses `context` for support examples; Robometer ignores it. Keep score-to-reward transformation in `shaping.py`, separate from model inference.

Important integration requirements:

- Robometer has `/evaluate_batch_npy`; it is not automatically compatible with RLinf's generic API reward worker. A bridge is required.
- Keep the reward inference model fixed during the initial policy-training comparison.
- Score causal prefixes: a reward at a timestep must not silently use future query observations.
- Record camera choice, sampling frequency, frame selection, context length, checkpoint revision, and score-to-reward transformation.
- Begin with an explicit chunk-endpoint reward assignment. Do not accidentally broadcast one chunk score into 50 repeated rewards.
- Handle episode reset, true terminal observations, truncated episodes, and padded/invalid actions correctly.
- For a pure learned-reward experiment, disable the simulator's contribution to training reward and any hidden ground-truth success bonus. Keep simulator success as an independent evaluation metric.
- Keep horizon and termination protocol fixed across reward models.
- ICL support examples must come from the designated training/support split; record their IDs and exclude evaluation trajectories.
- Switching `reward.use_reward_model` on alone is not a complete Robometer integration.

The Robometer policy-learning repository is a reference, not the selected trainer:
https://github.com/robometer/robometer-policy-learning

The current experiment is a controlled **π0.5/DSRL reward-model comparison in RoboTwin using RLinf**. Keep the DSRL implementation, frozen initial policy, simulator protocol, and interaction budget fixed across reward models. The Robometer policy-learning repository is a source reference; its supplied environment/checkpoint integration does not include RoboTwin.

## 13. Working checklist

- [x] Step 1 reported complete by user: parent repository and submodules.
- [x] Inspect actual repository, clean/modified state, and submodule pins.
- [x] Record OS, GPU model/count/VRAM, RAM, and free disk.
- [x] User installed the native uv runtime at `/home/yifei/icr/rlinf`.
- [x] Replace installed CUDA 12.6 Torch wheel with CUDA 12.8 and verify GPU execution.
- [x] Confirm the active Python imports the intended RLinf checkout and sees CUDA.
- [x] Download/configure RoboTwin assets.
- [x] Download the recommended π0.5 SFT checkpoint and verify normalization.
- [x] Prepare parent-owned PPO training/evaluation configs and hardware placement.
- [x] Complete two-episode SFT evaluation and inspect videos.
- [x] Complete two PPO updates and verify checkpoint creation, weights and optimizer state.
- [x] Reload and evaluate the PPO checkpoint on two episodes.
- [ ] User-run full 150-episode PPO evaluation.
- [ ] Port the π0.5 DSRL configuration to RoboTwin and fix/test chunk reward alignment.
- [ ] Complete a short DSRL run with simulator success rewards.
- [ ] Verify resume if needed.
- [x] Record the working commands and dependency versions.
- [ ] Run a larger simulator-reward baseline and a matched SFT evaluation.
- [ ] Integrate Robometer, verify scores offline, and run the same RL pipeline.
- [ ] Implement the replaceable ICL reward backend and controlled comparison.

## 14. Sources and local continuation prompt

These sources informed the design. Read the pinned local source when live documentation disagrees with it.

- RLinf RoboTwin training: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html
- RLinf RoboTwin evaluation: https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/robotwin.html
- RLinf reward workflow: https://rlinf.readthedocs.io/en/latest/rst_source/extending/reward_model.html
- RLinf OpenPI documentation: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_openpi.html
- π0.5 checkpoint: https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle
- Robometer project: https://robometer.github.io/
- Robometer code: https://github.com/robometer/robometer

Suggested instruction to the local GPT:

> Read the current PPO continuation first. SFT and a two-update PPO training/save/reload check passed; the verified checkpoint is outputs/ppo_verified/pi05_ppo/checkpoints/global_step_2. The user will run the full 150-seed evaluation. Use the tested job-scoped NCCL 2.31.2 preload and NCCL_P2P_DISABLE=1. Do not reinstall Torch or redownload existing assets. The 100-iteration training plan has not run. DSRL + Robometer remains the later research goal and needs its config, chunk-reward fix, and reward adapter. Preserve local code/config edits and submodule pins; do not recreate rejected setup helpers.
