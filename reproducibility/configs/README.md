# Recorded experiment configurations

These 54 YAML files preserve the saved, resolved configurations from 31 formal
150-episode evaluations and 23 associated training runs. They retain old settings
even when the public YAML defaults change. The index and result evidence are in
[`../experiments.json`](../experiments.json); the compact result table is
[`../../docs/experiment_results.csv`](../../docs/experiment_results.csv).

Filenames are `<experiment_id>__eval.yaml` and `<experiment_id>__train.yaml`.
Missing training snapshots are recorded explicitly in the index. Baseline
policies have no associated RL training snapshot.

Only machine/output/checkpoint paths were substituted:

| Variable | Meaning |
| --- | --- |
| `ICR_REPO_ROOT` | Absolute path to the restored checkout |
| `ICR_RUN_ROOT` | A fresh destination for this one training or evaluation invocation |
| `ICR_EVAL_CHECKPOINT` | Trained checkpoint to evaluate; required only by trained-policy evaluation snapshots |

The substitutions use Hydra's `${oc.env:NAME}` syntax. All other saved values,
including learning rates, seeds, model architecture, reward settings, renderer,
action limits, batch sizes and update schedules, are unchanged. Model weights,
simulator assets, fixed context demonstrations and environments must be restored
first as described in the main README. An ICON or Robometer training snapshot
expects the matching reward service at its recorded endpoint; it does not start
that service on its own.

The index records the source configuration SHA256, exported configuration
SHA256, native success count, original metrics SHA256, and available launch-time
source/model/context hashes. The results were checked against the exact
TensorBoard `eval/success_once` and `eval/num_trajectories` series. Each contains
150 held-out outcomes. Smoke checks, training success rates and incomplete
experiments are excluded from the completed-result count.

Historical implementation differences still matter. In particular, the old
reward-converter bug is documented rather than reintroduced, one original
simulator training configuration was removed before this backup, and several
historical checkpoints were deleted during previous authorized cleanups. A
recorded configuration preserves the recipe parameters but cannot recreate
removed historical source or checkpoint bytes. Use the current Motus recipe
for a fresh matched comparison, and compare the launch hashes before claiming
an exact recreation of an older implementation.

The current move-can-pot ICON and Robometer runs had not completed formal
evaluation at capture on November 20, 2026. Their pending status appears under
`unfinished_at_capture`; no success rate is assigned to either.
