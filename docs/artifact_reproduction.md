# Reproduction inputs and artifact transfer

The code commit is only one part of a reproduction. Model weights, simulator
assets, the tokenizer, fixed ICON references, and the inference configuration
must also match. Their revisions and hashes are recorded in
[`artifacts_manifest.json`](../reproducibility/artifacts_manifest.json).

The supported first-reproduction checkout is **`/home/yifei/icr`**. Public YAMLs
and some historical launchers contain that path. `scripts/repro_artifacts.py`
can relocate copied metadata to a different root, but it does not rewrite all
public YAMLs or claim a fully relocated checkout is ready to launch. See the
top-level reproduction guide before choosing a different location.

## Inputs stored in Git

The six successful reference demonstrations are preserved as five unique,
deterministic gzip archives totaling **15,082,280 bytes (14.38 MiB)**. Restoring
them reconstructs the original `.npy` bytes and verifies their SHA-256 hashes;
it does not recollect new demonstrations. The fixed Motus bottle reference and
object-bank base13 share the same frame archive. The references cover:

| Reference | Scene seed | Real actions | Raw frames |
| --- | ---: | ---: | ---: |
| Original task-specific SFT bottle | 1 | 127 | 128 |
| Motus fixed bottle | 1 | 119 | 120 |
| Motus fixed move-can-pot | 100000 | 118 | 119 |
| Motus fixed click-bell | 100001 | 120 | 121 |
| Motus object-bank bottle base13 | 1 | 119 | 120 |
| Motus object-bank bottle base16 | 7 | 137 | 138 |

Small historical cache-parity reports, their approval summary, the historical
Robometer converter receipt, Motus preparation receipts, object-bank manifest,
and the two actual inference `config.yaml` files are also versioned under
`reproducibility/bootstrap_inputs/`. These files used to exist only under ignored
`outputs/`, `data/` and `checkpoints/` paths; a clean clone otherwise fails
launcher preflight before training begins.

The parity reports remain **historical numerical evidence from the source
machine**. Restoring them does not claim that a new GPU has been benchmarked.
The current managed queue checks the relevant cache-core source hashes and
performs a new SAC smoke. Run the opt-in GPU parity tests on the new machine as
described in the reproduction guide.

## Restore the versioned small inputs

Use a fresh clone with vendor patches restored. The helper is standard-library
Python; it does not import CUDA or install packages. Inspect the plan first:

```bash
cd /home/yifei/icr
python3 scripts/repro_artifacts.py restore-small
python3 scripts/repro_artifacts.py restore-small --apply
```

The helper refuses to overwrite different existing files, validates all planned
inputs before writing any of them, and refuses mutation while a queue appears
active. It restores reference pixels exactly. Copied metadata's original NVMe
checkpoint path is rebased to the local Motus directory; copied object-bank
metadata hashes are updated to describe that path-only change. The versioned
original evidence is left intact.

Do **not** run this restore against the current training machine's live files.
The current live queue reads and pins those files. On a fresh machine, run it
before downloading reward weights so that downloads can preserve the recorded
inference overrides.

## Obtain the large public inputs

| Input | Repository | Exact revision |
| --- | --- | --- |
| Motus | `motus-robotics/pi0.5_robotwin2` | `effc6e1a99bd4d0edf9447e8ba313def67649cfe` |
| ICON | `sniper-qian/ICON-4B` | `e75b5fca997ab18f620ecb1931e5d9acba7c1f9b` |
| Robometer | `robometer/Robometer-4B` | `beef63bc914c5c189329d49c6d712d96d632aa34` |
| Qwen processor/base model | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Optional original bottle SFT | `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` | `fa8df6ed103db0f5549c122f3a17c00ba6426c98` |
| RoboTwin asset dataset | `TianxingChen/RoboTwin2.0` | `3dc3b798668feb99ac61cc9086d84cbcc3d79186` |

After the environments exist, download inference inputs from the manifest.
The inference directories can contain the reward weights directly; the old
source machine's symlink layout is not required. This command preserves the
versioned inference `config.yaml`, including `use_unsloth: false` and the local
Qwen path:

```bash
rlinf/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

manifest = json.loads(Path('reproducibility/artifacts_manifest.json').read_text())
for model in manifest['models']:
    if model.get('optional'):
        continue
    snapshot_download(
        repo_id=model['repo_id'], revision=model['revision'],
        local_dir=model['local_path'],
        allow_patterns=[item['path'] for item in model['files']],
        ignore_patterns=['config.yaml'] if model.get('inference_config_snapshot') else None,
    )
PY
```

The existing `scripts/download_motus_checkpoint.py` is another pinned,
parallel-download option for Motus alone. Use a Hugging Face login if access to
the selected model requires it; credentials never belong in this repository or
the transfer bundle.

Download and verify the three simulator archives before extraction:

```bash
rlinf/bin/python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

m = json.loads(Path('reproducibility/artifacts_manifest.json').read_text())['simulator_assets']
dest = Path(m['extract_to'])
dest.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=m['repo_id'], repo_type='dataset', revision=m['revision'],
                  local_dir=str(dest), allow_patterns=[x['filename'] for x in m['archives']])
for item in m['archives']:
    path = dest / item['filename']
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    assert h.hexdigest() == item['sha256'], path
    subprocess.run(['unzip', '-q', str(path), '-d', str(dest)], check=True)
PY
(cd third_party/robotwin && ../../rlinf/bin/python script/update_embodiment_config_path.py)
```

The old ZIP files were deleted after extraction. Their hashes in the committed
manifest come from the completed Hugging Face download metadata. A new download
is checked against those hashes rather than against the dataset's moving main
branch. For transfer instead of download, use a verified bundle below, which
records every extracted asset file.

The native launcher also preloads the exact NCCL library at
`data/nccl/2.31.2/nvidia/nccl/lib/libnccl.so.2`. Transfer that file or follow the
environment guide to install the pinned NCCL wheel into that separate path;
check its manifest SHA-256. This is separate from the NCCL dependency installed
with PyTorch.

SAPIEN also has a local native-library repair: its installed loader uses
**OIDN 2.3.3** under
`rlinf/lib/python3.11/site-packages/sapien/oidn_library/lib/`. The runtime group
includes all 2.3.3 libraries and their alias filenames, plus the original
installation receipt. Bundle copies dereference aliases into regular files;
the filenames and ELF contents needed by the loader remain the same. Create
the environments before importing this runtime group. The pinned OIDN release
download is an alternative described in the environment guide.

OpenPI additionally needs its 4,264,023-byte PaliGemma tokenizer, originally
cached outside this checkout. Download it from its source and verify it:

```bash
python3 - <<'PY'
import hashlib, json, urllib.request
from pathlib import Path

t = json.loads(Path('reproducibility/artifacts_manifest.json').read_text())['tokenizer']
p = Path(t['bundle_destination'])
p.parent.mkdir(parents=True, exist_ok=True)
if not p.exists():
    urllib.request.urlretrieve(t['source_url'], p)
assert hashlib.sha256(p.read_bytes()).hexdigest() == t['sha256']
PY
export OPENPI_DATA_HOME="$PWD/data/runtime/openpi"
```

The launcher preserves `OPENPI_DATA_HOME` from the shell. Set it in each launch
terminal, including detached launches, when using this repository-local cache.

Check the restored files before running preflight:

```bash
python3 scripts/repro_artifacts.py check
python3 scripts/repro_artifacts.py check --full-hashes
```

The first check hashes references, the tokenizer, the NCCL file and small model
files, while checking large weight sizes. `--full-hashes` reads the complete
model weights. Extracted simulator directories receive an existence check;
their full integrity comes from the checked source ZIPs or the bundle manifest.
Add `--legacy` when the optional original bottle-SFT checkpoint is installed.

## Transfer an exact local bundle instead of downloading

The bundle helper copies only allowlisted model, simulator and runtime inputs.
It dereferences source symlinks into ordinary files and computes a SHA-256 for
every copied file. It excludes package environments, caches, credentials, logs,
Ray sessions, and training outputs. It creates a new directory and never
overwrites an existing bundle. Export is a read-only operation on source inputs
and can require tens of gigabytes of destination space.

On the original machine, inspect size and paths, then export explicitly:

```bash
python3 scripts/repro_artifacts.py export-bundle --bundle /path/to/new/icr-inputs
python3 scripts/repro_artifacts.py export-bundle --bundle /path/to/new/icr-inputs --apply
```

The default groups are `models simulator runtime`. Add `--groups models
simulator runtime legacy` to include the original task-specific bottle-SFT
checkpoint. Fixed references and small receipts are already in Git and are not
duplicated into the external bundle. Transfer the complete bundle directory to
the other machine, create the environments, then inspect and import it
**before** `restore-small`:

```bash
# On a fresh destination venv, first move the stock OIDN directory aside.
# Skip if lib/ is absent. The backup destination must not already exist.
mv rlinf/lib/python3.11/site-packages/sapien/oidn_library/lib \
   rlinf/lib/python3.11/site-packages/sapien/oidn_library/lib.wheel-2.0.1-backup
python3 scripts/repro_artifacts.py import-bundle --bundle /path/to/icr-inputs
python3 scripts/repro_artifacts.py import-bundle --bundle /path/to/icr-inputs --apply
python3 scripts/repro_artifacts.py restore-small --apply
export OPENPI_DATA_HOME="$PWD/data/runtime/openpi"
python3 scripts/repro_artifacts.py check --full-hashes
```

Imports verify every source file before making any destination changes and
refuse different existing files. Import into fresh ordinary directories; the
helper rejects destination symlinks that escape the selected checkout root.
The stock SAPIEN OIDN 2.0.1 aliases conflict with the recorded 2.3.3 libraries,
which is why the destination-only move above is necessary. Do not move native
libraries in an active environment or repeat that step on an already-restored
2.3.3 environment. Keep the original directory as a backup.
The bundle hash protects transfer integrity; the committed model hashes protect
the selected model revisions. Keep both verifications.

## Re-evaluation versus retraining

Pretrained weights and fixed contexts are sufficient to retrain the recorded
recipes. Evaluating the **same learned policy** requires its retained trained
checkpoint as well; it is not included in the input bundle or Git. Consult
[`experiment_results.csv`](experiment_results.csv) for each retained checkpoint
and evaluation config. Transfer the final inference weights with their original
relative layout, and hash them before and after transfer. For exact training
resume, retain the complete checkpoint, optimizer state, FP32 EMA, RNG and
replay buffer; several historical replay buffers were intentionally removed.

Historical results marked as missing checkpoints cannot be reevaluated from a
deleted policy. Historical converter-bug or unfiltered-replay runs cannot be
recreated simply by running the latest repaired source. Match their archived
source/config provenance where available, and keep those limits explicit when
comparing results.

Even with the same recipe and inputs, a different GPU, driver, rendering stack
or nondeterministic training operation can change trajectories and final scores.
The committed setup preserves the actual recipe and inputs; it does not promise
bitwise-identical multi-hour training across machines.
