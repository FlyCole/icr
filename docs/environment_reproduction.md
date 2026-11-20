# Rebuild the three tested UV environments

Read [the reproduction runbook](REPRODUCE.md) first. These instructions rebuild
the Python and installed-package parts of the November 20, 2026 snapshot. They
must run on the destination computer before experiments start. They are not an
instruction to change the environments of a running source-machine experiment.

| Directory | Python | Torch / CUDA wheel | Transformers at runtime | Purpose |
| --- | --- | --- | --- | --- |
| `rlinf/` | 3.11.15 | 2.7.1+cu128 | OpenPI fork 4.53.2 | Motus, RoboTwin and SAC |
| `robometer/` | 3.10.20 | 2.8.0+cu128 | 4.57.6 | Original Robometer server |
| `.venv-icon/` | 3.10.20 | 2.8.0+cu128 | 4.57.6 | Context-conditioned ICON server |

Keep the environments separate. Both reward repositories install a Python
package named `robometer`, and the policy runtime has a different Torch and
Transformers stack. Do not move an existing venv directory between computers:
recreate it and reinstall editable projects from the restored checkout.

The [inventory](../reproducibility/environments/inventory.json) records all
339/128/128 installed distributions, including immutable cuRobo/PyTorch3D Git
revisions and relative editable paths. The three adjacent requirements files
are full installed-version snapshots. Run every command below from the
repository root after restoring the `vendor_patches/` changes.

## Operating system and native prerequisites

The source machine uses Ubuntu 24.04.4 x86_64, NVIDIA driver 580.178.04, two RTX
PRO 6000 Blackwell Max-Q GPUs with about 96 GB VRAM each, and CUDA 12.8 wheels.
The CUDA 12.8 toolkit is used for native extension builds; a system CUDA 13.2
installation also exists but is not the Torch wheel runtime. System package
versions and native-library hashes are in
[native_runtime.json](../reproducibility/environments/native_runtime.json).

Install the system dependencies required by the pinned RLInf checkout before
building Python packages. Review its
[`requirements/sys_deps.sh`](../third_party/rlinf/requirements/sys_deps.sh) for
Ubuntu package names, including compiler tools, FFmpeg libraries, Vulkan,
EGL/GL, OpenMP and NUMA libraries. This script needs administrator privileges.
Verify that `/usr/share/vulkan/icd.d/nvidia_icd.json` exists and that both GPUs
are visible to `nvidia-smi`. The launchers use the NVIDIA Vulkan ICD explicitly.

The shell `uv` used to create these environments was 0.11.14. Its separate
Python distribution inside the RLInf venv is 0.13.5. The captured package set
is applied with `--no-deps` because it intentionally contains both stock
Transformers metadata and the OpenPI fork, whose tokenizers constraints differ.
An unconstrained `uv sync` or a later dependency resolve changes this stack.

## Create the policy environment

```bash
uv python install 3.11.15 3.10.20
uv venv --python 3.11.15 rlinf

uv pip install --python rlinf/bin/python --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'

uv pip install --python rlinf/bin/python --no-deps \
  'setuptools==75.8.2' 'wheel==0.48.0' 'numpy==1.26.4' \
  'ninja==1.13.2' 'cmake==4.4.4' 'packaging==26.3' 'pybind11==3.1.0'

CUDA_HOME=/usr/local/cuda-12.8 \
uv pip install --python rlinf/bin/python --no-deps --no-build-isolation \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r reproducibility/environments/rlinf-index.txt

CUDA_HOME=/usr/local/cuda-12.8 \
uv pip install --python rlinf/bin/python --no-deps --no-build-isolation \
  -r reproducibility/environments/rlinf-vcs.txt

# Both OpenCV distributions are recorded; the full build owns cv2 at runtime.
uv pip install --python rlinf/bin/python --no-deps --reinstall \
  'opencv-python==4.11.0.86'

# The OpenPI fork must own the shared transformers/ files last.
uv pip install --python rlinf/bin/python --no-deps --reinstall \
  'rlinf-transformer-openpi==4.53.2'
```

The snapshot includes source builds of cuRobo and PyTorch3D at their recorded
Git commits, as well as the installed diagnostics and test dependencies.
These builds need the CUDA development toolkit and build dependencies already
installed above. Current RoboTwin policy rollout uses MPLIB; the cuRobo
planner is not selected by these experiment configs. Flash-attn and Apex were
not installed in the tested environments.

Two OpenCV distributions also share `cv2/`; the active `cv2.abi3.so` matches
`opencv-python==4.11.0.86`, so that distribution is reinstalled explicitly last
for this import name. Its binary hash and both metadata versions are recorded
in the inventory.

The pinned official installer is still useful as a reference:
`third_party/rlinf/requirements/install.sh embodied --model openpi --env robotwin`.
Its unqualified package resolutions are not the captured package snapshot, so
do not run it after the snapshot reconstruction as an environment update.

## Create both reward environments

```bash
uv venv --python 3.10.20 robometer
uv pip install --python robometer/bin/python --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128'
uv pip install --python robometer/bin/python --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r reproducibility/environments/robometer.txt

uv venv --python 3.10.20 .venv-icon
uv pip install --python .venv-icon/bin/python --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128'
uv pip install --python .venv-icon/bin/python --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r reproducibility/environments/icon.txt
```

The `-e` lines in each snapshot install the corresponding local repository.
The reward environments differ in a few transitive package versions; use each
file for its own environment rather than copying the Robometer environment.

## Restore installed source overlays and native libraries

Version pins alone omit modifications made directly in site-packages. The
checked-in [overlay manifest](../reproducibility/environments/installed_overlays.json)
preserves 12 small installed Python files: SAPIEN's OIDN loader and URDF loader,
MPLIB's planner, and nine Augmax files. The original copyright notices and
licenses are retained. Five OpenPI transformer replacements are verified
against the files supplied by `rlinf-openpi==0.1.1`.

After the preceding installs, apply and check them explicitly:

```bash
rlinf/bin/python reproducibility/environments/restore_installed_overlays.py --apply
rlinf/bin/python reproducibility/environments/restore_installed_overlays.py
```

The helper accepts only the recorded distribution versions, known original
file hashes, or already-restored files. Its default mode is read-only. It does
not copy native libraries and it does not change a running training process.

The launchers preload NCCL **2.31.2** from
`data/nccl/2.31.2/nvidia/nccl/lib/libnccl.so.2`. This is a separate target tree;
the RLInf environment's ordinary NCCL distribution stays at **2.26.2**. Restore
the external target tree from the artifact bundle, or reconstruct it from the
exact wheel URL/hash in `native_runtime.json`:

```bash
uv pip install --python rlinf/bin/python --no-deps \
  --target data/nccl/2.31.2 \
  'https://files.pythonhosted.org/packages/0f/36/104de52d6368f5b7f886e8fd252e0a438fe73a215e59b1b47f93a80ae2ea/nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_x86_64.whl#sha256=f9b1dc3c2a7e20176054144ebb3b32fea83b40402ee5d7ac7045cd11ecc956c0'
```

SAPIEN's OIDN loader points to **2.3.3**, replacing the wheel's original 2.0.1.
Restore the matching `sapien/oidn_library/lib/` directory from the external
artifact bundle. On a fresh destination, move the wheel's stock `lib/` directory
aside first, following the commands in the artifact guide: the importer refuses
to overwrite the different 2.0.1 alias files. Never do this to a running source
environment. Alternatively download
[`oidn-2.3.3.x86_64.linux.tar.gz`](https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz),
verify SHA256
`3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f`,
and copy its entire `lib/` directory, preserving symlinks, into
`rlinf/lib/python3.11/site-packages/sapien/oidn_library/lib/`. Preserve all
matching support libraries, not just the CUDA device plugin. The native
manifest records all 14 installed regular library files and the symlink
targets. The current comparisons retain raster rendering and denoiser `none`.

## Validate before launching

1. Check the restored vendor patches, external artifacts, and source overlays.
2. Check the package versions against `inventory.json` and the runtime fork
   version below. Stock Transformers metadata is still 4.57.6 in RLInf, while
   the imported module must be 4.53.2. Both reward servers must import 4.57.6.
3. Run the CPU tests, then the bounded GPU parity and simulator smoke checks
   from the main reproduction runbook before the full experiment queue.

```bash
rlinf/bin/python -c 'import torch,transformers; print(torch.__version__,torch.version.cuda,transformers.__version__,transformers.__file__)'
robometer/bin/python -c 'import torch,transformers; print(torch.__version__,torch.version.cuda,transformers.__version__,transformers.__file__)'
.venv-icon/bin/python -c 'import torch,transformers; print(torch.__version__,torch.version.cuda,transformers.__version__,transformers.__file__)'
```

The snapshots pin versions and source revisions; they are not wheel archives
with a hash for every downloaded package. Archive a wheelhouse or the original
downloaded wheels with the external artifact bundle when offline rebuilding is
required. Environment restoration has been checked against this machine's
installed state; a clean installation on the destination computer still needs
the validation above. Matching seeds, model files and configs supports a
matched rerun, but does not guarantee bit-for-bit training scores across GPU,
driver or native-kernel changes.
