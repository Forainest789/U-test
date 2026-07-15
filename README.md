# SlotMem for Wan2.2 I2V

This repository contains the training and inference code for SlotMem, a long-video generation pipeline built on top of Wan2.2 I2V.

Main entry points:

- `train_slotmem_stage1.sh`: stage-1 training launcher
- `train_slotmem_stage2.sh`: stage-2 training launcher
- `test_slotmem_stage1.sh`: stage-1 inference launcher
- `test_slotmem_stage2.sh`: stage-2 inference launcher

## Environment

The base package dependencies are defined in `pyproject.toml`, and the extra runtime packages used by the SlotMem training and inference launchers are listed in `requirements_slotmem.txt`. Optional packages for data curation, Streamlit interfaces, and local LLM benchmark helpers are listed in `requirements_optional.txt`.

Example setup:

```bash
conda create -n slotmem python=3.10 -y
conda activate slotmem

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -r requirements_slotmem.txt
# Optional:
pip install -r requirements_optional.txt
```

The current setup was checked against:

- `python3 train_mem_Encoder.py --help`
- `python3 test_mem_Encoder.py --help`

Prepare the Wan2.2 I2V base checkpoint directory, then set `CKPT_DIR` to that path. The launchers default to `/models/Wan2.2-I2V-A14B` as a public convention for local model storage; override `CKPT_DIR` when your model lives elsewhere.
SlotMem checkpoints can be downloaded from Hugging Face: https://huggingface.co/YilaiLiu-HKU/SlotMem/tree/main/ckpt
The shell launchers use repo-local `ckpt/stage1/*.pt` and `ckpt/stage2/*.pt` paths as defaults, but checkpoint files are not committed to Git.

The repository includes a vendored DiffSynth/Wan2.2 runtime under `diffsynth/`.
The installable package name is `slotmem`, while the runtime import namespace
remains `diffsynth` for compatibility with the upstream codebase.
The release is distributed under Apache-2.0; see `LICENSE` and `NOTICE`.

## Data

Minimal examples are included under `sample/`:

- `sample/train/`: one training group with
  - `candidate_groups.csv`
  - `stage2_candidate_groups.csv`
  - `character_lists/Top001.json`
  - `video/Top001/group_16/*.mp4`
- `sample/test/3_271/`: one inference example with
  - `frame.jpg`
  - `rewrite_caption.json`

These files are only for showing the expected input format. They are enough to inspect the data layout and to wire paths correctly, but not intended as a meaningful benchmark.
The `sample/train/video/` MP4 files are synthetic placeholders generated for format validation only.

## Training

Standard launcher:

```bash
cd /path/to/SlotMem
DATA_ROOT=/path/to/train_data CKPT_DIR=/path/to/Wan2.2-I2V-A14B bash train_slotmem_stage1.sh
DATA_ROOT=/path/to/train_data CKPT_DIR=/path/to/Wan2.2-I2V-A14B bash train_slotmem_stage2.sh
```

Important variables in `train_slotmem_stage1.sh` and `train_slotmem_stage2.sh`:

- `CKPT_DIR`: Wan2.2 I2V base model directory
- `DATA_ROOT`: dataset root
- `CANDIDATE_CSV`: sample-group metadata CSV
- `CHARACTER_LISTS_DIR`: character list directory
- `VIDEO_ROOT`: video directory
- `OUTPUT_ROOT`: experiment output directory
- `CUDA_VISIBLE_DEVICES`: training GPUs
- `HIGH_EXPERT_CKPT_PATH` / `LOW_EXPERT_CKPT_PATH`: stage-1 checkpoints used by stage 2

Example with the sample layout:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
CKPT_DIR=/path/to/Wan2.2-I2V-A14B \
DATA_ROOT=./sample/train \
CANDIDATE_CSV=./sample/train/candidate_groups.csv \
CHARACTER_LISTS_DIR=./sample/train/character_lists \
VIDEO_ROOT=./sample/train/video \
OUTPUT_ROOT=./experiments/wan22_slotmem_i2v \
bash train_slotmem_stage1.sh
```

For stage 2 with the sample layout, use `CANDIDATE_CSV=./sample/train/stage2_candidate_groups.csv` and set `HIGH_EXPERT_CKPT_PATH` / `LOW_EXPERT_CKPT_PATH` to the stage-1 checkpoints.

The launchers run `high_noise` / `low_noise` phases and save trainable weights under `OUTPUT_ROOT`.

## Inference

Standard launcher:

```bash
cd /path/to/SlotMem
JSON_PATH=/path/to/rewrite_caption.json REF_IMAGE_PATH=/path/to/frame.jpg bash test_slotmem_stage1.sh
JSON_PATH=/path/to/rewrite_caption.json REF_IMAGE_PATH=/path/to/frame.jpg bash test_slotmem_stage2.sh
```

Important variables in `test_slotmem_stage1.sh` and `test_slotmem_stage2.sh`:

- `CKPT_DIR`: Wan2.2 I2V base model directory
- `JSON_PATH`: story json
- `REF_IMAGE_PATH`: reference image
- `OUTPUT_ROOT`: output directory
- `HIGH_EXPERT_CKPT_PATH`: high-noise expert checkpoint
- `LOW_EXPERT_CKPT_PATH`: low-noise expert checkpoint
- `CUDA_VISIBLE_DEVICES`: inference GPU

Example:

```bash
CUDA_VISIBLE_DEVICES=0 \
CKPT_DIR=/path/to/Wan2.2-I2V-A14B \
JSON_PATH=./sample/test/3_271/rewrite_caption.json \
REF_IMAGE_PATH=./sample/test/3_271/frame.jpg \
HIGH_EXPERT_CKPT_PATH=/path/to/high_noise.pt \
LOW_EXPERT_CKPT_PATH=/path/to/low_noise.pt \
OUTPUT_ROOT=./inference_outputs/slotmem_i2v \
bash test_slotmem_stage1.sh
```

By default, the script keeps `chunk 0` as base Wan inference and loads the expert checkpoints for later chunks.

## Benchmarks

Benchmark helpers are provided for already generated SlotMem inference output
folders:

```bash
OPENAI_API_KEY=... \
NARRASTREAM_API_BASE_URL=https://api.openai.com/v1 \
bash run_slotmem_benchmarks_api.sh /path/to/inference_output
```

The full helper `run_slotmem_benchmarks_gpt41_qwen35.sh` also expects external
VBench, NarraStream-Bench, and local Qwen3.5 environments via environment
variables. ViStoryBench reference images are not auto-collected; they should be
prepared manually before enabling that benchmark.

Expected benchmark checkout layout:

```bash
mkdir -p bench
git clone https://github.com/Vchitect/VBench.git bench/VBench
git clone <NarraStream-Bench repository URL> bench/NarraStream-Bench
```

`bench/` is intentionally ignored by Git. Install each benchmark in its own environment and pass the interpreter paths to the helper.

For the full helper, provide the benchmark environments explicitly:

```bash
NARRASTREAM_REPO=./bench/NarraStream-Bench \
VBENCH_PYTHON=/path/to/vbench/python \
NARRASTREAM_API_PYTHON=/path/to/narrastream-api/python \
QWEN35_PYTHON=/path/to/qwen35/python \
OPENAI_COMPAT_API_KEY=... \
OPENAI_COMPAT_BASE_URL=https://api.openai.com/v1 \
bash run_slotmem_benchmarks_gpt41_qwen35.sh /path/to/inference_output
```

Data curation uses TransNetV2 for shot detection. Follow `data_curation/README.md` for the required TransNetV2 checkout and weights layout.
