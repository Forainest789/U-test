# Benchmark Repositories

SlotMem provides wrapper scripts for evaluating generated videos. The external benchmark and baseline repositories should be cloned here and installed by following their original GitHub instructions.

Quick checkout:

```bash
cd /path/to/SlotMem/bench

git clone https://github.com/Vchitect/VBench.git
git clone --recursive https://github.com/ViStoryBench/vistorybench.git
git clone https://github.com/Eddie0521/IAMFlow.git
```

Each repository can use its own Python environment. Pass the corresponding interpreter paths to `run_slotmem_benchmarks_gpt41_qwen35.sh`:

```bash
VBENCH_PYTHON=/path/to/vbench/python \
NARRASTREAM_API_PYTHON=/path/to/narrastream-api/python \
QWEN35_PYTHON=/path/to/qwen35/python \
bash run_slotmem_benchmarks_gpt41_qwen35.sh /path/to/inference_output
```

ViStoryBench requires manually prepared character reference images. The SlotMem wrappers do not auto-generate or auto-collect reference images.

We thank the VBench, ViStoryBench, and IAMFlow authors for releasing their code and resources to the community.
