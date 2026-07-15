# Benchmark Environments

SlotMem provides wrapper scripts for evaluating generated videos, but the benchmark suites themselves should be installed by following their original GitHub repositories.

Expected local checkout layout:

```text
bench/
  VBench/
  NarraStream-Bench/
  vistorybench/
```

Each benchmark can use its own Python environment. Pass the corresponding interpreter paths to `run_slotmem_benchmarks_gpt41_qwen35.sh`:

```bash
VBENCH_PYTHON=/path/to/vbench/python \
NARRASTREAM_API_PYTHON=/path/to/narrastream-api/python \
QWEN35_PYTHON=/path/to/qwen35/python \
bash run_slotmem_benchmarks_gpt41_qwen35.sh /path/to/inference_output
```

ViStoryBench requires manually prepared character reference images. The SlotMem wrappers do not auto-generate or auto-collect reference images.
