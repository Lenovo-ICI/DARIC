# DARIC

This repository contains the offitial implementation of the SIGMOD'27 paper "[DARIC: A Position-Independent KV Cache Reuse System with Distribution-Aware Recomputation](https://arxiv.org/abs/2611.09109)".

## Requirements

- Python `3.12`
- CUDA-enabled environment (for extension build)
- vLLM >=`0.11.0`

## Installation

1. Create and activate a virtual environment.

```bash
uv venv --python 3.12
source .venv/bin/activate
```

2. Install vLLM and DARIC.


```bash
uv pip install vllm==0.30.0
uv pip install --no-build-isolation -e .
```

## Quick Start

### 1) Start KV Cache Service

Use one backend mode:

```bash
daric-store --mem-size $MEM_SIZE --disk-size $DISK_SIZE --path $DISK_PATH
```

### 2) Configure DARIC in vLLM

Launch vLLM with the DARIC plugin:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS=daric \
vllm serve /path/to/model \
  --port 55111 \
  --max_model_len 8192 \
  --gpu_memory_utilization 0.9 \
  --kv-transfer-config \
  '{"kv_connector":"DARICConnector","kv_role":"kv_both","kv_connector_extra_config":{"config_file":"examples/daric_config.yaml"}}'
```

## Examples

Batch inference:

```bash
python examples/batch_inference.py --model $MODEL --config-file $CONFIG_FILE [--tp] [--pp]
```

Async inference:

```bash
python examples/async_inference.py --model $MODEL --config-file $CONFIG_FILE [--tp] [--pp]
```

## Benchmark

Benchmark scripts and configuration files are located in `benchmarks/`.

- Main script: `benchmarks/main.py`
- DARIC Throughput script: `benchmarks/bench_daric.sh`
- vLLM Throughput script: `benchmarks/bench_vllm.sh`

## Datasets

Download the processed datasets from the [CacheBlend repository](https://github.com/YaoJiayi/CacheBlend) and place them in `benchmarks/datas/`.

The expanded WikiMQA dataset is already provided at `benchmarks/datas/expanded_wikimqa_warmup.jsonl` and `benchmarks/datas/expanded_wikimqa_main.jsonl`.
