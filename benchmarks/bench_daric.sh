#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

VLLM_PID=""
BENCH_PID=""

VLLM_LOG="$LOG_DIR/vllm_server.log"
BENCH_LOG="$LOG_DIR/benchmark.log"

MODEL="${MODEL:-}"
CONFIG_FILE="${CONFIG_FILE:-configs/daric_pinmem.yaml}"

usage() {
  echo "Usage: $0 --model <model_name_or_path>"
  echo "       $0 --model <model_name_or_path> --config-file <config_path>"
  echo "   or: MODEL=<model_name_or_path> CONFIG_FILE=<config_path> $0"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m|--model)
        if [[ $# -lt 2 || -z "${2:-}" ]]; then
          echo "ERROR: --model requires a value." >&2
          usage
          exit 1
        fi
        MODEL="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      -c|--config-file)
        if [[ $# -lt 2 || -z "${2:-}" ]]; then
          echo "ERROR: --config-file requires a value." >&2
          usage
          exit 1
        fi
        CONFIG_FILE="$2"
        shift 2
        ;;
      *)
        echo "ERROR: Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "$MODEL" ]]; then
    echo "ERROR: MODEL is not set." >&2
    usage
    exit 1
  fi
}

graceful_kill() {
  local pid=$1 name=$2
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Terminating $name (PID: $pid)..."
    kill "$pid" 2>/dev/null
    (wait "$pid" 2>/dev/null) &
    local wait_pid=$!
    sleep 5
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing $name (PID: $pid)..."
      kill -9 "$pid" 2>/dev/null
    fi
    wait "$wait_pid" 2>/dev/null || true
  fi
}

cleanup() {
  echo "Cleaning up processes..."
  graceful_kill "$BENCH_PID" "Benchmark"
  graceful_kill "$VLLM_PID" "vLLM server"

  pkill -f "vllm.serve" 2>/dev/null || true
  pkill -f "benchmark_serving.py" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

launch_prefill() {
  local kv_transfer_config
  kv_transfer_config=$(printf '{"kv_connector":"DARICConnector", "kv_role":"kv_both", "kv_connector_extra_config":{"config_file":"%s"}}' "$CONFIG_FILE")

  echo "Launching vLLM server (log: $VLLM_LOG)..."
  CUDA_VISIBLE_DEVICES=0 \
  VLLM_PLUGINS="daric" \
  vllm serve "$MODEL" \
    --port 55111 \
    --max_model_len=8192 \
    --gpu_memory_utilization=0.9 \
    --kv-transfer-config \
    "$kv_transfer_config" \
    &> "$VLLM_LOG" &
  VLLM_PID=$!

  echo "vLLM PID: $VLLM_PID"

  if ! wait_for_server 55111; then
    exit 1
  fi
}

warmup() {
  echo "Starting warmup..."
  vllm bench serve \
    --port 55111 \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name custom \
    --dataset-path "$SCRIPT_DIR/datas/expanded_wikimqa_warmup.jsonl" \
    --num-prompts 150 \
    --burstiness 100 \
    --custom-output-len 1 \
    --request-rate 1 \
    &> "$BENCH_LOG" &
  BENCH_PID=$!

  if ! wait "$BENCH_PID"; then
    echo "ERROR: Benchmark failed!" >&2
    exit 1
  fi
}

benchmark() {
  echo "Starting benchmark (log: $BENCH_LOG)..."
  vllm bench serve \
    --port 55111 \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name custom \
    --dataset-path "$SCRIPT_DIR/datas/expanded_wikimqa_main.jsonl" \
    --num-prompts 600 \
    --burstiness 100 \
    --request-rate 1 \
    &> "$BENCH_LOG" &
  BENCH_PID=$!

  if ! wait "$BENCH_PID"; then
    echo "ERROR: Benchmark failed!" >&2
    exit 1
  fi
}

main() {
  parse_args "$@"

  launch_prefill
  warmup
  benchmark
  echo "All steps completed successfully."
}

main "$@"