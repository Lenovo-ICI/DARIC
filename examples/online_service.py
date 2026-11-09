"""
run service with:

CUDA_VISIBLE_DEVICES=0 \
VLLM_PLUGINS="daric" \
vllm serve /home/data/models/Qwen3-32B \
    --port 55111 \
    --max_model_len=8192 \
    --gpu_memory_utilization=0.9 \
    --kv-transfer-config \
    '{"kv_connector":"DARICConnector","kv_role":"kv_both", "kv_connector_extra_config":{"config_file": "examples/daric_config.yaml"}}'
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import error, request

from prompt_example import prompt


def post_completion(
    api_url: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    except error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e

    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError(f"Invalid response: {result}")
    return choices[0].get("text", "")


def main(args):
    requests = [prompt for _ in range(args.batch_size)]
    api_url = f"http://{args.host}:{args.port}/v1/completions"

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                post_completion,
                api_url,
                args.model,
                req,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.timeout,
            ): idx
            for idx, req in enumerate(requests)
        }

        results = [None] * len(requests)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:  # Keep the batch running on partial failure.
                results[idx] = f"[Request {idx}] ERROR: {e}"

    for i, text in enumerate(results):
        print(f"[Request {i}] {text}")
        print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARIC Integration Test")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=55111,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model name exposed by vLLM service.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
    )
    args = parser.parse_args()
    main(args)