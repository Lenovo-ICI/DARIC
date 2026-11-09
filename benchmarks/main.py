import argparse
import json
import os
import time
from pathlib import Path

from dataset import DataLoader
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TextPrompt


def run(datas: DataLoader, llm):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=1)
    ttfts, scores = [], []
    for data in datas:
        docs, question, answers = data
        all_chunks = [datas.system_prompt_template] + docs + [question]
        prompt_str = tokenizer.apply_chat_template(
            conversation=[
                {"role": "user", "content": "<DARIC_SEP>".join(all_chunks)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        # warmup
        llm.generate(prompts=[TextPrompt(prompt=prompt_str)], sampling_params=sampling_params)
        # ttft
        time0 = time.time()
        llm.generate(prompts=[TextPrompt(prompt=prompt_str)], sampling_params=sampling_params)
        time1 = time.time()
        ttft = (time1 - time0) * 1000
        print(f"TTFT for current sample: {ttft} ms")

        generation_sampling_params = SamplingParams(max_tokens=datas.max_tokens, temperature=0.0)

        outputs = llm.generate(prompts=[TextPrompt(prompt=prompt_str)], sampling_params=generation_sampling_params)
        pred = outputs[0].outputs[0].text
        try:
            score = max(
                [
                    datas.metric(pred, answer, tokenizer)
                    for answer in answers
                ]
            )
        except Exception:
            score = 0.0
        ttfts.append(ttft)
        scores.append(score)
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_ttft, avg_score


def evaluate(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_USE_V1"] = "1"
    # os.environ["DARIC_LOG_LEVEL"] = "debug"
    os.environ["DARIC_CLEAN_CACHE_ON_START"] = "1"
    os.environ["VLLM_PLUGINS"] = "daric"
    os.environ["DARIC_MAX_GROUP"] = "16"
    os.environ["USE_DARIC_SEP"] = "1"
    os.environ["USE_DARIC_PARSE"] = "1"
    os.environ["DARIC_SEP_TOKEN"] = "<DARIC_SEP>"

    benchmark_dir = Path(__file__).resolve().parent
    with open(benchmark_dir / "configs" / "dataset_config.json", "r") as f:
        config = json.load(f)

    if args.dataset not in config:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available datasets: {list(config.keys())}")

    llm = LLM(
        model=args.model,
        kv_transfer_config=KVTransferConfig(
            kv_connector="DARICConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "config_file": args.config_file,
            },
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False
    )

    datas = DataLoader(args.dataset, config[args.dataset])
    avg_ttft, avg_score = run(datas, llm)
    
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "num_samples": len(datas),
        "avg_ttft_ms": avg_ttft,
        "avg_score": round(avg_score, 6),
    }
    print(f"Results by dataset: {{{args.dataset!r}: {result['avg_score']}}}")
    print("Benchmark summary:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved benchmark result to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DARIC overall benchmark (TTFT + task score).")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF model name.")
    parser.add_argument("--dataset", type=str, default="wikimqa", help="Dataset key in dataset_config.json.")
    parser.add_argument("--config-file", type=str, required=True, help="Path to DARIC yaml config.")
    parser.add_argument("--cuda-visible-devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM gpu_memory_utilization.")
    parser.add_argument("--max-model-len", type=int, default=20480, help="vLLM max_model_len.")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save summary JSON.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)
