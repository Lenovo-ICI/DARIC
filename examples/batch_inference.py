import os
import argparse
import multiprocessing as mp
from prompt_example import prompt

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["DARIC_LOG_LEVEL"] = "debug"
    os.environ["VLLM_PLUGINS"] = "daric"
    from vllm import SamplingParams, LLM
    from vllm.inputs import TextPrompt
    from vllm.config import KVTransferConfig

    llm = LLM(
        model=args.model,
        kv_transfer_config=KVTransferConfig(
            kv_connector="DARICConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "config_file": args.config_file,
            },
        ),
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enable_prefix_caching=False,
        tensor_parallel_size=2 if args.tp else 1,
        pipeline_parallel_size=2 if args.pp else 1,
    )
    requests = [TextPrompt(prompt=prompt) for _ in range(args.batch_size)]
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, temperature=0
    )

    outputs = llm.generate(requests, sampling_params=sampling_params)
    for output in outputs:
        print(output.outputs[0].text)
        print("========================================")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="DARIC Integration Test")
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="6,7",
    )
    parser.add_argument(
        "--tp",
        action="store_true",
    )
    parser.add_argument(
        "--pp",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True
    )
    args = parser.parse_args()
    main(args)
