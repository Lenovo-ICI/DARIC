import os
import argparse
import asyncio
import multiprocessing as mp
from prompt_example import prompt


async def async_llm_process(input_queue: mp.Queue, output_queue: mp.Queue, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["DARIC_LOG_LEVEL"] = "debug"
    os.environ["VLLM_PLUGINS"] = "daric"
    
    from vllm import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TextPrompt
    from vllm.config import KVTransferConfig
    import uuid

    loop = asyncio.get_running_loop()
    llm = None
    while True:
        data, sign = await loop.run_in_executor(None, input_queue.get)
        if sign == "init":
            print("Initializing async LLM process.")
            engine_args = AsyncEngineArgs(
                model=args.model,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                max_model_len=8192,
                kv_transfer_config=KVTransferConfig(
                    kv_connector="DARICConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={
                        "config_file": args.config_file,
                    },
                ),
                enable_prefix_caching=False,
                tensor_parallel_size=2 if args.tp else 1,
                pipeline_parallel_size=2 if args.pp else 1
            )
            llm = AsyncLLM.from_engine_args(engine_args)
            output_queue.put("FINISH")
        elif sign == "generate":
            request = data
            prompt_text = request["prompt_text"]
            prompt_text = llm.get_tokenizer().apply_chat_template(
                [
                    {"role": "user", "content": prompt_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if os.getenv("USE_DARIC_SEP", "1") == "0":
                special_str = llm.get_tokenizer().all_special_tokens[0]
                prompt_text = prompt_text.replace("<DARIC_SEP>", f"{special_str}{special_str}{special_str}")

            async def generate_one_task(request=request):
                try:
                    sp = SamplingParams(
                        max_tokens=args.max_tokens, temperature=0.4, top_p=0.5, top_k=5
                    )
                    request_id = str(uuid.uuid4())
                    final_output = None
                    prompt = TextPrompt(prompt=prompt_text)
                    async for output in llm.generate(prompt=prompt, sampling_params=sp, request_id=request_id):
                        final_output = output
                    output_queue.put(
                        {
                            "request_index": request["request_index"],
                            "output": final_output,
                        }
                    )
                except Exception as e:
                    output_queue.put(("ERROR", repr(e)))

            loop.create_task(generate_one_task())
        elif sign == "exit":
            output_queue.put("FINISH")
            print("Exiting kv both process.")
            break


def async_llm_process_wrapper(input_queue: mp.Queue, output_queue: mp.Queue, args):
    asyncio.run(async_llm_process(input_queue, output_queue, args))


def main(args):
    input_queue, output_queue = mp.Queue(), mp.Queue()
    kv_both_process = mp.Process(
        target=async_llm_process_wrapper,
        args=(input_queue, output_queue, args),
    )
    
    kv_both_process.start()
    input_queue.put((None, "init"))
    init_response = output_queue.get()
    if init_response == "FINISH":
        print("Engine process is ready.")
    requests = [prompt for _ in range(args.batch_size)]
    import queue
    collected_outputs = {}

    async def run_async_test():
        async def submit_loop():
            for request_index, req in enumerate(requests):
                input_queue.put(
                    (
                        {
                            "prompt_text": req,
                            "request_index": request_index,
                        },
                        "generate",
                    )
                )
                await asyncio.sleep(0.1)

        async def collect_loop():
            remaining = len(requests)
            while remaining > 0:
                try:
                    res = output_queue.get_nowait()
                    if isinstance(res, tuple) and len(res) == 2 and res[0] == "ERROR":
                        raise RuntimeError(f"Worker error: {res[1]}")
                    if isinstance(res, dict):
                        collected_outputs[res["request_index"]] = res["output"].outputs[0].text
                        remaining -= 1
                except queue.Empty:
                    pass
                await asyncio.sleep(0.01)
        await asyncio.gather(submit_loop(), collect_loop())

    asyncio.run(run_async_test())

    input_queue.put((None, "exit"))
    finish_response = output_queue.get()
    if finish_response == "FINISH":
        print("KV both process has exited.")
    kv_both_process.join()

    for request_index in range(len(requests)):
        print(collected_outputs[request_index])
        print("========================================")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="DARIC Integration Test")
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="0,1",
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
