import copy
from functools import wraps
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from ..daric_connector import DARICConnector


def drain_execute_model_state(model_runner):
    execute_model_state = getattr(model_runner, "execute_model_state", None)
    kv_connector_output = getattr(model_runner, "kv_connector_output", None)
    if execute_model_state is None and kv_connector_output is None:
        return None
    use_async_scheduling = getattr(model_runner, "use_async_scheduling", False)
    if use_async_scheduling:
        model_runner.use_async_scheduling = False
    try:
        return model_runner.sample_tokens(None)
    finally:
        if use_async_scheduling:
            model_runner.use_async_scheduling = True

def complete_missing_req_ids(model_runner_output, scheduler_output):
    if isinstance(model_runner_output, ModelRunnerOutput):
        output = copy.deepcopy(model_runner_output) # Do copy to avoid modifying the input_batch in model_runner
        # clear req_ids that not in scheduler_output
        valid_req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(valid_req_ids)}
        sampled_token_ids = []
        for req_id in valid_req_ids:
            if req_id not in output.req_ids:
                sampled_token_ids.append([])
            else:
                sampled_token_ids.append(output.sampled_token_ids[output.req_id_to_index[req_id]])
        output.req_ids = valid_req_ids
        output.req_id_to_index = req_id_to_index
        output.sampled_token_ids = sampled_token_ids
        return output
    elif hasattr(model_runner_output, "get_output") and not isinstance(model_runner_output, ModelRunnerOutput):
        original_get_output = model_runner_output.get_output
        def wrapped_get_output():
            base_output = original_get_output()
            updated_output = complete_missing_req_ids(base_output, scheduler_output)
            return updated_output
        model_runner_output.get_output = wrapped_get_output
        return model_runner_output
    return model_runner_output

def wrap_execute(original_execute):
    from vllm.v1.worker.gpu_worker import Worker
    @wraps(original_execute)
    def _wrapper(self, *args, **kwargs):
        if not has_kv_transfer_group():
            return original_execute(self, *args, **kwargs)
        kv_connector = get_kv_transfer_group()
        daric_connector = kv_connector if isinstance(kv_connector, DARICConnector) else None
        if daric_connector is None:
            return original_execute(self, *args, **kwargs)
        
        # Lazy initialization of runner_tracker
        if not daric_connector.finish_runner_register():
            daric_connector.register_model_runner(self.model_runner)
        
        scheduler_output = args[0] if args else kwargs.get("scheduler_output")
        daric_connector.bind_scheduler_output(scheduler_output)
        if scheduler_output.total_num_scheduled_tokens == 0:
            return original_execute(self, *args, **kwargs)
        
        pre_scheduler_output, post_scheduler_outputs = (
            daric_connector.rebuild_scheduler_output(scheduler_output)
        )
        if not daric_connector.pic_available():
            # Prefix generation in this step
            return original_execute(self, *args, **kwargs)

        output_pre = original_execute(self, pre_scheduler_output)
        # vLLM V1 requires execute_model() to be paired with sample_tokens().
        # Drain intermediate stages instead of clearing execute_model_state.
        if hasattr(Worker, "sample_tokens") and output_pre is None and len(post_scheduler_outputs) > 0:
            drain_execute_model_state(self.model_runner)

        output_post = None
        for i, post_scheduler_output in enumerate(post_scheduler_outputs):
            output_post = original_execute(self, post_scheduler_output)
            is_last_post = i == len(post_scheduler_outputs) - 1
            if hasattr(Worker, "sample_tokens") and not is_last_post and output_post is None:
                drain_execute_model_state(self.model_runner)

        final_output = output_post if len(post_scheduler_outputs) != 0 else output_pre
        
        if final_output is not None:
            if get_pp_group().is_last_rank:
                final_output = complete_missing_req_ids(final_output, scheduler_output)
        return final_output
    return _wrapper


def wrap_sample(original_sample):
    @wraps(original_sample)
    def _wrapper(self, *args, **kwargs):
        output = original_sample(self, *args, **kwargs)
        if not has_kv_transfer_group():
            return output
        kv_connector = get_kv_transfer_group()
        daric_connector = kv_connector if isinstance(kv_connector, DARICConnector) else None
        if not daric_connector:
            return output
        if not daric_connector.pic_available():
            return output
        if output is None:
            output = EMPTY_MODEL_RUNNER_OUTPUT
        scheduler_output_step = getattr(
                daric_connector.daric_engine,
                "scheduler_output_step",
                None,
            )
        if get_pp_group().is_last_rank:
            output = complete_missing_req_ids(output, scheduler_output_step)
        return output
    return _wrapper


def wrap_worker(cls):
    cls.execute_model = wrap_execute(cls.execute_model)
    if hasattr(cls, "sample_tokens"):
        cls.sample_tokens = wrap_sample(cls.sample_tokens)
    return cls