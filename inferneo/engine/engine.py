"""Engine core: wires tokenizer, scheduler, KV manager, and model runner.

This module is the composition point, so it may import the tensor plane —
but it does so lazily, keeping ``inferneo.engine.scheduler`` and
``inferneo.kv`` importable without torch.
"""

from __future__ import annotations

import uuid

from inferneo.config import EngineConfig, ModelConfig
from inferneo.engine.request import EngineRequest
from inferneo.engine.scheduler import Scheduler
from inferneo.kv.block_manager import KVCacheManager
from inferneo.outputs import (
    CompletionOutput,
    RequestMetrics,
    RequestOutput,
)
from inferneo.sampling_params import SamplingParams


class InferneoEngine:
    def __init__(self, config: EngineConfig):
        from inferneo.executor.torch_runner import TorchModelRunner
        from inferneo.tokenizer.tokenizer import TokenizerWrapper

        self.config = config
        self.tokenizer = TokenizerWrapper(config.model)
        self.runner = TorchModelRunner(config)
        self.runner.load_model()
        num_blocks = self.runner.init_kv_cache()
        self.max_model_len = self.runner.max_model_len

        kv = KVCacheManager(
            num_blocks=num_blocks,
            block_size=config.cache.block_size,
            enable_caching=config.cache.enable_prefix_caching,
        )
        self.scheduler = Scheduler(config.scheduler, kv, self.max_model_len)

    @classmethod
    def from_model(cls, model: str, **kwargs) -> InferneoEngine:
        return cls(build_engine_config(model, **kwargs))

    # ------------------------------------------------------------------ #

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> str:
        params = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            prompt_text, prompt_ids = prompt, self.tokenizer.encode(prompt)
        else:
            prompt_text, prompt_ids = None, list(prompt)
        if not prompt_ids:
            raise ValueError("empty prompt")
        if len(prompt_ids) >= self.max_model_len:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens but max_model_len is "
                f"{self.max_model_len}"
            )
        req = EngineRequest(
            request_id=request_id or uuid.uuid4().hex,
            prompt_token_ids=prompt_ids,
            sampling_params=params,
            eos_token_id=self.tokenizer.eos_token_id,
            prompt=prompt_text,
        )
        self.scheduler.add_request(req)
        return req.request_id

    def abort_request(self, request_id: str) -> None:
        self.scheduler.abort(request_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> list[RequestOutput]:
        """One engine iteration: schedule -> execute -> update -> outputs."""
        scheduler_output = self.scheduler.schedule()
        if not scheduler_output.scheduled and not scheduler_output.finished_ids:
            return []
        runner_output = self.runner.execute(scheduler_output)
        updated = self.scheduler.update_from_output(scheduler_output, runner_output)

        outputs = []
        for req in updated:
            logprob = runner_output.logprobs.get(req.request_id)
            outputs.append(self._to_request_output(req, logprob))
        return outputs

    # ------------------------------------------------------------------ #

    def _to_request_output(self, req: EngineRequest, new_logprob) -> RequestOutput:
        completion = CompletionOutput(
            index=0,
            text="",  # detokenized by the caller (LLM / output processor)
            token_ids=list(req.output_token_ids),
            finish_reason=req.status.finish_reason,
        )
        if new_logprob is not None:
            completion.logprobs = [new_logprob]  # this step's token only
        return RequestOutput(
            request_id=req.request_id,
            prompt=req.prompt,
            prompt_token_ids=req.prompt_token_ids,
            outputs=[completion],
            finished=req.is_finished,
            metrics=RequestMetrics(
                arrival_time=req.arrival_time,
                first_scheduled_time=req.first_scheduled_time,
                first_token_time=req.first_token_time,
                finished_time=req.finished_time,
            ),
        )


def build_engine_config(
    model: str,
    *,
    dtype: str = "auto",
    device: str = "auto",
    max_model_len: int | None = None,
    block_size: int = 16,
    num_blocks: int | None = None,
    gpu_memory_utilization: float = 0.90,
    enable_prefix_caching: bool = False,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 2048,
    seed: int | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> EngineConfig:
    from inferneo.config import CacheConfig, SchedulerConfig

    return EngineConfig(
        model=ModelConfig(
            model=model,
            revision=revision,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        ),
        cache=CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
        ),
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        device=device,
        seed=seed,
    )
