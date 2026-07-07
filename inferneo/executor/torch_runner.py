"""PyTorch model runner: flat varlen batch prep, paged KV pool, forward, sample.

Implements the ``ModelRunner`` protocol from the control plane. All scheduled
tokens for a step run in ONE forward pass — no padding, no [batch, seq]
rectangles; each request's rows attend to its own paged KV via the attention
metadata. The only GPU->CPU sync per step is fetching the sampled tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

import torch

from inferneo.attention.selector import get_attention_backend
from inferneo.config import EngineConfig
from inferneo.engine.interfaces import ModelRunnerOutput, SchedulerOutput
from inferneo.models.loader import load_hf_config, load_model
from inferneo.sampling.sampler import RequestSamplerState, Sampler

# Tokens of KV capacity to allocate by default on cpu/mps, where there is no
# reliable free-memory query. Override with CacheConfig.num_blocks.
_DEFAULT_NON_CUDA_KV_TOKENS = 16384


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


@dataclass
class _KVShape:
    num_layers: int
    num_kv_heads: int
    head_dim: int

    def block_bytes(self, block_size: int, dtype: torch.dtype) -> int:
        return (
            2 * self.num_layers * block_size * self.num_kv_heads * self.head_dim
            * dtype.itemsize
        )


class TorchModelRunner:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.model.dtype, self.device)
        self._states: dict[str, RequestSamplerState] = {}
        self.kv_caches: list[torch.Tensor] = []

    def load_model(self) -> None:
        hf_config = load_hf_config(self.config.model)
        self.hf_config = hf_config
        limit = self.config.model.max_model_len
        self.max_model_len = min(
            hf_config.max_position_embeddings, limit or hf_config.max_position_embeddings
        )
        num_heads = hf_config.num_attention_heads
        num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // num_heads
        )
        self._kv_shape = _KVShape(hf_config.num_hidden_layers, num_kv_heads, head_dim)
        self.backend = get_attention_backend(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=self.config.cache.block_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.model = load_model(
            self.config.model, hf_config, self.backend, self.device, self.dtype
        )
        self.sampler = Sampler(self.config.seed)

    def init_kv_cache(self) -> int:
        cache_cfg = self.config.cache
        block_size = cache_cfg.block_size
        if cache_cfg.num_blocks is not None:
            num_blocks = cache_cfg.num_blocks
        elif self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            already_used = total - free
            budget = int(total * cache_cfg.gpu_memory_utilization) - already_used
            num_blocks = budget // self._kv_shape.block_bytes(block_size, self.dtype)
        else:
            num_blocks = -(-_DEFAULT_NON_CUDA_KV_TOKENS // block_size)
        if num_blocks < 4:
            raise RuntimeError(
                f"only {num_blocks} KV blocks fit; lower gpu_memory_utilization "
                f"pressure or set CacheConfig.num_blocks explicitly"
            )
        self.kv_caches = [
            self.backend.make_kv_cache(num_blocks)
            for _ in range(self._kv_shape.num_layers)
        ]
        return num_blocks

    @torch.inference_mode()
    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        for rid in scheduler_output.finished_ids:
            self._states.pop(rid, None)
        for rid in scheduler_output.preempted_ids:
            self._states.pop(rid, None)
        scheduled = scheduler_output.scheduled
        if not scheduled:
            return ModelRunnerOutput()

        input_ids: list[int] = []
        positions: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        block_tables: list[list[int]] = []
        for s in scheduled:
            if s.is_new:
                generator = None
                if s.sampling_params.seed is not None:
                    generator = torch.Generator().manual_seed(s.sampling_params.seed)
                self._states[s.request_id] = RequestSamplerState(
                    params=s.sampling_params,
                    prompt_len=s.prompt_len,
                    token_ids=list(s.cached_prefix_token_ids),
                    generator=generator,
                )
            state = self._states[s.request_id]
            assert len(state.token_ids) == s.start_pos, (
                f"runner out of sync for {s.request_id}: "
                f"{len(state.token_ids)} != {s.start_pos}"
            )
            state.token_ids.extend(s.chunk_token_ids)
            input_ids.extend(s.chunk_token_ids)
            positions.extend(range(s.start_pos, s.start_pos + s.num_new_tokens))
            query_lens.append(s.num_new_tokens)
            seq_lens.append(s.start_pos + s.num_new_tokens)
            block_tables.append(s.block_ids)

        metadata = self.backend.build_metadata(query_lens, seq_lens, block_tables)
        input_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        pos_t = torch.tensor(positions, dtype=torch.long, device=self.device)
        hidden = self.model(input_t, pos_t, self.kv_caches, metadata)

        row_ends = list(accumulate(query_lens))
        rows, sample_ids, sample_states = [], [], []
        for s, end in zip(scheduled, row_ends):
            if s.do_sample:
                rows.append(end - 1)
                sample_ids.append(s.request_id)
                sample_states.append(self._states[s.request_id])
        if not rows:
            return ModelRunnerOutput()

        logits = self.model.compute_logits(
            hidden[torch.tensor(rows, dtype=torch.long, device=self.device)]
        )
        tokens, logprobs = self.sampler.sample(logits, sample_states)

        out = ModelRunnerOutput()
        for rid, token, logprob in zip(sample_ids, tokens, logprobs):
            out.sampled[rid] = token
            if logprob is not None:
                out.logprobs[rid] = logprob
        return out
