"""Engine configuration.

Plain dataclasses only — the control plane (engine/, kv/) reads these and must
stay importable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Which model to serve and how to interpret it."""

    model: str  # HF repo id or local path
    revision: str | None = None
    dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    max_model_len: int | None = None  # None => from the model's HF config
    trust_remote_code: bool = False


@dataclass
class CacheConfig:
    """Paged KV cache sizing and behavior."""

    block_size: int = 16  # tokens per KV block
    num_blocks: int | None = None  # None => sized from device memory (cuda) or default
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = False


@dataclass
class SchedulerConfig:
    """Token-budget scheduling limits (vLLM-V1-style unified scheduling)."""

    max_num_seqs: int = 64  # max requests running concurrently
    max_num_batched_tokens: int = 2048  # per-step token budget
    # Cap on prompt tokens a single request may consume in one step (chunked
    # prefill). 0 disables the cap; the budget still chunks long prefills.
    long_prefill_token_threshold: int = 0


@dataclass
class EngineConfig:
    """Top-level engine configuration."""

    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    seed: int | None = None  # engine-level RNG seed for unseeded sampling
    enable_cuda_graph: bool = True  # graph-capture pure-decode steps on CUDA

    def __post_init__(self) -> None:
        if self.cache.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.scheduler.max_num_batched_tokens < self.cache.block_size:
            raise ValueError("max_num_batched_tokens must be >= block_size")
