"""Inferneo: a research testbed for LLM inference serving."""

from typing import TYPE_CHECKING

from inferneo.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from inferneo.outputs import CompletionOutput, RequestOutput
from inferneo.sampling_params import SamplingParams

if TYPE_CHECKING:
    from inferneo.engine.llm import LLM

__version__ = "0.1.0"

__all__ = [
    "LLM",
    "CacheConfig",
    "CompletionOutput",
    "EngineConfig",
    "ModelConfig",
    "RequestOutput",
    "SamplingParams",
    "SchedulerConfig",
    "__version__",
]


def __getattr__(name: str):
    # Lazy: importing `inferneo` must not pull in torch; `LLM` does.
    if name == "LLM":
        from inferneo.engine.llm import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
