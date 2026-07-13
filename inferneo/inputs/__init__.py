"""Multimodal preprocessing: everything that turns a user's request into the one
common input the engine understands.

    text  → token ids
    image → vision tower → embeddings spliced into the sequence
    audio → (needs an ASR model; not loaded)

The engine below this layer only ever sees an `EngineInput`. It has no idea
whether a picture was involved — which is exactly the point: no modality logic
leaks into AsyncEngine.
"""

from inferneo.inputs.types import EngineInput

__all__ = ["EngineInput"]
