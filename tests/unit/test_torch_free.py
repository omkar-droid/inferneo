"""The control plane must import without torch — this is the layering rule
that keeps scheduler/KV research hackable and CI cheap."""

import subprocess
import sys

CONTROL_PLANE = [
    "inferneo",
    "inferneo.config",
    "inferneo.sampling_params",
    "inferneo.outputs",
    "inferneo.engine.request",
    "inferneo.engine.scheduler",
    "inferneo.engine.interfaces",
    "inferneo.kv.block_pool",
    "inferneo.kv.block_manager",
    "inferneo.kv.hashing",
]


def test_control_plane_imports_without_torch():
    code = (
        f"import {', '.join(CONTROL_PLANE)}, sys; "
        "assert 'torch' not in sys.modules, 'control plane pulled in torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
