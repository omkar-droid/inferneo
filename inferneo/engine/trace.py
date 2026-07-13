"""Optional step-by-step trace of the scheduler, for understanding and debugging.

Enable with ``INFERNEO_TRACE=1``. Off by default and a no-op when off, so it costs
nothing in production. Each line is one engine step:

    step   7 | run=3 wait=1 | kv 47/128 (37%) | a:P64* b:D1 c:D1

    P64*  prefilled 64 prompt tokens, and * = the chunk finished, so it sampled
    P512  a *chunked* prefill — only part of the prompt, no token sampled yet
    D1    decoded 1 token (steady state)
    !x    request x was preempted (its KV blocks were reclaimed)
"""

from __future__ import annotations

import os
import sys

ENABLED = os.environ.get("INFERNEO_TRACE", "") not in ("", "0", "false")


def step(n: int, scheduler_output, scheduler) -> None:
    if not ENABLED:
        return
    pool = scheduler.kv.pool
    used = pool.num_blocks - pool.num_free_blocks
    pct = 100 * used / pool.num_blocks if pool.num_blocks else 0

    parts = []
    for s in scheduler_output.scheduled:
        rid = s.request_id[-4:]  # short id, readable
        if s.num_new_tokens == 1 and not s.is_new:
            parts.append(f"{rid}:D1")
        else:
            parts.append(f"{rid}:P{s.num_new_tokens}{'*' if s.do_sample else ''}")
    for rid in scheduler_output.preempted_ids:
        parts.append(f"!{rid[-4:]}")

    print(
        f"step {n:>4} | run={len(scheduler.running)} wait={len(scheduler.waiting)} "
        f"| kv {used}/{pool.num_blocks} ({pct:.0f}%) | {' '.join(parts)}",
        file=sys.stderr,
        flush=True,
    )
