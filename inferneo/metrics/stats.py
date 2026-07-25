"""Runtime stats for the live dashboard and Prometheus.

Torch-free by design: the engine feeds one sample per step, and GPU numbers come
from an injected probe (the torch runner), so this stays importable without torch
and unit-testable on CPU. A short rolling window gives instantaneous rates and
latency percentiles; monotonic counters feed Prometheus.

Thread-safety: the engine thread calls ``record_step``; HTTP handlers call
``snapshot`` from another thread. A single lock guards the shared deques/counters.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


class StatsCollector:
    """One sample per engine step -> windowed rates + cumulative counters."""

    def __init__(self, gpu_probe: Callable[[], dict | None] | None = None,
                 window_s: float = 30.0):
        self._gpu_probe = gpu_probe
        self._window = window_s
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        # instantaneous gauges (last recorded step)
        self.running = 0
        self.waiting = 0
        self.kv_usage = 0.0
        # cumulative counters
        self.total_steps = 0
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_finished = 0
        self.total_preemptions = 0
        # rolling windows of (timestamp, value)
        self._steps: deque[tuple[float, int, int]] = deque()  # (t, gen, processed)
        self._ttft: deque[tuple[float, float]] = deque()      # (t, ms)
        self._tpot: deque[tuple[float, float]] = deque()
        self._e2e: deque[tuple[float, float]] = deque()

    def record_step(
        self,
        *,
        running: int,
        waiting: int,
        kv_usage: float,
        processed_tokens: int,
        generation_tokens: int,
        preemptions: int,
        finished_latencies: Iterable[tuple[float | None, float | None, float | None]] = (),
    ) -> None:
        """Record one engine step. ``finished_latencies`` is (ttft, tpot, e2e) in
        *seconds* for each request that finished this step (any may be None)."""
        now = time.monotonic()
        with self._lock:
            self.running = running
            self.waiting = waiting
            self.kv_usage = kv_usage
            self.total_steps += 1
            self.total_prompt_tokens += max(processed_tokens - generation_tokens, 0)
            self.total_generation_tokens += generation_tokens
            self.total_preemptions += preemptions
            self._steps.append((now, generation_tokens, processed_tokens))
            for ttft, tpot, e2e in finished_latencies:
                self.total_finished += 1
                if ttft is not None:
                    self._ttft.append((now, ttft * 1e3))
                if tpot is not None:
                    self._tpot.append((now, tpot * 1e3))
                if e2e is not None:
                    self._e2e.append((now, e2e * 1e3))
            self._evict(now)

    def _evict(self, now: float) -> None:
        cut = now - self._window
        for dq in (self._steps, self._ttft, self._tpot, self._e2e):
            while dq and dq[0][0] < cut:
                dq.popleft()

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            span = max(now - self._steps[0][0], 1e-6) if self._steps else 1.0
            gen = sum(s[1] for s in self._steps)
            proc = sum(s[2] for s in self._steps)
            ttfts = sorted(v for _, v in self._ttft)
            tpots = sorted(v for _, v in self._tpot)
            e2es = sorted(v for _, v in self._e2e)
            snap = {
                "uptime_s": now - self._t0,
                "running": self.running,
                "waiting": self.waiting,
                "kv_cache_usage": self.kv_usage,
                "generation_tps": gen / span,
                "processed_tps": proc / span,
                "ttft_ms": {"p50": _pct(ttfts, 0.5), "p99": _pct(ttfts, 0.99)},
                "tpot_ms": {"p50": _pct(tpots, 0.5), "p99": _pct(tpots, 0.99)},
                "e2e_ms": {"p50": _pct(e2es, 0.5), "p99": _pct(e2es, 0.99)},
                "totals": {
                    "steps": self.total_steps,
                    "prompt_tokens": self.total_prompt_tokens,
                    "generation_tokens": self.total_generation_tokens,
                    "finished_requests": self.total_finished,
                    "preemptions": self.total_preemptions,
                },
            }
        gpu = self._gpu_probe() if self._gpu_probe else None
        if gpu:
            snap["gpu"] = gpu
        return snap


def prometheus_text(snap: dict) -> str:
    """Render a snapshot in Prometheus text exposition format (v0.0.4)."""
    lines: list[str] = []

    def metric(name: str, val: float, help_: str, kind: str) -> None:
        lines.append(f"# HELP {name} {help_}")
        lines.append(f"# TYPE {name} {kind}")
        lines.append(f"{name} {val}")

    def gauge(name, val, help_):
        metric(name, val, help_, "gauge")

    def counter(name, val, help_):
        metric(name, val, help_, "counter")

    gauge("inferneo_running_requests", snap["running"], "Requests currently decoding")
    gauge("inferneo_waiting_requests", snap["waiting"], "Requests queued for admission")
    gauge("inferneo_kv_cache_usage_ratio", round(snap["kv_cache_usage"], 6),
          "Fraction of KV-cache blocks in use")
    gauge("inferneo_generation_tokens_per_second", round(snap["generation_tps"], 3),
          "Output tokens per second (rolling window)")
    for stat in ("ttft", "tpot", "e2e"):
        for p in ("p50", "p99"):
            gauge(f"inferneo_{stat}_ms_{p}", round(snap[f"{stat}_ms"][p], 3),
                  f"{stat.upper()} {p} in milliseconds (rolling window)")

    t = snap["totals"]
    counter("inferneo_generation_tokens_total", t["generation_tokens"],
            "Output tokens generated since start")
    counter("inferneo_prompt_tokens_total", t["prompt_tokens"],
            "Prompt tokens processed since start")
    counter("inferneo_finished_requests_total", t["finished_requests"],
            "Requests completed since start")
    counter("inferneo_preemptions_total", t["preemptions"],
            "Request preemptions since start")

    gpu = snap.get("gpu")
    if gpu:
        gauge("inferneo_gpu_memory_used_bytes", gpu["mem_used_bytes"], "GPU memory in use")
        gauge("inferneo_gpu_memory_total_bytes", gpu["mem_total_bytes"], "Total GPU memory")
        if "sm_util" in gpu:
            gauge("inferneo_gpu_sm_utilization_ratio", round(gpu["sm_util"], 4),
                  "GPU SM utilization")
        if "power_w" in gpu:
            gauge("inferneo_gpu_power_watts", round(gpu["power_w"], 1), "GPU power draw")
        if "temp_c" in gpu:
            gauge("inferneo_gpu_temperature_celsius", gpu["temp_c"], "GPU temperature")
    return "\n".join(lines) + "\n"
