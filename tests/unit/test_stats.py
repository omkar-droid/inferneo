"""StatsCollector is torch-free and thread-safe: one sample per engine step ->
windowed rates, latency percentiles, cumulative counters, Prometheus text."""

import inspect

import pytest

from inferneo.metrics.stats import StatsCollector, prometheus_text


def test_records_gauges_and_totals():
    c = StatsCollector()
    c.record_step(running=3, waiting=2, kv_usage=0.5, processed_tokens=10,
                  generation_tokens=3, preemptions=1)
    snap = c.snapshot()
    assert snap["running"] == 3 and snap["waiting"] == 2
    assert snap["kv_cache_usage"] == 0.5
    assert snap["totals"]["generation_tokens"] == 3
    assert snap["totals"]["prompt_tokens"] == 7  # processed(10) - generation(3)
    assert snap["totals"]["preemptions"] == 1
    assert snap["generation_tps"] > 0


def test_latency_percentiles_from_finished():
    c = StatsCollector()
    # ttft in seconds -> collector stores ms: 10,20,30,40,50
    lat = [(t / 1000, None, None) for t in (10, 20, 30, 40, 50)]
    c.record_step(running=0, waiting=0, kv_usage=0.0, processed_tokens=5,
                  generation_tokens=5, preemptions=0, finished_latencies=lat)
    snap = c.snapshot()
    assert snap["totals"]["finished_requests"] == 5
    assert snap["ttft_ms"]["p50"] == 30
    assert snap["ttft_ms"]["p99"] == 50


def test_derived_metrics_and_mfu():
    info = {"flops_per_token": 2_000_000_000, "peak_flops": 800e12,
            "gpu_name": "NVIDIA H100 NVL", "cuda_version": "12.4",
            "compute_capability": "9.0", "gpu_count": 1, "vram_bytes": 100e9}
    c = StatsCollector(static_info=info)
    c.record_step(running=64, waiting=0, kv_usage=0.1, processed_tokens=70,
                  generation_tokens=64, preemptions=2)
    snap = c.snapshot()
    assert snap["effective_batch"] == 64            # 64 gen tokens / 1 step
    assert snap["prefill_fraction"] == (70 - 64) / 70
    assert snap["preemptions_per_s"] > 0
    assert snap["achieved_tflops"] is not None
    # MFU is exactly achieved / peak (magnitude is timing-dependent in a unit test;
    # the realistic ~few-percent decode value shows up in the H100 capture).
    assert snap["mfu"] == pytest.approx(snap["achieved_tflops"] * 1e12 / 800e12)
    assert snap["info"]["gpu_name"] == "NVIDIA H100 NVL"


def test_mfu_omitted_without_peak():
    # unknown GPU (no peak) -> raw TFLOP/s shown, MFU withheld rather than faked
    c = StatsCollector(static_info={"flops_per_token": 2_000_000_000})
    c.record_step(running=8, waiting=0, kv_usage=0.0, processed_tokens=8,
                  generation_tokens=8, preemptions=0)
    snap = c.snapshot()
    assert snap["achieved_tflops"] is not None
    assert snap["mfu"] is None


def test_gpu_probe_injected_into_snapshot():
    c = StatsCollector(gpu_probe=lambda: {"mem_used_bytes": 1, "mem_total_bytes": 2,
                                          "mem_used_frac": 0.5, "sm_util": 0.7})
    assert c.snapshot()["gpu"]["sm_util"] == 0.7


def test_no_gpu_key_without_probe():
    assert "gpu" not in StatsCollector().snapshot()


def test_prometheus_text_shape():
    c = StatsCollector(gpu_probe=lambda: {"mem_used_bytes": 10, "mem_total_bytes": 20,
                                          "mem_used_frac": 0.5, "sm_util": 0.3,
                                          "power_w": 300.0, "temp_c": 55})
    c.record_step(running=1, waiting=0, kv_usage=0.25, processed_tokens=4,
                  generation_tokens=2, preemptions=0)
    text = prometheus_text(c.snapshot())
    assert "inferneo_running_requests 1" in text
    assert "inferneo_gpu_sm_utilization_ratio 0.3" in text
    assert "# TYPE inferneo_generation_tokens_total counter" in text
    assert "# TYPE inferneo_running_requests gauge" in text


def test_module_is_torch_free():
    assert "import torch" not in inspect.getsource(
        __import__("inferneo.metrics.stats", fromlist=["x"])
    )
