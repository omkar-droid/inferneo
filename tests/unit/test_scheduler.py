"""Scheduler tests — no torch. A fake runner echoes a fixed next token so we
can exercise batching, chunked prefill, preemption, and finish handling."""

from inferneo.config import SchedulerConfig
from inferneo.engine.interfaces import ModelRunnerOutput
from inferneo.engine.request import EngineRequest, RequestStatus
from inferneo.engine.scheduler import Scheduler
from inferneo.kv.block_manager import KVCacheManager
from inferneo.sampling_params import SamplingParams


def build(num_blocks=64, block_size=4, max_model_len=256, **sched_kwargs):
    kv = KVCacheManager(num_blocks, block_size)
    cfg = SchedulerConfig(**sched_kwargs)
    return Scheduler(cfg, kv, max_model_len)


def add(sched, rid, num_prompt, max_tokens=8, priority=0):
    req = EngineRequest(
        request_id=rid,
        prompt_token_ids=list(range(num_prompt)),
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        priority=priority,
    )
    sched.add_request(req)
    return req


def fake_output(scheduler_output, token=7):
    out = ModelRunnerOutput()
    for s in scheduler_output.scheduled:
        if s.do_sample:
            out.sampled[s.request_id] = token
    return out


def test_prompt_then_decode():
    sched = build(max_num_batched_tokens=64)
    add(sched, "a", num_prompt=5, max_tokens=3)

    so = sched.schedule()
    assert len(so.scheduled) == 1
    s = so.scheduled[0]
    assert s.num_new_tokens == 5 and s.start_pos == 0 and s.do_sample
    sched.update_from_output(so, fake_output(so))

    # Now a pure decode step: exactly one token.
    so = sched.schedule()
    assert so.scheduled[0].num_new_tokens == 1
    assert so.scheduled[0].start_pos == 5


def test_chunked_prefill_by_budget():
    sched = build(max_num_batched_tokens=4, block_size=4)
    add(sched, "a", num_prompt=10)
    chunks, sampled_flags = [], []
    for _ in range(3):
        so = sched.schedule()
        s = so.scheduled[0]
        chunks.append(s.num_new_tokens)
        sampled_flags.append(s.do_sample)
        sched.update_from_output(so, fake_output(so))
        if s.do_sample:
            break
    assert chunks == [4, 4, 2]  # 10 prompt tokens chunked by a budget of 4
    assert sampled_flags == [False, False, True]  # only the last chunk samples


def test_running_served_before_waiting():
    sched = build(max_num_batched_tokens=6, max_num_seqs=8)
    add(sched, "a", num_prompt=3)
    so = sched.schedule()
    sched.update_from_output(so, fake_output(so))  # a is running/decoding

    add(sched, "b", num_prompt=3)
    so = sched.schedule()
    ids = [s.request_id for s in so.scheduled]
    assert ids[0] == "a"  # decode of a first
    assert so.scheduled[0].num_new_tokens == 1
    assert "b" in ids  # b admitted with remaining budget


def test_max_num_seqs_caps_concurrency():
    sched = build(max_num_batched_tokens=100, max_num_seqs=2)
    for rid in ("a", "b", "c"):
        add(sched, rid, num_prompt=2)
    so = sched.schedule()
    assert len(so.scheduled) == 2
    assert len(sched.waiting) == 1


def test_preemption_under_block_pressure():
    # 4 blocks * size 4 = 16 token capacity. Two long requests can't coexist.
    sched = build(num_blocks=4, block_size=4, max_model_len=64,
                  max_num_batched_tokens=100, max_num_seqs=8)
    add(sched, "a", num_prompt=8, max_tokens=20)
    add(sched, "b", num_prompt=8, max_tokens=20)

    # Admit both (16 tokens = exactly 4 blocks).
    so = sched.schedule()
    assert len(so.scheduled) == 2
    sched.update_from_output(so, fake_output(so))

    # Next decode needs a 5th block -> preempt the newer request (b).
    so = sched.schedule()
    assert so.preempted_ids == ["b"]
    assert sched.requests["b"].status == RequestStatus.PREEMPTED
    # a keeps going.
    assert any(s.request_id == "a" for s in so.scheduled)


def test_full_run_to_completion():
    sched = build(max_num_batched_tokens=64)
    add(sched, "a", num_prompt=3, max_tokens=5)
    tokens = []
    while sched.has_unfinished():
        so = sched.schedule()
        updated = sched.update_from_output(so, fake_output(so, token=42))
        for req in updated:
            if req.output_token_ids:
                tokens = req.output_token_ids
    assert tokens == [42] * 5
    assert not sched.requests  # cleaned up


def test_eos_stops_generation():
    sched = build(max_num_batched_tokens=64)
    req = EngineRequest(
        request_id="a",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=100),
        eos_token_id=42,
    )
    sched.add_request(req)
    so = sched.schedule()
    sched.update_from_output(so, fake_output(so, token=42))
    assert req.status == RequestStatus.FINISHED_STOPPED
    assert not sched.has_unfinished()


def test_abort_running_frees_blocks():
    sched = build(num_blocks=8, block_size=4, max_num_batched_tokens=64)
    add(sched, "a", num_prompt=6)
    so = sched.schedule()
    sched.update_from_output(so, fake_output(so))  # a running, holds 2 blocks
    free_before = sched.kv.pool.num_free_blocks

    sched.abort("a")
    assert "a" not in sched.requests  # finished + cleaned up
    assert sched.kv.pool.num_free_blocks > free_before
    assert not sched.has_unfinished()


def test_abort_waiting_request():
    sched = build(max_num_batched_tokens=64, max_num_seqs=1)
    add(sched, "a", num_prompt=3)
    add(sched, "b", num_prompt=3)
    sched.schedule()  # only a admitted (max_num_seqs=1); b waits
    sched.abort("b")
    assert "b" not in sched.requests
    assert all(r.request_id != "b" for r in sched.waiting)


# ---- priority scheduling ----


def test_high_priority_admitted_first():
    """With one admission slot per step, the highest-priority waiting request must
    be admitted before lower-priority ones, regardless of arrival order."""
    sched = build(max_num_batched_tokens=64, max_num_seqs=1)
    add(sched, "low1", num_prompt=3, priority=0)
    add(sched, "low2", num_prompt=3, priority=0)
    add(sched, "urgent", num_prompt=3, priority=10)  # arrived LAST, highest priority

    so = sched.schedule()
    assert [s.request_id for s in so.scheduled] == ["urgent"]  # jumped the queue


def test_same_priority_is_fcfs():
    """Ties broken by arrival order — priority must not scramble FCFS."""
    sched = build(max_num_batched_tokens=64, max_num_seqs=1)
    add(sched, "first", num_prompt=3, priority=5)
    add(sched, "second", num_prompt=3, priority=5)
    so = sched.schedule()
    assert [s.request_id for s in so.scheduled] == ["first"]


def test_priority_orders_a_backlog():
    """A full backlog drains in priority order as slots free up."""
    sched = build(max_num_batched_tokens=64, max_num_seqs=1)
    add(sched, "p1", num_prompt=2, max_tokens=1, priority=1)
    add(sched, "p5", num_prompt=2, max_tokens=1, priority=5)
    add(sched, "p3", num_prompt=2, max_tokens=1, priority=3)

    admitted = []
    for _ in range(3):
        so = sched.schedule()
        admitted += [s.request_id for s in so.scheduled if s.is_new]
        sched.update_from_output(so, fake_output(so))  # finishes (max_tokens=1) -> frees the slot
    assert admitted == ["p5", "p3", "p1"]  # strictly high -> low


# ---- preemptive priority ----


def test_priority_preempts_when_slots_full():
    """When the only slot is held by a low-priority job, a higher-priority arrival
    evicts it rather than waiting for it to finish."""
    sched = build(max_num_batched_tokens=64, max_num_seqs=1)
    add(sched, "low", num_prompt=3, max_tokens=50, priority=0)
    sched.update_from_output(so := sched.schedule(), fake_output(so))  # low is running

    add(sched, "urgent", num_prompt=3, max_tokens=5, priority=10)
    so = sched.schedule()
    assert so.preempted_ids == ["low"]                       # low evicted
    assert any(s.request_id == "urgent" and s.is_new for s in so.scheduled)  # urgent admitted now
    assert sched.requests["low"].status == RequestStatus.PREEMPTED


def test_no_thrash_after_priority_preemption():
    """The evicted request must NOT immediately evict the one it made way for —
    strict '>' means the (now lower-ranked) victim can't preempt back."""
    sched = build(max_num_batched_tokens=64, max_num_seqs=1)
    add(sched, "low", num_prompt=3, max_tokens=50, priority=0)
    sched.update_from_output(so := sched.schedule(), fake_output(so))
    add(sched, "urgent", num_prompt=3, max_tokens=50, priority=10)
    sched.update_from_output(so := sched.schedule(), fake_output(so))  # urgent preempts low

    for _ in range(5):                                       # several steps: must stay put
        so = sched.schedule()
        assert so.preempted_ids == []                       # no back-and-forth
        assert any(s.request_id == "urgent" for s in so.scheduled)
        sched.update_from_output(so, fake_output(so))
    assert sched.requests["urgent"].status == RequestStatus.RUNNING
    assert sched.requests["low"].status == RequestStatus.PREEMPTED


def test_equal_priority_does_not_preempt():
    """Only a *strictly* higher priority preempts — equal priority waits (FCFS)."""
    sched = build(max_num_batched_tokens=64, max_num_seqs=1)
    add(sched, "a", num_prompt=3, max_tokens=50, priority=5)
    sched.update_from_output(so := sched.schedule(), fake_output(so))
    add(sched, "b", num_prompt=3, max_tokens=50, priority=5)
    so = sched.schedule()
    assert so.preempted_ids == []                           # a keeps its slot
    assert sched.requests["a"].status == RequestStatus.RUNNING


def test_preemptive_priority_both_finish():
    """Stability: urgent runs to completion, then the preempted low resumes and
    finishes too — no hang, no lost request."""
    sched = build(num_blocks=64, block_size=4, max_num_batched_tokens=64, max_num_seqs=1)
    add(sched, "low", num_prompt=3, max_tokens=4, priority=0)
    sched.update_from_output(so := sched.schedule(), fake_output(so))
    add(sched, "urgent", num_prompt=3, max_tokens=4, priority=10)
    done = []
    for _ in range(40):
        if not sched.has_unfinished():
            break
        so = sched.schedule()
        for r in sched.update_from_output(so, fake_output(so)):
            if r.is_finished:
                done.append(r.request_id)
    assert set(done) == {"urgent", "low"}                   # both completed
    assert done.index("urgent") < done.index("low")         # urgent finished first
