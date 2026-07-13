"""Async wrapper around the synchronous engine, for the server.

The engine step loop is CPU/GPU-bound and single-threaded by design, so it
runs in one background thread. FastAPI request handlers await per-request
asyncio queues that the loop feeds via ``call_soon_threadsafe``. All engine
state mutation (add/abort/step) happens on the loop thread — the queues and a
small lock-protected inbox are the only cross-thread contact points.

The data crossing the thread boundary is plain ``RequestOutput`` objects, and
the server talks to this class through the ``EngineClient`` shape, so promoting
the loop to a separate process later (ZMQ) touches nothing above it.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator

from inferneo.config import EngineConfig
from inferneo.engine.engine import InferneoEngine, build_engine_config
from inferneo.outputs import RequestOutput
from inferneo.sampling_params import SamplingParams

_IDLE_SLEEP = 0.002  # seconds the loop parks when there is no work


class AsyncEngine:
    def __init__(self, config: EngineConfig):
        self.engine = InferneoEngine(config)
        self.tokenizer = self.engine.tokenizer
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._inbox: deque[tuple[str, object, SamplingParams]] = deque()
        self._aborts: deque[str] = deque()
        self._queues: dict[str, asyncio.Queue] = {}
        self._error: BaseException | None = None

    @classmethod
    def from_model(cls, model: str, **kwargs) -> AsyncEngine:
        return cls(build_engine_config(model, **kwargs))

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._ensure_thread()

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="inferneo-engine", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        engine = self.engine
        try:
            while self._running:
                with self._lock:
                    while self._inbox:
                        rid, prompt, params = self._inbox.popleft()
                        try:
                            engine.add_request(prompt, params, rid)
                        except Exception as exc:  # noqa: BLE001
                            # A bad request (e.g. prompt longer than max_model_len)
                            # must fail only *that* request — never take down the
                            # engine thread and with it everyone else's requests.
                            self._dispatch_to(rid, exc)
                    while self._aborts:
                        engine.abort_request(self._aborts.popleft())
                if not engine.has_unfinished():
                    time.sleep(_IDLE_SLEEP)
                    continue
                for out in engine.step():
                    self._dispatch(out)
        except BaseException as exc:  # noqa: BLE001 — surface to clients, don't hang
            self._fail_all(exc)

    def _dispatch(self, item) -> None:
        self._dispatch_to(getattr(item, "request_id", None), item)

    def _dispatch_to(self, request_id: str | None, item) -> None:
        queue = self._queues.get(request_id)
        if queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(queue.put_nowait, item)

    def _fail_all(self, exc: BaseException) -> None:
        """A crash in the engine thread must wake every waiter with the error
        and fail fast on any later request, rather than hang."""
        self._running = False
        self._error = exc
        if self._loop is None:
            return
        for queue in list(self._queues.values()):
            crash = RuntimeError("inferneo engine thread crashed")
            crash.__cause__ = exc
            self._loop.call_soon_threadsafe(queue.put_nowait, crash)

    # ------------------------------------------------------------------ #
    # EngineClient surface
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncIterator[RequestOutput]:
        if self._error is not None:
            raise RuntimeError("inferneo engine thread has crashed") from self._error
        request_id = request_id or uuid.uuid4().hex
        # Bind dispatch to the loop that owns this queue (the request's loop),
        # and make sure the engine thread is running.
        self._loop = asyncio.get_running_loop()
        self._ensure_thread()
        queue: asyncio.Queue[RequestOutput] = asyncio.Queue()
        self._queues[request_id] = queue
        with self._lock:
            self._inbox.append((request_id, prompt, sampling_params))
        try:
            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item  # a bad-request error, or the engine-crash RuntimeError
                yield item
                if item.finished:
                    return
        finally:
            # Client hung up (or we finished): stop work and drop the queue.
            self._queues.pop(request_id, None)
            with self._lock:
                self._aborts.append(request_id)

    async def abort(self, request_id: str) -> None:
        with self._lock:
            self._aborts.append(request_id)

    @property
    def model_name(self) -> str:
        return self.engine.config.model.model

    @property
    def max_model_len(self) -> int:
        return self.engine.max_model_len
