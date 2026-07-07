"""Synchronous offline API: LLM("model").generate(prompts, params)."""

from __future__ import annotations

from inferneo.engine.engine import InferneoEngine
from inferneo.outputs import RequestOutput
from inferneo.sampling_params import SamplingParams


class LLM:
    """Offline batch inference over the paged continuous-batching engine.

    Example:
        llm = LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        outs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=32))
    """

    def __init__(self, model: str, **kwargs):
        self.engine = InferneoEngine.from_model(model, **kwargs)

    def generate(
        self,
        prompts: str | list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[RequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        n = len(prompts)
        if isinstance(sampling_params, list):
            if len(sampling_params) != n:
                raise ValueError("need one SamplingParams per prompt")
            params_list = sampling_params
        else:
            params_list = [sampling_params or SamplingParams()] * n

        order: dict[str, int] = {}
        for i, (prompt, params) in enumerate(zip(prompts, params_list)):
            rid = self.engine.add_request(prompt, params)
            order[rid] = i

        finished: dict[str, RequestOutput] = {}
        while self.engine.has_unfinished():
            for out in self.engine.step():
                if out.finished:
                    finished[out.request_id] = out

        results = sorted(finished.values(), key=lambda o: order[o.request_id])
        for out in results:
            self._finalize_text(out, params_list[order[out.request_id]])
        return results

    def _finalize_text(self, out: RequestOutput, params: SamplingParams) -> None:
        """Detokenize and apply stop-string truncation post-hoc.

        Phase 1 does not stop generation early on stop *strings* (only stop
        token ids / eos); the text is truncated at the first stop string here.
        """
        for completion in out.outputs:
            text = self.engine.tokenizer.decode(completion.token_ids)
            for s in params.stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
                    completion.finish_reason = "stop"
            completion.text = text
