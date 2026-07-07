"""`inferneo serve --model ...` — launch the OpenAI-compatible server."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="inferneo", description="inferneo inference server")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the OpenAI-compatible API server")
    serve.add_argument("--model", required=True, help="HF repo id or local path")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", default="auto")
    serve.add_argument("--dtype", default="auto")
    serve.add_argument("--max-model-len", type=int, default=None)
    serve.add_argument("--max-num-seqs", type=int, default=256)
    serve.add_argument("--max-num-batched-tokens", type=int, default=8192)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    serve.add_argument("--enable-prefix-caching", action="store_true")
    serve.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()
    if args.command == "serve":
        _serve(args)


def _serve(args) -> None:
    import uvicorn

    from inferneo.engine.async_engine import AsyncEngine
    from inferneo.server.api_server import build_app

    engine = AsyncEngine.from_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        trust_remote_code=args.trust_remote_code,
    )
    app = build_app(engine)
    print(f"inferneo serving {args.model} on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
