"""Configuration for the vLLM benchmarking framework.

Centralizes every tunable parameter for a benchmark run behind a single
`BenchmarkConfig` dataclass, built either from CLI args (`parse_args`) or
constructed directly for programmatic use.

New experiment dimensions (compiled mode, prefix caching, `max_num_seqs`,
chunked prefill, etc.) are all *server-side* settings that live in how the
vLLM server itself was launched — they are invisible to this client and
require no changes here. Tag a run with `--run-name`/`--tag` to label which
server configuration it was measured against, and reuse this module as-is.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration for a single benchmark run."""

    server_url: str = "http://localhost:8000"
    model: str = "default"
    api_type: str = "completions"  # "completions" or "chat"
    api_key: str | None = None  # sent as "Authorization: Bearer <key>" when set

    num_users: int = 10
    num_requests: int = 100
    duration: float | None = None  # if set, run for this many seconds instead of a fixed request count
    warmup_requests: int = 0  # throwaway requests sent (and discarded) before timing starts

    prompts_file: Path = Path("prompts.json")
    workload: str = "text"  # "text", "long", "image", or "mixed" (capstone prompt-bank category)
    prompt_length: int | None = None  # target prompt length in words; None = use prompt as-is

    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0

    request_timeout: float = 120.0

    results_dir: Path = Path("results")
    run_name: str = "run"
    tags: dict[str, str] = field(default_factory=dict)

    log_level: str = "INFO"

    @property
    def output_csv(self) -> Path:
        """Path to the per-request CSV for this run."""
        return self.results_dir / f"{self.run_name}.csv"


def _parse_tags(raw_tags: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for item in raw_tags:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        tags[key.strip()] = value.strip()
    return tags


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (separate function so it can be reused/extended)."""
    parser = argparse.ArgumentParser(
        description="Benchmark a vLLM OpenAI-compatible inference server.",
    )
    parser.add_argument("--server-url", default="http://localhost:8000",
                         help="Base URL of the running vLLM server, e.g. http://<runpod-host>:8000")
    parser.add_argument("--model", default="default", help="Model name as registered on the server")
    parser.add_argument("--api-type", choices=["completions", "chat"], default="completions",
                         help="Which OpenAI-compatible endpoint to use")
    parser.add_argument("--api-key", default=None,
                         help="API key if the server was started with --api-key; "
                              "falls back to the OPENAI_API_KEY environment variable")

    parser.add_argument("--num-users", type=int, default=10,
                         help="Number of concurrent simulated users (concurrency level)")
    parser.add_argument("--num-requests", type=int, default=100,
                         help="Total number of requests to send across all users")

    parser.add_argument("--prompts-file", type=Path, default=Path("prompts.json"),
                         help="JSON file containing prompts to sample from")
    parser.add_argument("--prompt-length", type=int, default=None,
                         help="Target prompt length in words; prompts are truncated/repeated to match")

    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--request-timeout", type=float, default=120.0,
                         help="Per-request timeout in seconds")

    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="run", help="Used to name the output CSV file")
    parser.add_argument("--tag", action="append", default=[],
                         help="key=value metadata describing this run (repeatable), "
                              "e.g. --tag mode=compiled --tag max_num_seqs=256")

    parser.add_argument("--log-level", default="INFO")
    return parser


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    """Parse CLI arguments into a `BenchmarkConfig`."""
    args = build_arg_parser().parse_args(argv)

    return BenchmarkConfig(
        server_url=args.server_url.rstrip("/"),
        model=args.model,
        api_type=args.api_type,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        num_users=args.num_users,
        num_requests=args.num_requests,
        prompts_file=args.prompts_file,
        prompt_length=args.prompt_length,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        request_timeout=args.request_timeout,
        results_dir=args.results_dir,
        run_name=args.run_name,
        tags=_parse_tags(args.tag),
        log_level=args.log_level,
    )
