"""Async client for the vLLM OpenAI-compatible API.

Streams a single completion request and records per-token arrival times,
from which the orchestrator derives TTFT, inter-token latency, and total
tokens generated. Supports both the `/v1/completions` and
`/v1/chat/completions` endpoints so the same client works regardless of
how a given experiment prefers to prompt the server.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import aiohttp

from config import BenchmarkConfig
from metrics import RequestMetrics

logger = logging.getLogger("benchmark.client")


class VLLMClient:
    """Thin async wrapper around a vLLM OpenAI-compatible server.

    Holds no per-request state; a single instance is safe to share across
    concurrently running workers as long as they each pass their own
    `aiohttp.ClientSession` calls through the shared connector.
    """

    def __init__(self, config: BenchmarkConfig, session: aiohttp.ClientSession) -> None:
        self._config = config
        self._session = session

    def _endpoint(self) -> str:
        path = "/v1/chat/completions" if self._config.api_type == "chat" else "/v1/completions"
        return f"{self._config.server_url}{path}"

    def _headers(self) -> dict[str, str]:
        """Auth header, only present when the server requires an API key."""
        if self._config.api_key:
            return {"Authorization": f"Bearer {self._config.api_key}"}
        return {}

    def _payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._config.api_type == "chat":
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        return payload

    @staticmethod
    def _extract_text_and_usage(chunk: dict[str, Any], api_type: str) -> tuple[str, Optional[dict]]:
        """Pull the incremental token text and any usage block out of one SSE chunk."""
        text = ""
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            if api_type == "chat":
                text = (choice.get("delta") or {}).get("content") or ""
            else:
                text = choice.get("text") or ""
        return text, chunk.get("usage")

    async def send_request(
        self,
        request_id: int,
        user_id: int,
        prompt_index: int,
        prompt: str,
    ) -> RequestMetrics:
        """Send one streaming completion request and return its recorded metrics.

        Never raises: any network, HTTP, or parsing error is caught and
        recorded on the returned `RequestMetrics` as a failure, so a single
        bad request can't take down a concurrent benchmark run.
        """
        metrics = RequestMetrics(
            request_id=request_id,
            user_id=user_id,
            prompt_index=prompt_index,
            prompt_words=len(prompt.split()),
        )

        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout)
        metrics.start_time = time.perf_counter()
        last_token_time = metrics.start_time

        try:
            async with self._session.post(
                self._endpoint(),
                json=self._payload(prompt),
                headers=self._headers(),
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed SSE chunk: %s", data[:200])
                        continue

                    text, usage = self._extract_text_and_usage(chunk, self._config.api_type)
                    now = time.perf_counter()

                    if text:
                        if metrics.ttft is None:
                            metrics.ttft = now - metrics.start_time
                        else:
                            metrics.inter_token_gaps.append(now - last_token_time)
                        last_token_time = now
                        metrics.tokens_generated += 1

                    if usage and usage.get("completion_tokens") is not None:
                        metrics.tokens_generated = usage["completion_tokens"]

            metrics.success = True

        except Exception as exc:  # noqa: BLE001 - any failure becomes a recorded result, not a crash
            metrics.success = False
            metrics.error = f"{type(exc).__name__}: {exc}"
            logger.debug("Request %d failed: %s", request_id, metrics.error)

        finally:
            metrics.end_time = time.perf_counter()

        return metrics
