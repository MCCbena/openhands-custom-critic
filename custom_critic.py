"""Custom Critic for OpenHands that works with any OpenAI-compatible LLM (DeepSeek, etc.).

Usage in settings.json:
  "verification": {
    "critic_enabled": true,
    "critic_mode": "finish_and_message",
    "enable_iterative_refinement": false,
    "critic_threshold": 0.6,
    "max_refinement_iterations": 3,
    "critic_server_url": "https://api.deepseek.com/v1",
    "critic_model_name": "deepseek-chat",
    "critic_api_key": "sk-4ad9afa61e7749e1b079c8b693824257"
  }

This file must be importable by OpenHands. Place it in a known path and reference
it from agent_context.skills or PYTHONPATH.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import Field, SecretStr

from openhands.sdk.critic.base import CriticBase, CriticResult

if TYPE_CHECKING:
    from openhands.sdk.event import LLMConvertibleEvent


SYSTEM_PROMPT = """You are an expert evaluation judge for AI coding agents.
Your job is to evaluate whether the agent's recent actions successfully completed the user's task.

Evaluate the following conversation and return a JSON object with:
- "success": boolean - Whether the task appears to be completed correctly
- "score": float between 0.0 and 1.0 - Your confidence in success (0=definitely failed, 1=definitely succeeded)
- "rationale": string - Brief explanation of your assessment

Consider:
1. Did the agent address all requirements from the user's request?
2. Are there any obvious errors or incomplete work?
3. Does the final state look correct (files created/modified, tests passing)?
4. Was the task scope properly understood and executed?

Be fair but thorough. If the agent made reasonable progress but didn't fully finish, score accordingly (e.g., 0.6-0.7)."""


class DeepSeekCritic(CriticBase):
    """A critic that uses any OpenAI-compatible LLM to evaluate agent actions."""

    server_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="Base URL of the LLM API (OpenAI-compatible).",
    )
    model_name: str = Field(
        default="deepseek-chat",
        description="Model name to use for evaluation.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key. If None, reads from CRITIC_API_KEY env var.",
    )

    def _get_api_key(self) -> str:
        if self.api_key is not None:
            return (
                self.api_key.get_secret_value()
                if isinstance(self.api_key, SecretStr)
                else str(self.api_key)
            )
        key = os.environ.get("CRITIC_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set api_key in config or CRITIC_API_KEY env var."
            )
        return key

    def _build_prompt(self, events: Sequence[LLMConvertibleEvent]) -> list[dict[str, Any]]:
        """Convert OpenHands events into a conversation for the LLM."""
        messages: list[dict[str, Any]] = []

        from openhands.sdk.event import SystemPromptEvent

        system_prompt_event: SystemPromptEvent | None = None
        tools_defs: list[Any] | None = None

        for event in events:
            if isinstance(event, SystemPromptEvent):
                system_prompt_event = event
                tools_defs = event.tools
                continue

            try:
                msg = event.to_llm_message()
                role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                content_parts = []

                if hasattr(msg, 'content') and msg.content:
                    for c in msg.content:
                        from openhands.sdk.utils.text_content import TextContent
                        if isinstance(c, TextContent):
                            content_parts.append(c.text)

                # Add tool call info if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        func = getattr(tc, 'function', None)
                        if func:
                            args_str = json.dumps(getattr(func, 'arguments', {}))
                            content_parts.append(
                                f"\n[Tool Call] {getattr(func, 'name', '?')}({args_str})"
                            )

                text = "\n".join(content_parts).strip() or None

                # Add reasoning if present
                if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                    reasoning = getattr(msg, 'reasoning_content', '')
                    if reasoning:
                        if text:
                            text = f"[Reasoning] {reasoning}\n\n{text}"
                        else:
                            text = f"[Reasoning] {reasoning}"

                if text:
                    messages.append({"role": role.lower(), "content": text})
            except Exception:
                continue

        return messages

    def _evaluate(self, conversation: str) -> CriticResult:
        """Call the LLM to evaluate the conversation."""
        api_key = self._get_api_key()
        url = f"{self.server_url.rstrip('/')}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": conversation},
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]

        # Parse JSON from response (the LLM may wrap it in markdown code blocks)
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
            else:
                return CriticResult(
                    score=0.5,
                    message=f"Failed to parse LLM response: {content[:200]}",
                )

        score = float(result.get("score", 0.5))
        success = bool(result.get("success", False))
        rationale = result.get("rationale", "")

        # Clamp score
        score = max(0.0, min(1.0, score))

        return CriticResult(
            score=score,
            message=f"{'✓ Success' if success else '✗ Incomplete'}: {rationale}",
            metadata={"llm_response": content},
        )

    def evaluate(
        self,
        events: Sequence[LLMConvertibleEvent],
        git_patch: str | None = None,  # noqa: ARG002
    ) -> CriticResult:
        messages = self._build_prompt(events)
        conversation = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages if m.get("content"))

        if not conversation.strip():
            return CriticResult(score=0.5, message="No conversation content to evaluate.")

        return self._evaluate(conversation)
