import json
from typing import Any, Optional

from openai import OpenAI

from app.core.config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.openai_api_key)
        self._client: Optional[OpenAI] = None
        if self.enabled:
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        if not self.enabled or self._client is None:
            raise RuntimeError("LLM is not configured. Set OPENAI_API_KEY.")

        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

        choice = resp.choices[0]
        msg = choice.message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": args,
                    }
                )
        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason,
        }

    def complete_json(self, system: str, user: str) -> dict:
        raw = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )["content"]
        return _extract_json(raw)


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {"answer": text, "parse_error": True}
    return {"answer": text, "parse_error": True}
