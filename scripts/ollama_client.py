"""Thin wrapper around the local Ollama HTTP API.

Used for both the attacker (red-teamer LLM) and the judge, so they must be
DIFFERENT models to avoid the judge rubber-stamping its own attacker's outputs.
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"


def chat(model: str, messages: list[dict], temperature: float = 1.0, timeout: int = 300) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
