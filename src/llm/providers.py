"""
Pluggable LLM providers for the verified pipeline.

A provider turns (system prompt, user prompt, JSON schema) into a JSON
string. Two implementations cover most deployments:

    ollama:<model>   local Ollama via its native API (default: llama3.2)
    openai:<model>   any OpenAI-compatible endpoint — OpenAI itself, or
                     vLLM, Together, Groq, or Ollama's own /v1 endpoint.
                     Configure with OPENAI_BASE_URL and OPENAI_API_KEY.

Select one with get_provider("ollama:llama3.2") or the LLM_PROVIDER
environment variable.
"""

import os


DEFAULT_PROVIDER = "ollama:llama3.2"

# Deterministic decoding so evaluation runs are repeatable.
TEMPERATURE = 0.0
SEED = 7


class OllamaProvider:
    def __init__(self, model):
        self.model = model
        self.name = f"ollama:{model}"

    def complete(self, system, user, json_schema=None):
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=json_schema,
            options={"temperature": TEMPERATURE, "seed": SEED},
        )
        return response["message"]["content"]


class OpenAICompatibleProvider:
    def __init__(self, model, base_url=None, api_key=None):
        self.model = model
        self.name = f"openai:{model}"
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")

    def complete(self, system, user, json_schema=None):
        import openai

        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        kwargs = {}
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            seed=SEED,
            **kwargs,
        )
        return response.choices[0].message.content or ""


_PROVIDERS = {
    "ollama": OllamaProvider,
    "openai": OpenAICompatibleProvider,
}


def get_provider(spec=None):
    """Build a provider from 'kind:model', e.g. 'ollama:llama3.2'."""
    spec = spec or os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    kind, sep, model = spec.partition(":")
    if not sep or kind not in _PROVIDERS or not model:
        raise ValueError(
            f"Unknown provider spec {spec!r}; expected one of "
            f"{', '.join(k + ':<model>' for k in _PROVIDERS)}"
        )
    return _PROVIDERS[kind](model)
