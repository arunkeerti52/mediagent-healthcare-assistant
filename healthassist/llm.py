"""LLM and embedding clients: Cloudflare Workers AI through its OpenAI-compatible API.

Configure in .env:
    CLOUDFLARE_ACCOUNT_ID=...
    CLOUDFLARE_API_TOKEN=...
Optional overrides: CLOUDFLARE_MODEL, CLOUDFLARE_EMBED_MODEL
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
DEFAULT_EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"


class LLMConfigError(RuntimeError):
    """Raised when Cloudflare credentials are missing."""


def _credentials() -> tuple[str, str]:
    account, token = os.getenv("CLOUDFLARE_ACCOUNT_ID"), os.getenv("CLOUDFLARE_API_TOKEN")
    if not account or not token:
        raise LLMConfigError("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the .env file.")
    return f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1", token


def model_name() -> str:
    return os.getenv("CLOUDFLARE_MODEL", DEFAULT_MODEL)


def provider_name() -> str:
    _credentials()
    return f"Cloudflare Workers AI · {model_name()}"


def get_llm(temperature: float = 0.1, max_tokens: int = 1200) -> ChatOpenAI:
    base_url, token = _credentials()
    return ChatOpenAI(model=model_name(), base_url=base_url, api_key=token, temperature=temperature,
                      max_tokens=max_tokens, max_retries=3, timeout=90)


def get_embeddings() -> OpenAIEmbeddings:
    base_url, token = _credentials()
    return OpenAIEmbeddings(model=os.getenv("CLOUDFLARE_EMBED_MODEL", DEFAULT_EMBED_MODEL), base_url=base_url,
                            api_key=token, check_embedding_ctx_length=False, chunk_size=50)
