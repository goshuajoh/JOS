"""Backend factory. Heavy imports (torch, gliner2) happen only when that backend is chosen."""
from __future__ import annotations

from .base import Backend, Probs, softmax


def make_backend(name: str, model: str | None = None, adapter: str | None = None,
                 url: str | None = None, device: str | None = None, **kw) -> Backend:
    if name == "keyword":
        from .keyword import KeywordBackend
        return KeywordBackend()
    if name == "gliner":
        from .gliner import DEFAULT_MODEL, GlinerBackend
        return GlinerBackend(model or DEFAULT_MODEL, device=device, **kw)
    if name == "qwen":
        from .qwen_logit import DEFAULT_MODEL, QwenLogitBackend
        return QwenLogitBackend(model or DEFAULT_MODEL, adapter=adapter, device=device, **kw)
    if name == "endpoint":
        from .endpoint import EndpointBackend
        if not url:
            raise ValueError("--url is required for the endpoint backend")
        return EndpointBackend(url)
    if name == "frontier":
        from ..synth import make_llm
        from .frontier import FrontierBackend
        llm = make_llm(kw.pop("provider", "anthropic"), model, kw.pop("base_url", None) or url)
        return FrontierBackend(llm, guide_path=kw.pop("guide", None))
    raise ValueError(f"unknown backend {name!r} (keyword, gliner, qwen, endpoint, frontier)")


__all__ = ["Backend", "Probs", "softmax", "make_backend"]
