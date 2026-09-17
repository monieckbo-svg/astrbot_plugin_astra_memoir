"""Resolve the one configured embedding provider and verify its dimension."""

import inspect


def _provider_meta(provider):
    meta_fn = getattr(provider, "meta", None)
    if callable(meta_fn):
        try:
            return meta_fn()
        except Exception:
            return None
    return None


def _provider_id(provider) -> str:
    direct = getattr(provider, "id", None)
    if direct:
        return str(direct)
    meta = _provider_meta(provider)
    mid = getattr(meta, "id", None)
    return str(mid) if mid else ""


def provider_display_name(provider, fallback_id: str) -> str:
    meta = _provider_meta(provider)
    for value in (
        getattr(meta, "display_name", None),
        getattr(meta, "name", None),
        getattr(meta, "model", None),
        getattr(provider, "name", None),
        getattr(provider, "model", None),
    ):
        if value:
            return str(value)
    config = getattr(provider, "provider_config", None)
    if isinstance(config, dict):
        for key in ("model", "model_name", "name"):
            if config.get(key):
                return str(config[key])
    return fallback_id


def resolve_embedding_provider(context, provider_id: str):
    provider_id = str(provider_id or "").strip()
    if not provider_id:
        raise RuntimeError("Memoir 必须选择 Embedding provider")
    for provider in context.get_all_embedding_providers():
        if _provider_id(provider) == provider_id:
            return provider
    raise RuntimeError(f"Memoir 配置的 Embedding provider 不存在: {provider_id}")


async def embedding_dimension(provider) -> int:
    get_dim = getattr(provider, "get_dim", None)
    dim = get_dim() if callable(get_dim) else None
    if inspect.isawaitable(dim):
        dim = await dim
    if not dim:
        probe = await provider.get_embedding("memoir dimension probe")
        dim = len(probe) if probe is not None else 0
    dim = int(dim)
    if dim <= 0:
        raise RuntimeError("Embedding provider 返回了无效维度")
    return dim
