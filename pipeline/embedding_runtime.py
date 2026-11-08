"""A shared, recoverable provider reference; never gates raw ingestion."""
import asyncio
import time
from .embedding import resolve_embedding_provider, embedding_dimension, provider_display_name


class EmbeddingRuntime:
    def __init__(self, context, provider_id, db, health, *, max_input_chars=450):
        self.context, self.id, self.db, self.health = context, provider_id, db, health
        self.provider = None
        self.name = provider_id
        self.max_input_chars = max(128, min(int(max_input_chars), 4000))
        self._lock = asyncio.Lock()

    def _prepare_text(self, text):
        """Bound provider input while keeping the complete episode in SQLite/FTS."""
        clean = "".join(
            ch if ch in "\n\t" or ord(ch) >= 32 else " " for ch in str(text or "")
        ).strip()
        if len(clean) <= self.max_input_chars:
            return clean
        marker = "\n…[向量输入已截断]…\n"
        available = self.max_input_chars - len(marker)
        head = max(1, available * 3 // 4)
        tail = max(1, available - head)
        prepared = clean[:head] + marker + clean[-tail:]
        self.health.update(
            last_embedding_truncated_at=int(time.time()),
            last_embedding_original_chars=len(clean),
            embedding_truncated_count=int(
                self.health.data.get('embedding_truncated_count', 0)) + 1,
        )
        return prepared

    async def recover(self):
        async with self._lock:
            try:
                provider = resolve_embedding_provider(self.context, self.id)
                async def probe():
                    dim = await embedding_dimension(provider)
                    vector = await provider.get_embedding('memoir recovery probe')
                    if len(vector) != dim:
                        raise ValueError('Embedding probe 维度与get_dim不一致')
                    return dim
                dim = await asyncio.wait_for(probe(), 20)
                await self.db.run(self.db.configure_vectors, dim, self.id)
                self.provider = provider
                self.name = provider_display_name(provider, self.id)
                self.health.update(embedding_status='ok', embedding_error=None)
                return True
            except Exception as exc:
                self.provider = None
                self.health.error('embedding', exc)
                self.health.update(embedding_status='degraded')
                return False

    async def get_embedding(self, text):
        async with self._lock:
            try:
                # Provider objects may be replaced without a plugin reload.
                current = resolve_embedding_provider(self.context, self.id)
                if self.provider is None or current is not self.provider:
                    raise RuntimeError('Embedding 暂未就绪或provider已更换，等待自动恢复')
                vector = await asyncio.wait_for(
                    current.get_embedding(self._prepare_text(text)), 20)
                if len(vector) != self.db.embedding_dim:
                    raise ValueError('Embedding维度变化，等待自动恢复')
                return vector
            except Exception as exc:
                self.provider = None
                self.health.error('embedding', exc)
                self.health.update(embedding_status='degraded')
                raise
