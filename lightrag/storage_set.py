"""StorageSet: authoritative collection of all LightRAG storage backends.

Owns collective lifecycle (initialize, finalize_all, flush_all) so callers
address the 12 storage instances as a unit rather than individually.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

from lightrag.base import StorageNameSpace
from lightrag.utils import logger


@dataclass
class StorageSet:
    """All 12 storage backends with collective lifecycle methods."""

    full_docs: Any
    doc_status: Any
    text_chunks: Any
    chunks_vdb: Any
    entities_vdb: Any
    relationships_vdb: Any
    full_entities: Any
    full_relations: Any
    entity_chunks: Any
    relation_chunks: Any
    chunk_entity_relation_graph: Any
    llm_response_cache: Any

    async def initialize(self) -> None:
        """Initialize all storages sequentially (parallel init causes deadlocks)."""
        for s in self._all():
            if s:
                await s.initialize()

    async def flush_all(self) -> None:
        """Persist all in-memory state to backing store."""
        tasks = [
            cast(StorageNameSpace, s).index_done_callback()
            for s in self._all()
            if s is not None
        ]
        await asyncio.gather(*tasks)
        logger.info("In memory DB persist to disk")

    async def finalize_all(self) -> tuple[list[str], list[str]]:
        """Finalize all storages, tolerating individual failures.

        Returns (succeeded_names, failed_names).
        """
        succeeded: list[str] = []
        failed: list[str] = []
        for name, s in self._named():
            if s:
                try:
                    await s.finalize()
                    succeeded.append(name)
                    logger.debug(f"Successfully finalized {name}")
                except Exception as e:
                    logger.error(f"Failed to finalize {name}: {e}")
                    failed.append(name)
        return succeeded, failed

    def _all(self) -> list[Any]:
        return [
            self.full_docs,
            self.doc_status,
            self.text_chunks,
            self.chunks_vdb,
            self.entities_vdb,
            self.relationships_vdb,
            self.full_entities,
            self.full_relations,
            self.entity_chunks,
            self.relation_chunks,
            self.chunk_entity_relation_graph,
            self.llm_response_cache,
        ]

    def _named(self) -> list[tuple[str, Any]]:
        return [
            ("full_docs", self.full_docs),
            ("doc_status", self.doc_status),
            ("text_chunks", self.text_chunks),
            ("chunks_vdb", self.chunks_vdb),
            ("entities_vdb", self.entities_vdb),
            ("relationships_vdb", self.relationships_vdb),
            ("full_entities", self.full_entities),
            ("full_relations", self.full_relations),
            ("entity_chunks", self.entity_chunks),
            ("relation_chunks", self.relation_chunks),
            ("chunk_entity_relation_graph", self.chunk_entity_relation_graph),
            ("llm_response_cache", self.llm_response_cache),
        ]
