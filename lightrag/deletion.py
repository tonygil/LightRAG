"""Document deletion pipeline extracted from LightRAG.

Owns the four-stage deletion flow: validate → delete chunks/graph → rebuild → finalize.
LightRAG delegates to delete_document() and remains a thin orchestrator.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from lightrag.base import DocStatus, DeletionResult
from lightrag.config import PipelineConfig
from lightrag.storage_set import StorageSet as StorageBundle
from lightrag.utils import (
    CancellationToken,
    logger,
)
from lightrag.text_utils import (
    compute_mdhash_id,
    make_relation_chunk_key,
    subtract_source_ids,
)
from lightrag.constants import GRAPH_FIELD_SEP


def _normalize_string_list(raw_values: Any, context: str = "") -> list[str]:
    """Return a list of non-empty strings from raw_values.

    Non-string elements are dropped and logged as warnings. If raw_values is
    not a list, an empty list is returned.
    """
    if not isinstance(raw_values, list):
        return []
    result = []
    for i, value in enumerate(raw_values):
        if isinstance(value, str) and value:
            result.append(value)
        else:
            logger.warning(
                "Non-string element dropped from list%s at index %d: %r",
                f" ({context})" if context else "",
                i,
                value,
            )
    return result


async def _update_delete_retry_state(
    doc_id: str,
    doc_status_data: dict[str, Any],
    doc_status_storage: Any,
    *,
    deletion_stage: str,
    doc_llm_cache_ids: list[str],
    error_message: str | None = None,
    failed: bool,
) -> dict[str, Any]:
    """Persist deletion retry metadata and return the updated status record."""
    metadata = doc_status_data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    backup_cache_ids = _normalize_string_list(
        metadata.get("deletion_llm_cache_ids", []),
        context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
    )
    retry_cache_ids = doc_llm_cache_ids or backup_cache_ids

    updated_metadata = dict(metadata)
    if retry_cache_ids:
        updated_metadata["deletion_llm_cache_ids"] = retry_cache_ids
    updated_metadata["last_deletion_attempt_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    if failed:
        updated_metadata["deletion_failed"] = True
        updated_metadata["deletion_failure_stage"] = deletion_stage
    else:
        updated_metadata.pop("deletion_failed", None)
        updated_metadata.pop("deletion_failure_stage", None)

    updated_status_data = {
        **doc_status_data,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": updated_metadata,
        "error_msg": error_message if failed else "",
    }

    await doc_status_storage.upsert({doc_id: updated_status_data})
    return updated_status_data


async def _get_existing_llm_cache_ids(
    cache_ids: list[str],
    llm_response_cache: Any,
) -> list[str]:
    """Return cache IDs that still exist in cache storage.

    Some KV storage backends only log delete failures and return without
    raising, so callers must verify which records still exist after delete.

    Returns an empty list immediately if cache storage is unavailable.
    Callers must check storage availability independently before treating
    an empty result as a confirmed deletion.
    """
    if not llm_response_cache or not cache_ids:
        return []

    try:
        existing_records = await llm_response_cache.get_by_ids(cache_ids)
    except Exception as verification_error:
        raise Exception(
            f"Failed to verify LLM cache deletion "
            f"(delete may have succeeded): {verification_error}"
        ) from verification_error
    return [
        cache_id
        for cache_id, record in zip(cache_ids, existing_records)
        if record is not None
    ]


async def delete_document(
    doc_id: str,
    delete_llm_cache: bool,
    storages: StorageBundle,
    config: PipelineConfig,
    pipeline_status: dict,
    pipeline_status_lock: Any,
    we_acquired_pipeline: bool,
    rebuild_fn: Callable,
    insert_done_fn: Callable,
) -> DeletionResult:
    """Execute the full document deletion pipeline.

    Stages: validate → collect LLM cache IDs → analyze graph dependencies →
    delete chunks → delete relationships → delete entities → persist → rebuild →
    delete LLM cache → delete doc entries.

    ``rebuild_fn`` is the caller's reference to rebuild_knowledge_from_chunks so
    tests can monkey-patch it via the lightrag.lightrag module namespace.
    ``insert_done_fn`` is similarly passed so tests can patch rag._insert_done.
    """
    token = CancellationToken(pipeline_status, pipeline_status_lock)

    deletion_operations_started = False
    deletion_fully_completed = False
    in_final_delete_stage = False
    original_exception = None
    doc_llm_cache_ids: list[str] = []
    deletion_stage = "initializing"
    doc_status_data: dict[str, Any] | None = None
    file_path: str | None = None

    async with pipeline_status_lock:
        log_message = f"Starting deletion process for document {doc_id}"
        logger.info(log_message)
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    try:
        # 1. Get the document status and related data
        doc_status_data = await storages.doc_status.get_by_id(doc_id)
        file_path = doc_status_data.get("file_path") if doc_status_data else None
        if not doc_status_data:
            logger.warning(f"Document {doc_id} not found")
            return DeletionResult(
                status="not_found",
                doc_id=doc_id,
                message=f"Document {doc_id} not found.",
                status_code=404,
                file_path="",
            )

        # Check document status and log warning for non-completed documents
        raw_status = doc_status_data.get("status")
        try:
            doc_status = DocStatus(raw_status)
        except ValueError:
            doc_status = raw_status

        if doc_status != DocStatus.PROCESSED:
            if doc_status == DocStatus.PENDING:
                warning_msg = (
                    f"Deleting {doc_id} {file_path}(previous status: PENDING)"
                )
            elif doc_status == DocStatus.PROCESSING:
                warning_msg = (
                    f"Deleting {doc_id} {file_path}(previous status: PROCESSING)"
                )
            elif doc_status == DocStatus.PREPROCESSED:
                warning_msg = (
                    f"Deleting {doc_id} {file_path}(previous status: PREPROCESSED)"
                )
            elif doc_status == DocStatus.FAILED:
                warning_msg = (
                    f"Deleting {doc_id} {file_path}(previous status: FAILED)"
                )
            else:
                status_text = (
                    doc_status.value
                    if isinstance(doc_status, DocStatus)
                    else str(doc_status)
                )
                warning_msg = (
                    f"Deleting {doc_id} {file_path}(previous status: {status_text})"
                )
            logger.info(warning_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = warning_msg
                pipeline_status["history_messages"].append(warning_msg)

        # 2. Get chunk IDs from document status
        metadata = doc_status_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata_cache_ids = _normalize_string_list(
            metadata.get("deletion_llm_cache_ids", []),
            context=f"doc {doc_id} metadata.deletion_llm_cache_ids",
        )
        chunk_ids = set(
            _normalize_string_list(
                doc_status_data.get("chunks_list", []),
                context=f"doc {doc_id} chunks_list",
            )
        )

        if not chunk_ids:
            logger.warning(f"No chunks found for document {doc_id}")
            # Mark that deletion operations have started
            deletion_operations_started = True

            # A prior failed deletion may have collected LLM cache IDs before the
            # chunks were removed. If delete_llm_cache is requested and persisted IDs
            # exist, clean them up now before removing the doc/status entries.
            if delete_llm_cache and metadata_cache_ids:
                if not storages.llm_response_cache:
                    no_cache_msg = (
                        f"Cannot delete LLM cache for document {doc_id}: "
                        "cache storage is unavailable"
                    )
                    logger.error(no_cache_msg)
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = no_cache_msg
                        pipeline_status["history_messages"].append(no_cache_msg)
                    raise Exception(no_cache_msg)
                try:
                    deletion_stage = "delete_llm_cache"
                    await storages.llm_response_cache.delete(metadata_cache_ids)
                    remaining_cache_ids = await _get_existing_llm_cache_ids(
                        metadata_cache_ids, storages.llm_response_cache
                    )
                    if remaining_cache_ids:
                        raise Exception(
                            f"{len(remaining_cache_ids)} LLM cache entries still exist after delete"
                        )
                    logger.info(
                        "Cleaned up %d LLM cache entries from prior attempt for document %s",
                        len(metadata_cache_ids),
                        doc_id,
                    )
                except Exception as cache_err:
                    raise Exception(
                        f"Failed to delete LLM cache for document {doc_id}: {cache_err}"
                    ) from cache_err

            try:
                # Still need to delete the doc status and full doc.
                # Delete doc_status first: if full_docs.delete fails on retry, the
                # doc_status record is already gone so the retry finds no record and
                # treats the document as already deleted rather than creating a zombie.
                deletion_stage = "delete_doc_entries"
                await storages.doc_status.delete([doc_id])
                await storages.full_docs.delete([doc_id])
            except Exception as e:
                logger.error(
                    f"Failed to delete document {doc_id} with no chunks: {e}"
                )
                raise Exception(f"Failed to delete document entry: {e}") from e

            async with pipeline_status_lock:
                log_message = (
                    f"Document deleted without associated chunks: {doc_id}"
                )
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

            deletion_fully_completed = True
            return DeletionResult(
                status="success",
                doc_id=doc_id,
                message=log_message,
                status_code=200,
                file_path=file_path,
            )

        # Mark that deletion operations have started
        deletion_operations_started = True

        if chunk_ids:
            # Always collect/persist cache IDs for chunk-backed documents, even when
            # this call does not request cache deletion. If a delete fails after the
            # chunks/graph have already been removed, a later retry may turn on
            # delete_llm_cache=True, and doc_status metadata is then the only durable
            # place left to recover the cache keys for cleanup.
            deletion_stage = "collect_llm_cache"
            doc_llm_cache_ids = list(metadata_cache_ids)
            if not storages.text_chunks:
                logger.info(
                    "Skipping LLM cache id collection for document %s because text chunk storage is unavailable",
                    doc_id,
                )
            else:
                try:
                    chunk_data_list = await storages.text_chunks.get_by_ids(
                        list(chunk_ids)
                    )
                    seen_cache_ids: set[str] = set(doc_llm_cache_ids)
                    for chunk_data in chunk_data_list:
                        if not chunk_data or not isinstance(chunk_data, dict):
                            continue
                        cache_ids = chunk_data.get("llm_cache_list", [])
                        if not isinstance(cache_ids, list):
                            continue
                        for cache_id in cache_ids:
                            if (
                                isinstance(cache_id, str)
                                and cache_id
                                and cache_id not in seen_cache_ids
                            ):
                                doc_llm_cache_ids.append(cache_id)
                                seen_cache_ids.add(cache_id)
                except Exception as cache_collect_error:
                    logger.error(
                        "Failed to collect LLM cache ids for document %s: %s",
                        doc_id,
                        cache_collect_error,
                    )
                    raise Exception(
                        f"Failed to collect LLM cache ids for document {doc_id}: {cache_collect_error}"
                    ) from cache_collect_error

            if doc_llm_cache_ids:
                try:
                    doc_status_data = await _update_delete_retry_state(
                        doc_id,
                        doc_status_data,
                        storages.doc_status,
                        deletion_stage=deletion_stage,
                        doc_llm_cache_ids=doc_llm_cache_ids,
                        failed=False,
                    )
                except Exception as status_write_error:
                    logger.error(
                        "Failed to persist LLM cache IDs for document %s to retry state: %s",
                        doc_id,
                        status_write_error,
                    )
                    attempt_context = (
                        "retry — prior partial deletions may exist"
                        if metadata_cache_ids
                        else "deletion not yet started"
                    )
                    raise Exception(
                        f"Failed to persist LLM cache IDs for document {doc_id} "
                        f"({attempt_context}): {status_write_error}"
                    ) from status_write_error
                logger.info(
                    "Collected %d LLM cache entries for document %s",
                    len(doc_llm_cache_ids),
                    doc_id,
                )
            else:
                logger.info("No LLM cache entries found for document %s", doc_id)

        # 4. Analyze entities and relationships that will be affected
        entities_to_delete = set()
        entities_to_rebuild = {}  # entity_name -> remaining chunk id list
        relationships_to_delete = set()
        relationships_to_rebuild = {}  # (src, tgt) -> remaining chunk id list
        entity_chunk_updates: dict[str, list[str]] = {}
        relation_chunk_updates: dict[tuple[str, str], list[str]] = {}

        try:
            deletion_stage = "analyze_graph_dependencies"
            # Get affected entities and relations from full_entities and full_relations storage
            doc_entities_data = await storages.full_entities.get_by_id(doc_id)
            doc_relations_data = await storages.full_relations.get_by_id(doc_id)

            affected_nodes = []
            affected_edges = []

            # Get entity data from graph storage using entity names from full_entities
            if doc_entities_data and "entity_names" in doc_entities_data:
                entity_names = doc_entities_data["entity_names"]
                nodes_dict = await storages.chunk_entity_relation_graph.get_nodes_batch(
                    entity_names
                )
                for entity_name in entity_names:
                    node_data = nodes_dict.get(entity_name)
                    if node_data:
                        if "id" not in node_data:
                            node_data["id"] = entity_name
                        affected_nodes.append(node_data)

            # Get relation data from graph storage using relation pairs from full_relations
            if doc_relations_data and "relation_pairs" in doc_relations_data:
                relation_pairs = doc_relations_data["relation_pairs"]
                edge_pairs_dicts = [
                    {"src": pair[0], "tgt": pair[1]} for pair in relation_pairs
                ]
                edges_dict = await storages.chunk_entity_relation_graph.get_edges_batch(
                    edge_pairs_dicts
                )

                for pair in relation_pairs:
                    src, tgt = pair[0], pair[1]
                    edge_key = (src, tgt)
                    edge_data = edges_dict.get(edge_key)
                    if edge_data:
                        if "source" not in edge_data:
                            edge_data["source"] = src
                        if "target" not in edge_data:
                            edge_data["target"] = tgt
                        affected_edges.append(edge_data)

        except Exception as e:
            logger.error(f"Failed to analyze affected graph elements: {e}")
            raise Exception(f"Failed to analyze graph dependencies: {e}") from e

        try:
            # Process entities
            for node_data in affected_nodes:
                node_label = node_data.get("entity_id")
                if not node_label:
                    continue

                existing_sources: list[str] = []
                graph_sources: list[str] = []
                if storages.entity_chunks:
                    stored_chunks = await storages.entity_chunks.get_by_id(node_label)
                    if stored_chunks and isinstance(stored_chunks, dict):
                        existing_sources = [
                            chunk_id
                            for chunk_id in stored_chunks.get("chunk_ids", [])
                            if chunk_id
                        ]

                if node_data.get("source_id"):
                    graph_sources = [
                        chunk_id
                        for chunk_id in node_data["source_id"].split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]

                if not existing_sources:
                    existing_sources = graph_sources

                if not existing_sources:
                    entities_to_delete.add(node_label)
                    entity_chunk_updates[node_label] = []
                    continue

                remaining_sources = subtract_source_ids(existing_sources, chunk_ids)
                # `existing_sources` comes from chunk-tracking storage when available, but
                # graph `source_id` can still be stale after a failed prior delete. If the
                # graph still references any chunk being deleted in this attempt, force a
                # rebuild/delete so the graph metadata gets synchronized instead of being
                # left untouched with orphaned source references.
                graph_references_deleted_chunks = bool(
                    graph_sources and set(graph_sources) & chunk_ids
                )

                if not remaining_sources:
                    entities_to_delete.add(node_label)
                    entity_chunk_updates[node_label] = []
                elif (
                    remaining_sources != existing_sources
                    or graph_references_deleted_chunks
                ):
                    entities_to_rebuild[node_label] = remaining_sources
                    entity_chunk_updates[node_label] = remaining_sources
                else:
                    logger.info(f"Untouch entity: {node_label}")

            async with pipeline_status_lock:
                log_message = f"Found {len(entities_to_rebuild)} affected entities"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

            # Process relationships
            for edge_data in affected_edges:
                src = edge_data.get("source")
                tgt = edge_data.get("target")

                if not src or not tgt or "source_id" not in edge_data:
                    continue

                edge_tuple = tuple(sorted((src, tgt)))
                if (
                    edge_tuple in relationships_to_delete
                    or edge_tuple in relationships_to_rebuild
                ):
                    continue

                existing_sources = []
                graph_sources = []
                if storages.relation_chunks:
                    storage_key = make_relation_chunk_key(src, tgt)
                    stored_chunks = await storages.relation_chunks.get_by_id(
                        storage_key
                    )
                    if stored_chunks and isinstance(stored_chunks, dict):
                        existing_sources = [
                            chunk_id
                            for chunk_id in stored_chunks.get("chunk_ids", [])
                            if chunk_id
                        ]

                if edge_data.get("source_id"):
                    graph_sources = [
                        chunk_id
                        for chunk_id in edge_data["source_id"].split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]

                if not existing_sources:
                    existing_sources = graph_sources

                if not existing_sources:
                    relationships_to_delete.add(edge_tuple)
                    relation_chunk_updates[edge_tuple] = []
                    continue

                remaining_sources = subtract_source_ids(existing_sources, chunk_ids)
                # Same as the entity path above: even when relation chunk-tracking is already
                # correct, the graph edge may still carry a stale `source_id` that mentions a
                # chunk deleted in this attempt. Treat that as an affected relation so retry
                # deletion can repair the graph metadata rather than skipping it as "untouched".
                graph_references_deleted_chunks = bool(
                    graph_sources and set(graph_sources) & chunk_ids
                )

                if not remaining_sources:
                    relationships_to_delete.add(edge_tuple)
                    relation_chunk_updates[edge_tuple] = []
                elif (
                    remaining_sources != existing_sources
                    or graph_references_deleted_chunks
                ):
                    relationships_to_rebuild[edge_tuple] = remaining_sources
                    relation_chunk_updates[edge_tuple] = remaining_sources
                else:
                    logger.info(f"Untouch relation: {edge_tuple}")

            async with pipeline_status_lock:
                log_message = (
                    f"Found {len(relationships_to_rebuild)} affected relations"
                )
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

            import time
            current_time = int(time.time())
            deletion_stage = "update_chunk_tracking"

            if entity_chunk_updates and storages.entity_chunks:
                entity_upsert_payload = {}
                for entity_name, remaining in entity_chunk_updates.items():
                    if not remaining:
                        continue
                    entity_upsert_payload[entity_name] = {
                        "chunk_ids": remaining,
                        "count": len(remaining),
                        "updated_at": current_time,
                    }
                if entity_upsert_payload:
                    await storages.entity_chunks.upsert(entity_upsert_payload)

            if relation_chunk_updates and storages.relation_chunks:
                relation_upsert_payload = {}
                for edge_tuple, remaining in relation_chunk_updates.items():
                    if not remaining:
                        continue
                    storage_key = make_relation_chunk_key(*edge_tuple)
                    relation_upsert_payload[storage_key] = {
                        "chunk_ids": remaining,
                        "count": len(remaining),
                        "updated_at": current_time,
                    }

                if relation_upsert_payload:
                    await storages.relation_chunks.upsert(relation_upsert_payload)

        except Exception as e:
            logger.error(f"Failed to process graph analysis results: {e}")
            raise Exception(f"Failed to process graph dependencies: {e}") from e

        # Data integrity is ensured by allowing only one process to hold pipeline at a time

        # 5. Delete chunks from storage
        if chunk_ids:
            try:
                deletion_stage = "delete_chunks"
                await storages.chunks_vdb.delete(chunk_ids)
                await storages.text_chunks.delete(chunk_ids)

                async with pipeline_status_lock:
                    log_message = (
                        f"Successfully deleted {len(chunk_ids)} chunks from storage"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

            except Exception as e:
                logger.error(f"Failed to delete chunks: {e}")
                raise Exception(f"Failed to delete document chunks: {e}") from e

        # 6. Delete relationships that have no remaining sources
        if relationships_to_delete:
            try:
                deletion_stage = "delete_relationships"
                rel_ids_to_delete = []
                for src, tgt in relationships_to_delete:
                    rel_ids_to_delete.extend(
                        [
                            compute_mdhash_id(src + tgt, prefix="rel-"),
                            compute_mdhash_id(tgt + src, prefix="rel-"),
                        ]
                    )
                await storages.relationships_vdb.delete(rel_ids_to_delete)

                await storages.chunk_entity_relation_graph.remove_edges(
                    list(relationships_to_delete)
                )

                if storages.relation_chunks:
                    relation_storage_keys = [
                        make_relation_chunk_key(src, tgt)
                        for src, tgt in relationships_to_delete
                    ]
                    await storages.relation_chunks.delete(relation_storage_keys)

                async with pipeline_status_lock:
                    log_message = f"Successfully deleted {len(relationships_to_delete)} relations"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

            except Exception as e:
                logger.error(f"Failed to delete relationships: {e}")
                raise Exception(f"Failed to delete relationships: {e}") from e

        # 7. Delete entities that have no remaining sources
        if entities_to_delete:
            try:
                deletion_stage = "delete_entities"
                nodes_edges_dict = (
                    await storages.chunk_entity_relation_graph.get_nodes_edges_batch(
                        list(entities_to_delete)
                    )
                )

                edges_to_delete = set()
                edges_still_exist = 0

                for entity, edges in nodes_edges_dict.items():
                    if edges:
                        for src, tgt in edges:
                            edge_tuple = tuple(sorted((src, tgt)))
                            edges_to_delete.add(edge_tuple)

                            if (
                                src in entities_to_delete
                                and tgt in entities_to_delete
                            ):
                                logger.warning(
                                    f"Edge still exists: {src} <-> {tgt}"
                                )
                            elif src in entities_to_delete:
                                logger.warning(
                                    f"Edge still exists: {src} --> {tgt}"
                                )
                            else:
                                logger.warning(
                                    f"Edge still exists: {src} <-- {tgt}"
                                )
                        edges_still_exist += 1

                if edges_still_exist:
                    logger.warning(
                        f"⚠️ {edges_still_exist} entities still has edges before deletion"
                    )

                if edges_to_delete:
                    rel_ids_to_delete = []
                    for src, tgt in edges_to_delete:
                        rel_ids_to_delete.extend(
                            [
                                compute_mdhash_id(src + tgt, prefix="rel-"),
                                compute_mdhash_id(tgt + src, prefix="rel-"),
                            ]
                        )
                    await storages.relationships_vdb.delete(rel_ids_to_delete)

                    if storages.relation_chunks:
                        relation_storage_keys = [
                            make_relation_chunk_key(src, tgt)
                            for src, tgt in edges_to_delete
                        ]
                        await storages.relation_chunks.delete(relation_storage_keys)

                    logger.info(
                        f"Cleaned {len(edges_to_delete)} residual edges from VDB and chunk-tracking storage"
                    )

                await storages.chunk_entity_relation_graph.remove_nodes(
                    list(entities_to_delete)
                )

                entity_vdb_ids = [
                    compute_mdhash_id(entity, prefix="ent-")
                    for entity in entities_to_delete
                ]
                await storages.entities_vdb.delete(entity_vdb_ids)

                if storages.entity_chunks:
                    await storages.entity_chunks.delete(list(entities_to_delete))

                async with pipeline_status_lock:
                    log_message = (
                        f"Successfully deleted {len(entities_to_delete)} entities"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

            except Exception as e:
                logger.error(f"Failed to delete entities: {e}")
                raise Exception(f"Failed to delete entities: {e}") from e

        # Persist changes to graph database before entity and relationship rebuild
        try:
            deletion_stage = "persist_pre_rebuild_changes"
            await insert_done_fn()
        except Exception as e:
            logger.error(f"Failed to persist pre-rebuild changes: {e}")
            raise Exception(f"Failed to persist pre-rebuild changes: {e}") from e

        # 8. Rebuild entities and relationships from remaining chunks
        if entities_to_rebuild or relationships_to_rebuild:
            try:
                deletion_stage = "rebuild_knowledge_graph"
                await rebuild_fn(
                    entities_to_rebuild=entities_to_rebuild,
                    relationships_to_rebuild=relationships_to_rebuild,
                    knowledge_graph_inst=storages.chunk_entity_relation_graph,
                    entities_vdb=storages.entities_vdb,
                    relationships_vdb=storages.relationships_vdb,
                    text_chunks_storage=storages.text_chunks,
                    llm_response_cache=storages.llm_response_cache,
                    config=config,
                    token=token,
                    entity_chunks_storage=storages.entity_chunks,
                    relation_chunks_storage=storages.relation_chunks,
                )

            except Exception as e:
                logger.error(f"Failed to rebuild knowledge from chunks: {e}")
                raise Exception(f"Failed to rebuild knowledge graph: {e}") from e

        # 9. Delete LLM cache while the document status still exists so a failure
        # remains retryable via the same doc_id.
        log_message = f"Document {doc_id} successfully deleted"
        if delete_llm_cache and doc_llm_cache_ids:
            if not storages.llm_response_cache:
                log_message = (
                    f"Cannot delete LLM cache for document {doc_id}: "
                    "cache storage is unavailable"
                )
                logger.error(log_message)
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                raise Exception(log_message)
            try:
                deletion_stage = "delete_llm_cache"
                await storages.llm_response_cache.delete(doc_llm_cache_ids)
                remaining_cache_ids = await _get_existing_llm_cache_ids(
                    doc_llm_cache_ids, storages.llm_response_cache
                )
                if remaining_cache_ids:
                    doc_llm_cache_ids = remaining_cache_ids
                    raise Exception(
                        f"{len(remaining_cache_ids)} LLM cache entries still exist after delete"
                    )
                cache_log_message = f"Successfully deleted {len(doc_llm_cache_ids)} LLM cache entries for document {doc_id}"
                logger.info(cache_log_message)
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = cache_log_message
                    pipeline_status["history_messages"].append(cache_log_message)
                log_message = cache_log_message
            except Exception as cache_delete_error:
                log_message = (
                    f"Failed to delete LLM cache for document {doc_id}: "
                    f"{cache_delete_error}"
                )
                logger.error(log_message)
                logger.error(traceback.format_exc())
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                raise Exception(log_message) from cache_delete_error

        # 10. Delete from full_entities and full_relations storage
        try:
            deletion_stage = "delete_doc_graph_metadata"
            await storages.full_entities.delete([doc_id])
            await storages.full_relations.delete([doc_id])
        except Exception as e:
            logger.error(f"Failed to delete from full_entities/full_relations: {e}")
            raise Exception(
                f"Failed to delete from full_entities/full_relations: {e}"
            ) from e

        # 11. Delete original document and status.
        # doc_status is deleted first so that if full_docs.delete fails, a retry
        # finds no doc_status record and treats the document as already gone,
        # rather than finding a doc_status that points to a missing full_docs entry.
        try:
            deletion_stage = "delete_doc_entries"
            in_final_delete_stage = True
            await storages.doc_status.delete([doc_id])
            await storages.full_docs.delete([doc_id])
        except Exception as e:
            logger.error(f"Failed to delete document and status: {e}")
            raise Exception(f"Failed to delete document and status: {e}") from e

        deletion_fully_completed = True
        return DeletionResult(
            status="success",
            doc_id=doc_id,
            message=log_message,
            status_code=200,
            file_path=file_path,
        )

    except Exception as e:
        original_exception = e
        error_message = f"Error while deleting document {doc_id}: {e}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        try:
            # Do not attempt to write retry state if doc_status was already deleted:
            # upsert would re-create the record as a zombie. All earlier stages still
            # have doc_status intact and can safely update it, even if some chunk/graph
            # data has already been removed.
            if doc_status_data is not None and not in_final_delete_stage:
                doc_status_data = await _update_delete_retry_state(
                    doc_id,
                    doc_status_data,
                    storages.doc_status,
                    deletion_stage=deletion_stage,
                    doc_llm_cache_ids=doc_llm_cache_ids,
                    error_message=error_message,
                    failed=True,
                )
        except Exception as status_update_error:
            logger.error(
                "Failed to update deletion retry state for document %s: %s",
                doc_id,
                status_update_error,
            )
            logger.error(traceback.format_exc())
            error_message = (
                f"{error_message}. Additionally, failed to persist retry state: "
                f"{status_update_error}. Manual cleanup may be required."
            )
        return DeletionResult(
            status="fail",
            doc_id=doc_id,
            message=error_message,
            status_code=500,
            file_path=file_path,
        )

    finally:
        # ALWAYS ensure persistence if any deletion operations were started
        if deletion_operations_started:
            try:
                await insert_done_fn()
            except Exception as persistence_error:
                persistence_error_msg = f"Failed to persist data after deletion attempt for {doc_id}: {persistence_error}"
                logger.error(persistence_error_msg)
                logger.error(traceback.format_exc())

                if deletion_fully_completed:
                    logger.error(
                        "Post-deletion persistence flush failed for %s, "
                        "but deletion completed successfully: %s",
                        doc_id,
                        persistence_error,
                    )
                elif original_exception is None:
                    # Deletion stages were in-flight but the try-block return was never
                    # reached; treat the persistence failure as the primary error.
                    return DeletionResult(
                        status="fail",
                        doc_id=doc_id,
                        message=f"Deletion completed but failed to persist changes: {persistence_error}",
                        status_code=500,
                        file_path=file_path,
                    )
                # If there was an original exception, log the persistence error but
                # don't override it — the original error result was already returned.
        else:
            logger.debug(
                f"No deletion operations were started for document {doc_id}, skipping persistence"
            )

        # Release pipeline only if WE acquired it
        if we_acquired_pipeline:
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["cancellation_requested"] = False
                completion_msg = (
                    f"Deletion process completed for document: {doc_id}"
                )
                pipeline_status["latest_message"] = completion_msg
                pipeline_status["history_messages"].append(completion_msg)
                logger.info(completion_msg)
