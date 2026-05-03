"""Document processing pipeline extracted from LightRAG.

Owns the entire pipeline loop: consistency validation, chunking, extraction,
merge, and status tracking. LightRAG delegates to run_pipeline_loop() and
remains a thin orchestrator.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.config import PipelineConfig
from lightrag.exceptions import PipelineCancelledException
from lightrag.merge import merge_nodes_and_edges
from lightrag.storage_set import StorageSet
from lightrag.utils import (
    CancellationToken,
    logger,
)
from lightrag.text_utils import compute_mdhash_id

StorageBundle = StorageSet  # backward-compat alias used by deletion.py


def _chunk_fields_from_status_doc(
    status_doc: DocProcessingStatus,
) -> tuple[list[str], int]:
    """Return (chunks_list, chunks_count) preserved from a status document.

    Filters out any non-string or empty chunk IDs.  When chunks_count is
    absent or invalid, it is inferred from the length of chunks_list.
    """
    chunks_list: list[str] = []
    if isinstance(status_doc.chunks_list, list):
        chunks_list = [
            chunk_id
            for chunk_id in status_doc.chunks_list
            if isinstance(chunk_id, str) and chunk_id
        ]

    if isinstance(status_doc.chunks_count, int) and status_doc.chunks_count >= 0:
        return chunks_list, status_doc.chunks_count

    return chunks_list, len(chunks_list)


def _resolve_doc_file_path(
    status_doc: DocProcessingStatus | None = None,
    content_data: dict[str, Any] | None = None,
) -> str:
    """Resolve the best available document file path.

    Prefer a non-placeholder path from doc_status, then fall back to full_docs.
    This avoids overwriting historical file paths with placeholder values during
    retries or early-cancellation paths.
    """
    placeholder_paths = {"", "no-file-path", "unknown_source"}

    def _normalize_path(candidate: Any) -> str | None:
        if not isinstance(candidate, str):
            return None
        normalized = candidate.strip()
        if not normalized:
            return None
        return normalized

    candidates = [
        _normalize_path(getattr(status_doc, "file_path", None)),
        _normalize_path(content_data.get("file_path") if content_data else None),
    ]

    for candidate in candidates:
        if candidate and candidate not in placeholder_paths:
            return candidate

    for candidate in candidates:
        if candidate:
            return "unknown_source" if candidate == "no-file-path" else candidate

    return "unknown_source"


async def validate_and_fix_document_consistency(
    to_process_docs: dict[str, DocProcessingStatus],
    token: CancellationToken,
    full_docs: Any,
    doc_status: Any,
) -> dict[str, DocProcessingStatus]:
    """Validate and fix document data consistency by deleting inconsistent entries, but preserve failed documents."""
    inconsistent_docs = []
    failed_docs_to_preserve = []

    for doc_id, status_doc in to_process_docs.items():
        content_data = await full_docs.get_by_id(doc_id)
        if not content_data:
            if (
                hasattr(status_doc, "status")
                and status_doc.status == DocStatus.FAILED
            ):
                failed_docs_to_preserve.append(doc_id)
            else:
                inconsistent_docs.append(doc_id)

    if failed_docs_to_preserve:
        preserve_message = f"Preserving {len(failed_docs_to_preserve)} failed document entries for manual review"
        logger.info(preserve_message)
        await token.post_status(preserve_message)

        for doc_id in failed_docs_to_preserve:
            to_process_docs.pop(doc_id, None)

    if inconsistent_docs:
        summary_message = (
            f"Inconsistent document entries found: {len(inconsistent_docs)}"
        )
        logger.info(summary_message)
        await token.post_status(summary_message)

        for doc_id in inconsistent_docs:
            try:
                status_doc = to_process_docs[doc_id]
                file_path = _resolve_doc_file_path(status_doc=status_doc)

                await doc_status.delete([doc_id])

                log_message = f"Deleted inconsistent entry: {doc_id} ({file_path})"
                logger.info(log_message)
                await token.post_status(log_message)

                to_process_docs.pop(doc_id, None)

            except Exception as e:
                error_message = f"Failed to delete entry: {doc_id} - {str(e)}"
                logger.error(error_message)
                await token.post_status(error_message)

    docs_to_reset = {}
    reset_count = 0

    for doc_id, status_doc in to_process_docs.items():
        content_data = await full_docs.get_by_id(doc_id)
        if content_data:
            if hasattr(status_doc, "status") and status_doc.status in [
                DocStatus.PROCESSING,
                DocStatus.FAILED,
            ]:
                preserved_chunks_list, preserved_chunks_count = (
                    _chunk_fields_from_status_doc(status_doc)
                )
                resolved_file_path = _resolve_doc_file_path(
                    status_doc=status_doc,
                    content_data=content_data,
                )
                docs_to_reset[doc_id] = {
                    "status": DocStatus.PENDING,
                    "content_summary": status_doc.content_summary,
                    "content_length": status_doc.content_length,
                    "chunks_count": preserved_chunks_count,
                    "chunks_list": preserved_chunks_list,
                    "created_at": status_doc.created_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": resolved_file_path,
                    "track_id": getattr(status_doc, "track_id", ""),
                    "error_msg": "",
                    "metadata": {},
                }
                status_doc.status = DocStatus.PENDING
                status_doc.file_path = resolved_file_path
                reset_count += 1

    if docs_to_reset:
        await doc_status.upsert(docs_to_reset)

        reset_message = f"Reset {reset_count} documents from PROCESSING/FAILED to PENDING status"
        logger.info(reset_message)
        await token.post_status(reset_message)

    return to_process_docs


async def process_document(
    doc_id: str,
    status_doc: DocProcessingStatus,
    split_by_character: str | None,
    split_by_character_only: bool,
    token: CancellationToken,
    semaphore: asyncio.Semaphore,
    storages: StorageBundle,
    config: PipelineConfig,
    chunking_func: Callable,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
    pipeline_status: dict,
    total_files: int,
    processed_count_ref: list,
    process_extract_fn: Callable,
    insert_done_fn: Callable,
) -> None:
    """Process a single document through chunking, extraction, and merge."""
    file_path = _resolve_doc_file_path(status_doc=status_doc)
    current_file_number = 0
    file_extraction_stage_ok = False
    processing_start_time = int(time.time())
    first_stage_tasks = []
    entity_relation_task = None
    chunks: dict[str, Any] = {}
    content_data: dict[str, Any] | None = None

    def get_failed_chunk_snapshot() -> tuple[list[str], int]:
        if chunks:
            chunk_ids = list(chunks.keys())
            return chunk_ids, len(chunk_ids)
        return _chunk_fields_from_status_doc(status_doc)

    async with semaphore:
        # Initialize to prevent UnboundLocalError in error handling
        first_stage_tasks = []
        entity_relation_task = None
        try:
            # Resolve file_path from full_docs before honoring a queued
            # cancellation so corrupted doc_status placeholders do not
            # get written back again during retry/cancel flows.
            content_data = await storages.full_docs.get_by_id(doc_id)
            if content_data:
                file_path = _resolve_doc_file_path(
                    status_doc=status_doc,
                    content_data=content_data,
                )
                status_doc.file_path = file_path

            # Check for cancellation before starting document processing.
            # file_path is resolved before this check so queued documents
            # do not lose their source path on early cancellation.
            await token.raise_if_cancelled()

            async with token._lock:
                # Update processed file count and save current file number
                processed_count_ref[0] += 1
                current_file_number = processed_count_ref[0]
                pipeline_status["cur_batch"] = processed_count_ref[0]

                log_message = f"Extracting stage {current_file_number}/{total_files}: {file_path}"
                logger.info(log_message)
                pipeline_status["history_messages"].append(log_message)
                log_message = f"Processing d-id: {doc_id}"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Prevent memory growth: keep only latest 5000 messages when exceeding 10000
                if len(pipeline_status["history_messages"]) > 10000:
                    logger.info(
                        f"Trimming pipeline history from {len(pipeline_status['history_messages'])} to 5000 messages"
                    )
                    # Trim in place so Manager.list-backed shared state
                    # remains appendable and visible across processes.
                    del pipeline_status["history_messages"][:-5000]

            # Get document content from full_docs
            if not content_data:
                raise Exception(
                    f"Document content not found in full_docs for doc_id: {doc_id}"
                )
            content = content_data["content"]

            # Call chunking function, supporting both sync and async implementations
            chunking_result = chunking_func(
                config.tokenizer,
                content,
                split_by_character,
                split_by_character_only,
                chunk_overlap_token_size,
                chunk_token_size,
            )

            # If result is awaitable, await to get actual result
            if inspect.isawaitable(chunking_result):
                chunking_result = await chunking_result

            # Validate return type
            if not isinstance(chunking_result, (list, tuple)):
                raise TypeError(
                    f"chunking_func must return a list or tuple of dicts, "
                    f"got {type(chunking_result)}"
                )

            # Build chunks dictionary
            chunks = {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                    **dp,
                    "full_doc_id": doc_id,
                    "file_path": file_path,
                    "llm_cache_list": [],
                }
                for dp in chunking_result
            }

            if not chunks:
                logger.warning("No document chunks to process")

            # Record processing start time
            processing_start_time = int(time.time())

            # Check for cancellation before entity extraction
            await token.raise_if_cancelled()

            # Process document in two stages
            # Stage 1: Process text chunks and docs (parallel execution)
            doc_status_task = asyncio.create_task(
                storages.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.PROCESSING,
                            "chunks_count": len(chunks),
                            "chunks_list": list(chunks.keys()),
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "created_at": status_doc.created_at,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "file_path": file_path,
                            "track_id": status_doc.track_id,
                            "metadata": {
                                "processing_start_time": processing_start_time
                            },
                        }
                    }
                )
            )
            chunks_vdb_task = asyncio.create_task(
                storages.chunks_vdb.upsert(chunks)
            )
            text_chunks_task = asyncio.create_task(
                storages.text_chunks.upsert(chunks)
            )

            first_stage_tasks = [doc_status_task, chunks_vdb_task, text_chunks_task]
            entity_relation_task = None

            # Execute first stage tasks
            await asyncio.gather(*first_stage_tasks)

            # Stage 2: Process entity relation graph (after text_chunks are saved)
            entity_relation_task = asyncio.create_task(
                process_extract_fn(chunks, token)
            )
            chunk_results = await entity_relation_task
            file_extraction_stage_ok = True

        except Exception as e:
            # Check if this is a user cancellation
            if isinstance(e, PipelineCancelledException):
                error_msg = f"User cancelled {current_file_number}/{total_files}: {file_path}"
                logger.warning(error_msg)
                await token.post_status(error_msg)
            else:
                logger.error(traceback.format_exc())
                error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                logger.error(error_msg)
                async with token._lock:
                    pipeline_status["latest_message"] = error_msg
                    pipeline_status["history_messages"].append(
                        traceback.format_exc()
                    )
                    pipeline_status["history_messages"].append(error_msg)

            # Cancel tasks that are not yet completed
            all_tasks = first_stage_tasks + (
                [entity_relation_task] if entity_relation_task else []
            )
            for task in all_tasks:
                if task and not task.done():
                    task.cancel()

            # Persistent llm cache with error handling
            if storages.llm_response_cache:
                try:
                    await storages.llm_response_cache.index_done_callback()
                except Exception as persist_error:
                    logger.error(f"Failed to persist LLM cache: {persist_error}")

            # Record processing end time for failed case
            processing_end_time = int(time.time())
            failed_chunks_list, failed_chunks_count = get_failed_chunk_snapshot()

            # Update document status to failed
            await storages.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.FAILED,
                        "error_msg": str(e),
                        "chunks_count": failed_chunks_count,
                        "chunks_list": failed_chunks_list,
                        "content_summary": status_doc.content_summary,
                        "content_length": status_doc.content_length,
                        "created_at": status_doc.created_at,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": file_path,
                        "track_id": status_doc.track_id,
                        "metadata": {
                            "processing_start_time": processing_start_time,
                            "processing_end_time": processing_end_time,
                        },
                    }
                }
            )

        # Concurrency is controlled by keyed lock for individual entities and relationships
        if file_extraction_stage_ok:
            try:
                # Check for cancellation before merge
                await token.raise_if_cancelled()

                # Use chunk_results from entity_relation_task
                await merge_nodes_and_edges(
                    chunk_results=chunk_results,
                    knowledge_graph_inst=storages.chunk_entity_relation_graph,
                    entity_vdb=storages.entities_vdb,
                    relationships_vdb=storages.relationships_vdb,
                    config=config,
                    full_entities_storage=storages.full_entities,
                    full_relations_storage=storages.full_relations,
                    doc_id=doc_id,
                    token=token,
                    llm_response_cache=storages.llm_response_cache,
                    entity_chunks_storage=storages.entity_chunks,
                    relation_chunks_storage=storages.relation_chunks,
                    current_file_number=current_file_number,
                    total_files=total_files,
                    file_path=file_path,
                )

                # Record processing end time
                processing_end_time = int(time.time())

                await storages.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.PROCESSED,
                            "chunks_count": len(chunks),
                            "chunks_list": list(chunks.keys()),
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "created_at": status_doc.created_at,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "file_path": file_path,
                            "track_id": status_doc.track_id,
                            "metadata": {
                                "processing_start_time": processing_start_time,
                                "processing_end_time": processing_end_time,
                            },
                        }
                    }
                )

                # Call insert_done_fn after processing each file
                await insert_done_fn()

                log_message = f"Completed processing file {current_file_number}/{total_files}: {file_path}"
                logger.info(log_message)
                await token.post_status(log_message)

            except Exception as e:
                # Check if this is a user cancellation
                if isinstance(e, PipelineCancelledException):
                    error_msg = f"User cancelled during merge {current_file_number}/{total_files}: {file_path}"
                    logger.warning(error_msg)
                    await token.post_status(error_msg)
                else:
                    logger.error(traceback.format_exc())
                    error_msg = f"Merging stage failed in document {current_file_number}/{total_files}: {file_path}"
                    logger.error(error_msg)
                    async with token._lock:
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(
                            traceback.format_exc()
                        )
                        pipeline_status["history_messages"].append(error_msg)

                # Persistent llm cache with error handling
                if storages.llm_response_cache:
                    try:
                        await storages.llm_response_cache.index_done_callback()
                    except Exception as persist_error:
                        logger.error(f"Failed to persist LLM cache: {persist_error}")

                # Record processing end time for failed case
                processing_end_time = int(time.time())
                failed_chunks_list, failed_chunks_count = get_failed_chunk_snapshot()

                # Update document status to failed
                await storages.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.FAILED,
                            "error_msg": str(e),
                            "chunks_count": failed_chunks_count,
                            "chunks_list": failed_chunks_list,
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "created_at": status_doc.created_at,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "file_path": file_path,
                            "track_id": status_doc.track_id,
                            "metadata": {
                                "processing_start_time": processing_start_time,
                                "processing_end_time": processing_end_time,
                            },
                        }
                    }
                )


async def run_pipeline_loop(
    to_process_docs: dict[str, DocProcessingStatus],
    token: CancellationToken,
    pipeline_status: dict,
    split_by_character: str | None,
    split_by_character_only: bool,
    storages: StorageBundle,
    config: PipelineConfig,
    chunking_func: Callable,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
    max_parallel_insert: int,
    process_extract_fn: Callable,
    insert_done_fn: Callable,
) -> None:
    """Run the main document processing loop until all docs are processed or cancelled."""
    while True:
        # Check for cancellation request at the start of main loop
        async with token._lock:
            if pipeline_status.get("cancellation_requested", False):
                # Clear pending request
                pipeline_status["request_pending"] = False
                # Clear cancellation flag
                pipeline_status["cancellation_requested"] = False

                log_message = "Pipeline cancelled by user"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Exit directly, skipping request_pending check
                return

        if not to_process_docs:
            log_message = "All enqueued documents have been processed"
            logger.info(log_message)
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)
            break

        # Validate document data consistency and fix any issues as part of the pipeline
        to_process_docs = await validate_and_fix_document_consistency(
            to_process_docs, token, storages.full_docs, storages.doc_status
        )

        if not to_process_docs:
            log_message = (
                "No valid documents to process after consistency check"
            )
            logger.info(log_message)
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)
            break

        log_message = f"Processing {len(to_process_docs)} document(s)"
        logger.info(log_message)

        # Update pipeline_status, batchs now represents the total number of files to be processed
        pipeline_status["docs"] = len(to_process_docs)
        pipeline_status["batchs"] = len(to_process_docs)
        pipeline_status["cur_batch"] = 0
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

        # Get first document's file path and total count for job name
        first_doc_id, first_doc = next(iter(to_process_docs.items()))
        first_doc_path = first_doc.file_path

        # Handle cases where first_doc_path is None
        if first_doc_path:
            path_prefix = first_doc_path[:20] + (
                "..." if len(first_doc_path) > 20 else ""
            )
        else:
            path_prefix = "unknown_source"

        total_files = len(to_process_docs)
        job_name = f"{path_prefix}[{total_files} files]"
        pipeline_status["job_name"] = job_name

        # Mutable counter shared across concurrent process_document coroutines
        processed_count_ref = [0]
        # Create a semaphore to limit the number of concurrent file processing
        semaphore = asyncio.Semaphore(max_parallel_insert)

        # Create processing tasks for all documents
        doc_tasks = []
        for doc_id, status_doc in to_process_docs.items():
            doc_tasks.append(
                process_document(
                    doc_id=doc_id,
                    status_doc=status_doc,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    token=token,
                    semaphore=semaphore,
                    storages=storages,
                    config=config,
                    chunking_func=chunking_func,
                    chunk_overlap_token_size=chunk_overlap_token_size,
                    chunk_token_size=chunk_token_size,
                    pipeline_status=pipeline_status,
                    total_files=total_files,
                    processed_count_ref=processed_count_ref,
                    process_extract_fn=process_extract_fn,
                    insert_done_fn=insert_done_fn,
                )
            )

        # Wait for all document processing to complete
        try:
            await asyncio.gather(*doc_tasks)
        except PipelineCancelledException:
            # Cancel all remaining tasks
            for task in doc_tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete cancellation
            await asyncio.wait(doc_tasks, return_when=asyncio.ALL_COMPLETED)

            # Exit directly (document statuses already updated in process_document)
            return

        # Check if there's a pending request to process more documents (with lock)
        has_pending_request = False
        async with token._lock:
            has_pending_request = pipeline_status.get("request_pending", False)
            if has_pending_request:
                # Clear the request flag before checking for more documents
                pipeline_status["request_pending"] = False

        if not has_pending_request:
            break

        log_message = "Processing additional documents due to pending request"
        logger.info(log_message)
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

        # Check for pending documents again
        to_process_docs = await storages.doc_status.get_docs_by_statuses(
            [DocStatus.PROCESSING, DocStatus.FAILED, DocStatus.PENDING]
        )
