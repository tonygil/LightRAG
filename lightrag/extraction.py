from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from lightrag.utils import (
    logger,
    create_prefixed_exception,
    _cooperative_yield,
    CancellationToken,
)
from lightrag.llm_cache import (
    use_llm_func_with_cache,
    update_chunk_cache_list,
)
from lightrag.text_utils import (
    sanitize_and_normalize_extracted_text,
    is_float_regex,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    fix_tuple_delimiter_corruption,
)
from lightrag.tokenization import Tokenizer
from lightrag.base import (
    BaseKVStorage,
    TextChunkSchema,
)
from lightrag.prompt import PROMPTS
from lightrag.constants import (
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
)
from lightrag.exceptions import (
    PipelineCancelledException,
    ChunkTokenLimitExceededError,
)
from lightrag.config import PipelineConfig
from lightrag._summary import _handle_entity_relation_summary  # noqa: F401 (used by merge.py via this module)
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


def _truncate_entity_identifier(
    identifier: str, limit: int, chunk_key: str, identifier_role: str
) -> str:
    """Truncate entity identifiers that exceed the configured length limit."""

    if len(identifier) <= limit:
        return identifier

    display_value = identifier[:limit]
    preview = identifier[:20]  # Show first 20 characters as preview
    logger.warning(
        "%s: %s len %d > %d chars (Name: '%s...')",
        chunk_key,
        identifier_role,
        len(identifier),
        limit,
        preview,
    )
    return display_value


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    logger.warning(
                        "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                        len(_tokens),
                        chunk_token_size,
                    )
                    raise ChunkTokenLimitExceededError(
                        chunk_tokens=len(_tokens),
                        chunk_token_limit=chunk_token_size,
                        chunk_preview=chunk[:120],
                    )
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    for start in range(
                        0, len(_tokens), chunk_token_size - chunk_overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + chunk_token_size]
                        )
                        new_chunks.append(
                            (min(chunk_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append(
                {
                    "tokens": min(chunk_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 fields on ENTITY `{record_attributes[1]}` @ `{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )

        # Validate entity name after all cleaning steps
        if not entity_name or not entity_name.strip():
            logger.info(
                f"Empty entity name found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        # Process entity type with same cleaning pipeline
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        if not entity_type.strip() or any(
            char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(
                f"Entity extraction error: invalid entity type in: {record_attributes}"
            )
            return None

        # Handle comma-separated entity types by finding the first non-empty token
        if "," in entity_type:
            original = entity_type
            tokens = [t.strip() for t in entity_type.split(",")]
            non_empty = [t for t in tokens if t]
            if not non_empty:
                logger.warning(
                    f"Entity extraction error: all tokens empty after comma-split: '{original}'"
                )
                return None
            entity_type = non_empty[0]
            logger.warning(
                f"Entity type contains comma, taking first non-empty token: '{original}' -> '{entity_type}'"
            )

        # Remove spaces and convert to lowercase
        entity_type = entity_type.replace(" ", "").lower()

        # Process entity description with same cleaning pipeline
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Entity extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if (
        len(record_attributes) != 5 or "relation" not in record_attributes[0]
    ):  # treat "relationship" and "relation" interchangeable
        if len(record_attributes) > 1 and "relation" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/5 fields on RELATION `{record_attributes[1]}`~`{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        source = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        target = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        # Validate entity names after all cleaning steps
        if not source:
            logger.info(
                f"Empty source entity found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        if not target:
            logger.info(
                f"Empty target entity found after sanitization. Original: '{record_attributes[2]}'"
            )
            return None

        if source == target:
            logger.debug(
                f"Relationship source and target are the same in: {record_attributes}"
            )
            return None

        # Process keywords with same cleaning pipeline
        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")

        # Process relationship description with same cleaning pipeline
        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])
        if not edge_description.strip():
            logger.warning(
                f"Relationship extraction error: empty description for relation '{source}'~'{target}' in chunk '{chunk_key}'"
            )
            return None

        edge_source_id = chunk_key
        weight = (
            float(record_attributes[-1].strip('"').strip("'"))
            if is_float_regex(record_attributes[-1].strip('"').strip("'"))
            else 1.0
        )

        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            keywords=edge_keywords,
            source_id=edge_source_id,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.warning(
            f"Relationship extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Relationship extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def _process_extraction_result(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
) -> tuple[dict, dict]:
    """Process a single extraction result (either initial or gleaning)
    Args:
        result (str): The extraction result to process
        chunk_key (str): The chunk key for source tracking
        file_path (str): The file path for citation
        tuple_delimiter (str): Delimiter for tuple fields
        record_delimiter (str): Delimiter for records
        completion_delimiter (str): Delimiter for completion
    Returns:
        tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
    """
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    if completion_delimiter not in result:
        logger.warning(
            f"{chunk_key}: Complete delimiter can not be found in extraction result"
        )

    # Split LLL output result to records by "\n"
    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )

    # Fix LLM output format error which use tuple_delimiter to separate record instead of "\n"
    fixed_records = []
    for i, record in enumerate(records, start=1):
        record = record.strip()
        if record is None:
            continue
        entity_records = split_string_by_multi_markers(
            record, [f"{tuple_delimiter}entity{tuple_delimiter}"]
        )
        for entity_record in entity_records:
            if not entity_record.startswith("entity") and not entity_record.startswith(
                "relation"
            ):
                entity_record = f"entity<|{entity_record}"
            entity_relation_records = split_string_by_multi_markers(
                # treat "relationship" and "relation" interchangeable
                entity_record,
                [
                    f"{tuple_delimiter}relationship{tuple_delimiter}",
                    f"{tuple_delimiter}relation{tuple_delimiter}",
                ],
            )
            for entity_relation_record in entity_relation_records:
                if not entity_relation_record.startswith(
                    "entity"
                ) and not entity_relation_record.startswith("relation"):
                    entity_relation_record = (
                        f"relation{tuple_delimiter}{entity_relation_record}"
                    )
                fixed_records.append(entity_relation_record)
        await _cooperative_yield(i, every=8)

    if len(fixed_records) != len(records):
        logger.warning(
            f"{chunk_key}: LLM output format error; find LLM use {tuple_delimiter} as record separators instead new-line"
        )

    delimiter_core = tuple_delimiter[2:-2]  # Extract "#" from "<|#|>"
    delimiter_core_lower = delimiter_core.lower()
    for i, record in enumerate(fixed_records, start=1):
        record = record.strip()
        if record is None:
            continue

        # Fix various forms of tuple_delimiter corruption from the LLM output using the dedicated function
        record = fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        if delimiter_core != delimiter_core_lower:
            # change delimiter_core to lower case, and fix again
            record = fix_tuple_delimiter_corruption(
                record, delimiter_core_lower, tuple_delimiter
            )

        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

        # Try to parse as entity
        entity_data = _handle_single_entity_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if entity_data is not None:
            truncated_name = _truncate_entity_identifier(
                entity_data["entity_name"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Entity name",
            )
            entity_data["entity_name"] = truncated_name
            maybe_nodes[truncated_name].append(entity_data)
            await _cooperative_yield(i, every=8)
            continue

        # Try to parse as relationship
        relationship_data = _handle_single_relationship_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if relationship_data is not None:
            truncated_source = _truncate_entity_identifier(
                relationship_data["src_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            truncated_target = _truncate_entity_identifier(
                relationship_data["tgt_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            relationship_data["src_id"] = truncated_source
            relationship_data["tgt_id"] = truncated_target
            maybe_edges[(truncated_source, truncated_target)].append(relationship_data)
        await _cooperative_yield(i, every=8)

    return dict(maybe_nodes), dict(maybe_edges)


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    config: PipelineConfig,
    token: CancellationToken | None = None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
) -> list:
    # Check for cancellation at the start of entity extraction
    if token is not None:
        await token.raise_if_cancelled()

    use_llm_func: callable = config.llm_model_func
    entity_extract_max_gleaning = config.entity_extract_max_gleaning

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = config.language
    entity_types = config.entity_types

    examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")

        # Create cache keys collector for batch processing
        cache_keys_collector = []

        # Get initial extraction
        # Format system prompt without input_text for each chunk (enables OpenAI prompt caching across chunks)
        entity_extraction_system_prompt = PROMPTS[
            "entity_extraction_system_prompt"
        ].format(**context_base)
        # Format user prompts with input_text for each chunk
        entity_extraction_user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            **{**context_base, "input_text": content}
        )
        entity_continue_extraction_user_prompt = PROMPTS[
            "entity_continue_extraction_user_prompt"
        ].format(**{**context_base, "input_text": content})

        final_result, timestamp = await use_llm_func_with_cache(
            entity_extraction_user_prompt,
            use_llm_func,
            system_prompt=entity_extraction_system_prompt,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )

        history = pack_user_ass_to_openai_messages(
            entity_extraction_user_prompt, final_result
        )

        # Process initial extraction with file path
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result,
            chunk_key,
            timestamp,
            file_path,
            tuple_delimiter=context_base["tuple_delimiter"],
            completion_delimiter=context_base["completion_delimiter"],
        )

        # Process additional gleaning results only 1 time when entity_extract_max_gleaning is greater than zero.
        if entity_extract_max_gleaning > 0:
            # Calculate total tokens for the gleaning request to prevent context window overflow
            tokenizer = config.tokenizer
            max_input_tokens = config.max_extract_input_tokens

            # Approximate total tokens: system prompt + history + user prompt.
            # This slightly underestimates actual API usage (missing role/framing tokens)
            # but is sufficient as a safety guard against context window overflow.
            history_str = json.dumps(history, ensure_ascii=False)
            full_context_str = (
                entity_extraction_system_prompt
                + history_str
                + entity_continue_extraction_user_prompt
            )
            token_count = len(tokenizer.encode(full_context_str))

            if token_count > max_input_tokens:
                logger.warning(
                    f"Gleaning stopped for chunk {chunk_key}: Input tokens ({token_count}) exceeded limit ({max_input_tokens})."
                )
            else:
                glean_result, timestamp = await use_llm_func_with_cache(
                    entity_continue_extraction_user_prompt,
                    use_llm_func,
                    system_prompt=entity_extraction_system_prompt,
                    llm_response_cache=llm_response_cache,
                    history_messages=history,
                    cache_type="extract",
                    chunk_id=chunk_key,
                    cache_keys_collector=cache_keys_collector,
                )

                # Process gleaning result separately with file path
                glean_nodes, glean_edges = await _process_extraction_result(
                    glean_result,
                    chunk_key,
                    timestamp,
                    file_path,
                    tuple_delimiter=context_base["tuple_delimiter"],
                    completion_delimiter=context_base["completion_delimiter"],
                )

                # Merge results - compare description lengths to choose better version
                for i, (entity_name, glean_entities) in enumerate(
                    glean_nodes.items(), start=1
                ):
                    if entity_name in maybe_nodes:
                        # Compare description lengths and keep the better one
                        original_desc_len = len(
                            maybe_nodes[entity_name][0].get("description", "") or ""
                        )
                        glean_desc_len = len(
                            glean_entities[0].get("description", "") or ""
                        )

                        if glean_desc_len > original_desc_len:
                            maybe_nodes[entity_name] = list(glean_entities)
                        # Otherwise keep original version
                    else:
                        # New entity from gleaning stage
                        maybe_nodes[entity_name] = list(glean_entities)
                    await _cooperative_yield(i, every=8)

                for i, (edge_key, glean_edge_list) in enumerate(
                    glean_edges.items(), start=1
                ):
                    if edge_key in maybe_edges:
                        # Compare description lengths and keep the better one
                        original_desc_len = len(
                            maybe_edges[edge_key][0].get("description", "") or ""
                        )
                        glean_desc_len = len(
                            glean_edge_list[0].get("description", "") or ""
                        )

                        if glean_desc_len > original_desc_len:
                            maybe_edges[edge_key] = list(glean_edge_list)
                        # Otherwise keep original version
                    else:
                        # New edge from gleaning stage
                        maybe_edges[edge_key] = list(glean_edge_list)
                    await _cooperative_yield(i, every=8)

        # Batch update chunk's llm_cache_list with all collected cache keys
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel {chunk_key}"
        logger.info(log_message)
        if token is not None:
            await token.post_status(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Get max async tasks limit from config
    chunk_max_async = config.llm_model_max_async
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            # Check for cancellation before processing chunk
            if token is not None:
                await token.raise_if_cancelled()

            try:
                result = await _process_single_content(chunk)
                # Yield once between chunk completions so API coroutines can resume
                # even when many chunk tasks are hitting cache and finishing quickly.
                await asyncio.sleep(0)
                return result
            except Exception as e:
                chunk_id = chunk[0]  # Extract chunk_id from chunk[0]
                prefixed_exception = create_prefixed_exception(e, chunk_id)
                raise prefixed_exception from e

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for tasks to complete or for the first exception to occur
    # This allows us to cancel remaining tasks if any task fails
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None
    chunk_results = []

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                chunk_results.append(task.result())
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Add progress prefix to the exception message
        progress_prefix = f"C[{processed_chunks + 1}/{total_chunks}]"

        # Re-raise the original exception with a prefix
        prefixed_exception = create_prefixed_exception(first_exception, progress_prefix)
        raise prefixed_exception from first_exception

    # If all tasks completed successfully, chunk_results already contains the results
    # Return the chunk_results for later processing in merge_nodes_and_edges
    return chunk_results
