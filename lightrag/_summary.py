from __future__ import annotations

import json
from functools import partial

from lightrag.utils import (
    logger,
    Tokenizer,
    truncate_list_by_token_size,
    use_llm_func_with_cache,
    _cooperative_yield,
)
from lightrag.base import (
    BaseKVStorage,
)
from lightrag.prompt import PROMPTS
from lightrag.config import PipelineConfig


async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    separator: str,
    config: PipelineConfig,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing tokenizer and limits
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Tuple of (final_summarized_description_string, llm_was_used_boolean)
    """
    # Handle empty input
    if not description_list:
        return "", False

    # If only one description, return it directly (no need for LLM call)
    if len(description_list) == 1:
        return description_list[0], False

    # Get configuration
    tokenizer: Tokenizer = config.tokenizer
    summary_context_size = config.summary_context_size
    summary_max_tokens = config.summary_max_tokens
    force_llm_summary_on_merge = config.force_llm_summary_on_merge

    current_list = description_list[:]  # Copy the list to avoid modifying original
    llm_was_used = False  # Track whether LLM was used during the entire process

    # Iterative map-reduce process
    while True:
        # Calculate total tokens in current list while periodically yielding so
        # a large merge does not monopolize the event loop in single-worker mode.
        total_tokens = 0
        for i, desc in enumerate(current_list, start=1):
            total_tokens += len(tokenizer.encode(desc))
            await _cooperative_yield(i, every=32)

        # If total length is within limits, perform final summarization
        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (
                len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens
            ):
                # no LLM needed, just join the descriptions
                final_description = separator.join(current_list)
                return final_description if final_description else "", llm_was_used
            else:
                if total_tokens > summary_context_size and len(current_list) <= 2:
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize description found"
                    )
                # Final summarization of remaining descriptions - LLM will be used
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    config,
                    llm_response_cache,
                )
                return final_summary, True  # LLM was used for final summarization

        # Need to split into chunks - Map phase
        # Ensure each chunk has minimum 2 descriptions to guarantee progress
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Currently least 3 descriptions in current_list
        for i, desc in enumerate(current_list, start=1):
            desc_tokens = len(tokenizer.encode(desc))
            await _cooperative_yield(i, every=32)

            # If adding current description would exceed limit, finalize current chunk
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                # Ensure we have at least 2 descriptions in the chunk (when possible)
                if len(current_chunk) == 1:
                    # Force add one more description to ensure minimum 2 per chunk
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize description found"
                    )
                    current_chunk = []  # next group is empty
                    current_tokens = 0
                else:  # curren_chunk is ready for summary in reduce phase
                    chunks.append(current_chunk)
                    current_chunk = [desc]  # leave it for next group
                    current_tokens = desc_tokens
            else:
                current_chunk.append(desc)
                current_tokens += desc_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )

        # Reduce phase: summarize each group from chunks
        new_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                new_summaries.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    chunk,
                    config,
                    llm_response_cache,
                )
                new_summaries.append(summary)
                llm_was_used = True  # Mark that LLM was used in reduce phase

        # Update current list with new summaries for next iteration
        current_list = new_summaries


async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    config: PipelineConfig,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Helper function to summarize a list of descriptions using LLM.

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        descriptions: List of description strings to summarize
        global_config: Global configuration containing LLM function and settings
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Summarized description string
    """
    use_llm_func: callable = config.llm_model_func
    # Apply higher priority (8) to entity/relation summary tasks
    use_llm_func = partial(use_llm_func, _priority=8)

    language = config.language

    summary_length_recommended = config.summary_length_recommended

    prompt_template = PROMPTS["summarize_entity_descriptions"]

    # Convert descriptions to JSONL format and apply token-based truncation
    tokenizer = config.tokenizer
    summary_context_size = config.summary_context_size

    # Create list of JSON objects with "Description" field
    json_descriptions = [{"Description": desc} for desc in description_list]

    # Use truncate_list_by_token_size for length truncation
    truncated_json_descriptions = truncate_list_by_token_size(
        json_descriptions,
        key=lambda x: json.dumps(x, ensure_ascii=False),
        max_token_size=summary_context_size,
        tokenizer=tokenizer,
    )

    # Convert to JSONL format (one JSON object per line)
    joined_descriptions = "\n".join(
        json.dumps(desc, ensure_ascii=False) for desc in truncated_json_descriptions
    )

    # Prepare context for the prompt
    context_base = dict(
        description_type=description_type,
        description_name=description_name,
        description_list=joined_descriptions,
        summary_length=summary_length_recommended,
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)

    # Use LLM function with cache (higher priority for summary generation)
    summary, _ = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="summary",
    )

    # Check summary token length against embedding limit
    embedding_token_limit = config.embedding_token_limit
    if embedding_token_limit is not None and summary:
        tokenizer = config.tokenizer
        summary_token_count = len(tokenizer.encode(summary))
        threshold = int(embedding_token_limit)

        if summary_token_count > threshold:
            logger.warning(
                f"Summary tokens({summary_token_count}) exceeds embedding_token_limit({embedding_token_limit}) "
                f" for {description_type}: {description_name}"
            )

    return summary
