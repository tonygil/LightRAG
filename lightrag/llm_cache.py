"""LLM response caching machinery.

Covers cache key generation, cache read/write, and the main LLM-with-cache orchestrator.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lightrag.text_utils import (
    compute_args_hash,
    remove_think_tags,
    sanitize_text_for_encoding,
)

if TYPE_CHECKING:
    from lightrag.base import BaseKVStorage

logger = logging.getLogger("lightrag")

statistic_data = {"llm_call": 0, "llm_cache": 0, "embed_call": 0}


def generate_cache_key(mode: str, cache_type: str, hash_value: str) -> str:
    """Generate a flattened cache key in the format {mode}:{cache_type}:{hash}"""
    return f"{mode}:{cache_type}:{hash_value}"


def parse_cache_key(cache_key: str) -> tuple[str, str, str] | None:
    """Parse a flattened cache key back into its components.

    Returns:
        (mode, cache_type, hash) or None if invalid format
    """
    parts = cache_key.split(":", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None


async def handle_cache(
    hashing_kv,
    args_hash,
    prompt,
    mode="default",
    cache_type="unknown",
) -> tuple[str, int] | None:
    """Check cache and return (content, timestamp) or None on miss.

    Returns:
        tuple[str, int] | None: (content, create_time) if cache hit, None if cache miss
    """
    if hashing_kv is None:
        return None

    if mode != "default":  # handle cache for all type of query
        if not hashing_kv.global_config.get("enable_llm_cache"):
            return None
    else:  # handle cache for entity extraction
        if not hashing_kv.global_config.get("enable_llm_cache_for_entity_extract"):
            return None

    flattened_key = generate_cache_key(mode, cache_type, args_hash)
    cache_entry = await hashing_kv.get_by_id(flattened_key)
    if cache_entry:
        logger.debug(f"Flattened cache hit(key:{flattened_key})")
        content = cache_entry["return"]
        timestamp = cache_entry.get("create_time", 0)
        return content, timestamp

    logger.debug(f"Cache missed(mode:{mode} type:{cache_type})")
    return None


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    mode: str = "default"
    cache_type: str = "query"
    chunk_id: str | None = None
    queryparam: dict | None = None


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """Save data to cache using flattened key structure."""
    if hashing_kv is None or not cache_data.content:
        return

    if hasattr(cache_data.content, "__aiter__"):
        logger.debug("Streaming response detected, skipping cache")
        return

    flattened_key = generate_cache_key(
        cache_data.mode, cache_data.cache_type, cache_data.args_hash
    )

    existing_cache = await hashing_kv.get_by_id(flattened_key)
    if existing_cache:
        existing_content = existing_cache.get("return")
        if existing_content == cache_data.content:
            logger.warning(
                f"Cache duplication detected for {flattened_key}, skipping update"
            )
            return

    cache_entry = {
        "return": cache_data.content,
        "cache_type": cache_data.cache_type,
        "chunk_id": cache_data.chunk_id if cache_data.chunk_id is not None else None,
        "original_prompt": cache_data.prompt,
        "queryparam": cache_data.queryparam
        if cache_data.queryparam is not None
        else None,
    }

    logger.info(f" == LLM cache == saving: {flattened_key}")
    await hashing_kv.upsert({flattened_key: cache_entry})


async def update_chunk_cache_list(
    chunk_id: str,
    text_chunks_storage: "BaseKVStorage",
    cache_keys: list[str],
    cache_scenario: str = "batch_update",
) -> None:
    """Update chunk's llm_cache_list with the given cache keys."""
    if not cache_keys:
        return

    try:
        chunk_data = await text_chunks_storage.get_by_id(chunk_id)
        if chunk_data:
            if "llm_cache_list" not in chunk_data:
                chunk_data["llm_cache_list"] = []

            existing_keys = set(chunk_data["llm_cache_list"])
            new_keys = [key for key in cache_keys if key not in existing_keys]

            if new_keys:
                chunk_data["llm_cache_list"].extend(new_keys)
                await text_chunks_storage.upsert({chunk_id: chunk_data})
                logger.debug(
                    f"Updated chunk {chunk_id} with {len(new_keys)} cache keys ({cache_scenario})"
                )
    except Exception as e:
        logger.warning(
            f"Failed to update chunk {chunk_id} with cache references on {cache_scenario}: {e}"
        )


async def use_llm_func_with_cache(
    user_prompt: str,
    use_llm_func: callable,
    llm_response_cache: "BaseKVStorage | None" = None,
    system_prompt: str | None = None,
    max_tokens: int = None,
    history_messages: list[dict[str, str]] = None,
    cache_type: str = "extract",
    chunk_id: str | None = None,
    cache_keys_collector: list = None,
) -> tuple[str, int]:
    """Call LLM function with cache support and text sanitization.

    If cache is available and enabled, retrieve result from cache; otherwise call LLM
    and save result to cache. Applies text sanitization to prevent UTF-8 encoding errors.

    Returns:
        tuple[str, int]: (response_text, timestamp)
            - For cache hits: (content, cache_create_time)
            - For cache misses: (content, current_timestamp)
    """
    safe_user_prompt = sanitize_text_for_encoding(user_prompt)
    safe_system_prompt = (
        sanitize_text_for_encoding(system_prompt) if system_prompt else None
    )

    safe_history_messages = None
    if history_messages:
        safe_history_messages = []
        for i, msg in enumerate(history_messages):
            safe_msg = msg.copy()
            if "content" in safe_msg:
                safe_msg["content"] = sanitize_text_for_encoding(safe_msg["content"])
            safe_history_messages.append(safe_msg)
        history = json.dumps(safe_history_messages, ensure_ascii=False)
    else:
        history = None

    if llm_response_cache:
        prompt_parts = []
        if safe_user_prompt:
            prompt_parts.append(safe_user_prompt)
        if safe_system_prompt:
            prompt_parts.append(safe_system_prompt)
        if history:
            prompt_parts.append(history)
        _prompt = "\n".join(prompt_parts)

        arg_hash = compute_args_hash(_prompt)
        cache_key = generate_cache_key("default", cache_type, arg_hash)

        cached_result = await handle_cache(
            llm_response_cache,
            arg_hash,
            _prompt,
            "default",
            cache_type=cache_type,
        )
        if cached_result:
            content, timestamp = cached_result
            logger.debug(f"Found cache for {arg_hash}")
            statistic_data["llm_cache"] += 1

            if cache_keys_collector is not None:
                cache_keys_collector.append(cache_key)

            return content, timestamp
        statistic_data["llm_call"] += 1

        kwargs = {}
        if safe_history_messages:
            kwargs["history_messages"] = safe_history_messages
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        res: str = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )

        res = remove_think_tags(res)
        current_timestamp = int(time.time())

        if llm_response_cache.global_config.get("enable_llm_cache_for_entity_extract"):
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type=cache_type,
                    chunk_id=chunk_id,
                ),
            )

            if cache_keys_collector is not None:
                cache_keys_collector.append(cache_key)

        return res, current_timestamp

    kwargs = {}
    if safe_history_messages:
        kwargs["history_messages"] = safe_history_messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        res = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )
    except Exception as e:
        error_msg = f"[LLM func] {str(e)}"
        raise type(e)(error_msg) from e

    current_timestamp = int(time.time())
    return remove_think_tags(res), current_timestamp
