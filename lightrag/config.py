from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from lightrag.constants import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_MAX_ASYNC,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    DEFAULT_SUMMARY_LANGUAGE,
)


@dataclass
class PipelineConfig:
    """Typed configuration threaded through extraction, merge, and query stages.

    Replaces the untyped ``global_config: dict`` that was previously passed by
    spreading the entire LightRAG dataclass via ``asdict(self)``.  Every field
    here corresponds to a concrete key that operate.py used to pull from that
    dict, so callers no longer need to know secret key names.
    """

    # --- services ---------------------------------------------------------
    llm_model_func: Callable
    tokenizer: Any  # lightrag.utils.Tokenizer

    # --- extraction -------------------------------------------------------
    entity_extract_max_gleaning: int
    max_extract_input_tokens: int

    # --- merge / description summarisation --------------------------------
    summary_context_size: int
    summary_max_tokens: int
    summary_length_recommended: int
    force_llm_summary_on_merge: int

    # --- source-ID bookkeeping --------------------------------------------
    max_source_ids_per_entity: int
    max_source_ids_per_relation: int
    source_ids_limit_method: str = DEFAULT_SOURCE_IDS_LIMIT_METHOD
    max_file_paths: int = DEFAULT_MAX_FILE_PATHS
    file_path_more_placeholder: str = DEFAULT_FILE_PATH_MORE_PLACEHOLDER

    # --- query token budgets ----------------------------------------------
    max_entity_tokens: int = DEFAULT_MAX_ENTITY_TOKENS
    max_relation_tokens: int = DEFAULT_MAX_RELATION_TOKENS
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS

    # --- async concurrency ------------------------------------------------
    llm_model_max_async: int = DEFAULT_MAX_ASYNC

    # --- workspace isolation ----------------------------------------------
    workspace: str = ""

    # --- promoted from addon_params ---------------------------------------
    language: str = DEFAULT_SUMMARY_LANGUAGE
    entity_types: list[str] = field(
        default_factory=lambda: list(DEFAULT_ENTITY_TYPES)
    )

    # --- reranking --------------------------------------------------------
    rerank_model_func: Any | None = None
    min_rerank_score: float = 0.0

    # --- optional ---------------------------------------------------------
    embedding_token_limit: int | None = None
    system_prompt_template: str | None = None
