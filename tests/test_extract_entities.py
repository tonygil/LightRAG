"""Tests for entity extraction gleaning token limit guard."""

from unittest.mock import AsyncMock, patch

import pytest

from lightrag.config import PipelineConfig
from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for testing."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _make_config(
    max_extract_input_tokens: int = 20480,
    entity_extract_max_gleaning: int = 1,
) -> PipelineConfig:
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    llm_func = AsyncMock(return_value="")
    return PipelineConfig(
        llm_model_func=llm_func,
        tokenizer=tokenizer,
        entity_extract_max_gleaning=entity_extract_max_gleaning,
        max_extract_input_tokens=max_extract_input_tokens,
        summary_context_size=12000,
        summary_max_tokens=1200,
        summary_length_recommended=600,
        force_llm_summary_on_merge=8,
        max_source_ids_per_entity=300,
        max_source_ids_per_relation=300,
        llm_model_max_async=1,
    )


# Minimal valid extraction result that _process_extraction_result can parse
_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)


def _make_chunks(content: str = "Test content.") -> dict[str, dict]:
    return {
        "chunk-001": {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
        }
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_skipped_when_tokens_exceed_limit():
    """Gleaning should be skipped when estimated tokens exceed max_extract_input_tokens."""
    from lightrag.operate import extract_entities

    config = _make_config(max_extract_input_tokens=10, entity_extract_max_gleaning=1)
    config.llm_model_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.extraction.logger") as mock_logger:
        await extract_entities(chunks=_make_chunks(), config=config)

    # LLM should be called exactly once (initial extraction only, no gleaning)
    assert config.llm_model_func.await_count == 1
    # Warning should be logged about skipping gleaning
    mock_logger.warning.assert_called_once()
    warning_msg = mock_logger.warning.call_args[0][0]
    assert "Gleaning stopped" in warning_msg
    assert "exceeded limit" in warning_msg


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_proceeds_when_tokens_within_limit():
    """Gleaning should proceed when estimated tokens are within max_extract_input_tokens."""
    from lightrag.operate import extract_entities

    config = _make_config(max_extract_input_tokens=999999, entity_extract_max_gleaning=1)
    config.llm_model_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.extraction.logger"):
        await extract_entities(chunks=_make_chunks(), config=config)

    # LLM should be called twice (initial extraction + gleaning)
    assert config.llm_model_func.await_count == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_gleaning_when_max_gleaning_zero():
    """No gleaning when entity_extract_max_gleaning is 0, regardless of token limit."""
    from lightrag.operate import extract_entities

    config = _make_config(max_extract_input_tokens=999999, entity_extract_max_gleaning=0)
    config.llm_model_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.extraction.logger"):
        await extract_entities(chunks=_make_chunks(), config=config)

    # LLM should be called exactly once (initial extraction only)
    assert config.llm_model_func.await_count == 1
