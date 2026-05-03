"""Tokenization utilities: interface, implementations, and token-counting helpers."""

from __future__ import annotations

from typing import Any, Callable, List, Protocol


class TokenizerInterface(Protocol):
    """Defines the interface for a tokenizer, requiring encode and decode methods."""

    def encode(self, content: str) -> List[int]:
        """Encodes a string into a list of tokens."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of tokens into a string."""
        ...


class Tokenizer:
    """A wrapper around a tokenizer to provide a consistent interface for encoding and decoding."""

    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        self.model_name: str = model_name
        self.tokenizer: TokenizerInterface = tokenizer

    def encode(self, content: str) -> List[int]:
        return self.tokenizer.encode(content)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)


class TiktokenTokenizer(Tokenizer):
    """A Tokenizer implementation using the tiktoken library."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is not installed. Please install it with `pip install tiktoken` or define custom `tokenizer_func`."
            )

        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
            super().__init__(model_name=model_name, tokenizer=tokenizer)
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}.")


def truncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[int]:
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


class TokenTracker:
    """Track token usage for LLM calls."""

    def __init__(self):
        self.reset()

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self)

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def add_usage(self, token_counts):
        """Add token usage from one LLM call.

        Args:
            token_counts: A dictionary containing prompt_tokens, completion_tokens, total_tokens
        """
        self.prompt_tokens += token_counts.get("prompt_tokens", 0)
        self.completion_tokens += token_counts.get("completion_tokens", 0)

        if "total_tokens" in token_counts:
            self.total_tokens += token_counts["total_tokens"]
        else:
            self.total_tokens += token_counts.get(
                "prompt_tokens", 0
            ) + token_counts.get("completion_tokens", 0)

        self.call_count += 1

    def get_usage(self):
        """Get current usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def __str__(self):
        usage = self.get_usage()
        return (
            f"LLM call count: {usage['call_count']}, "
            f"Prompt tokens: {usage['prompt_tokens']}, "
            f"Completion tokens: {usage['completion_tokens']}, "
            f"Total tokens: {usage['total_tokens']}"
        )
