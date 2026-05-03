"""Text, JSON, and source-ID utilities.

Covers:
- JSON sanitization and file I/O (load_json, write_json, SanitizingJSONEncoder)
- Text sanitization and normalization for LLM extraction
- String helpers used by extraction/merge/query pipelines
- ID and hash generation (compute_mdhash_id, make_relation_vdb_ids, ...)
- Source-ID set operations (merge, subtract, apply_limit, ...)
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import tempfile
from hashlib import md5
from typing import Any, Collection, Iterable, Sequence

from lightrag.constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    VALID_SOURCE_IDS_LIMIT_METHODS,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
)

logger = logging.getLogger("lightrag")

_SURROGATE_PATTERN = re.compile(r"[\uD800-\uDFFF￾￿]")
_CONTROL_CHAR_PATTERN_ALL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


# ---------------------------------------------------------------------------
# Hashing / ID generation
# ---------------------------------------------------------------------------


def compute_args_hash(*args: Any) -> str:
    """Compute a hash for the given arguments with safe Unicode handling."""
    args_str = "".join([str(arg) for arg in args])
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute a unique ID for a given content string."""
    return prefix + compute_args_hash(content)


def make_relation_vdb_ids(src_entity: str, tgt_entity: str) -> list[str]:
    """Return candidate relation VDB IDs for an undirected edge.

    The normalized ID is returned first for all new writes. The reverse-order ID is
    kept as a compatibility fallback for historical custom-KG imports that hashed
    the relation using the original endpoint order.
    """
    normalized_src, normalized_tgt = sorted((src_entity, tgt_entity))
    relation_ids = [compute_mdhash_id(normalized_src + normalized_tgt, prefix="rel-")]
    reverse_relation_id = compute_mdhash_id(
        normalized_tgt + normalized_src, prefix="rel-"
    )
    if reverse_relation_id not in relation_ids:
        relation_ids.append(reverse_relation_id)
    return relation_ids


# ---------------------------------------------------------------------------
# Source-ID set operations
# ---------------------------------------------------------------------------


def normalize_source_ids_limit_method(method: str | None) -> str:
    """Normalize the source ID limiting strategy and fall back to default when invalid."""
    if not method:
        return DEFAULT_SOURCE_IDS_LIMIT_METHOD

    normalized = method.upper()
    if normalized not in VALID_SOURCE_IDS_LIMIT_METHODS:
        logger.warning(
            "Unknown SOURCE_IDS_LIMIT_METHOD '%s', falling back to %s",
            method,
            DEFAULT_SOURCE_IDS_LIMIT_METHOD,
        )
        return DEFAULT_SOURCE_IDS_LIMIT_METHOD

    return normalized


def merge_source_ids(
    existing_ids: Iterable[str] | None, new_ids: Iterable[str] | None
) -> list[str]:
    """Merge two iterables of source IDs while preserving order and removing duplicates."""
    merged: list[str] = []
    seen: set[str] = set()

    for sequence in (existing_ids, new_ids):
        if not sequence:
            continue
        for source_id in sequence:
            if not source_id:
                continue
            if source_id not in seen:
                seen.add(source_id)
                merged.append(source_id)

    return merged


def apply_source_ids_limit(
    source_ids: Sequence[str],
    limit: int,
    method: str,
    *,
    identifier: str | None = None,
) -> list[str]:
    """Apply a limit strategy to a sequence of source IDs."""
    if limit <= 0:
        return []

    source_ids_list = list(source_ids)
    if len(source_ids_list) <= limit:
        return source_ids_list

    normalized_method = normalize_source_ids_limit_method(method)

    if normalized_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
        truncated = source_ids_list[-limit:]
    else:  # IGNORE_NEW
        truncated = source_ids_list[:limit]

    if identifier and len(truncated) < len(source_ids_list):
        logger.debug(
            "Source_id truncated: %s | %s keeping %s of %s entries",
            identifier,
            normalized_method,
            len(truncated),
            len(source_ids_list),
        )

    return truncated


def compute_incremental_chunk_ids(
    existing_full_chunk_ids: list[str],
    old_chunk_ids: list[str],
    new_chunk_ids: list[str],
) -> list[str]:
    """Compute incrementally updated chunk IDs based on changes.

    Applies delta changes (additions and removals) to an existing list of chunk IDs
    while maintaining order and ensuring deduplication. Delta additions from
    new_chunk_ids are placed at the end.

    Example:
        >>> existing = ['chunk-1', 'chunk-2', 'chunk-3']
        >>> old = ['chunk-1', 'chunk-2']
        >>> new = ['chunk-2', 'chunk-4']
        >>> compute_incremental_chunk_ids(existing, old, new)
        ['chunk-3', 'chunk-2', 'chunk-4']
    """
    chunks_to_remove = set(old_chunk_ids) - set(new_chunk_ids)
    chunks_to_add = set(new_chunk_ids) - set(old_chunk_ids)

    updated_chunk_ids = [
        cid for cid in existing_full_chunk_ids if cid not in chunks_to_remove
    ]

    for cid in new_chunk_ids:
        if cid in chunks_to_add and cid not in updated_chunk_ids:
            updated_chunk_ids.append(cid)

    return updated_chunk_ids


def subtract_source_ids(
    source_ids: Iterable[str],
    ids_to_remove: Collection[str],
) -> list[str]:
    """Remove a collection of IDs from an ordered iterable while preserving order."""
    removal_set = set(ids_to_remove)
    if not removal_set:
        return [source_id for source_id in source_ids if source_id]

    return [
        source_id
        for source_id in source_ids
        if source_id and source_id not in removal_set
    ]


def make_relation_chunk_key(src: str, tgt: str) -> str:
    """Create a deterministic storage key for relation chunk tracking."""
    return GRAPH_FIELD_SEP.join(sorted((src, tgt)))


def parse_relation_chunk_key(key: str) -> tuple[str, str]:
    """Parse a relation chunk storage key back into its entity pair."""
    parts = key.split(GRAPH_FIELD_SEP)
    if len(parts) != 2:
        raise ValueError(f"Invalid relation chunk key: {key}")
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# JSON file I/O
# ---------------------------------------------------------------------------


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8-sig") as f:
        return json.load(f)


def _sanitize_string_for_json(text: str) -> str:
    """Remove characters that cannot be encoded in UTF-8 for JSON serialization.

    Fast detection path for clean strings (99% of cases) with efficient removal
    for dirty strings.
    """
    if not text:
        return text

    if not _SURROGATE_PATTERN.search(text):
        return text  # Zero-copy for clean strings - most common case

    return _SURROGATE_PATTERN.sub("", text)


class SanitizingJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that sanitizes data during serialization.

    Cleans strings during encoding without creating a full copy of the data structure,
    making it memory-efficient for large datasets.
    """

    def encode(self, o):
        if isinstance(o, str):
            return json.encoder.encode_basestring(_sanitize_string_for_json(o))
        return super().encode(o)

    def iterencode(self, o, _one_shot=False):
        sanitized = self._sanitize_for_encoding(o)
        for chunk in super().iterencode(sanitized, _one_shot):
            yield chunk

    def _sanitize_for_encoding(self, obj):
        if isinstance(obj, str):
            return _sanitize_string_for_json(obj)
        elif isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                clean_k = _sanitize_string_for_json(k) if isinstance(k, str) else k
                clean_v = self._sanitize_for_encoding(v)
                new_dict[clean_k] = clean_v
            return new_dict
        elif isinstance(obj, (list, tuple)):
            cleaned = [self._sanitize_for_encoding(item) for item in obj]
            return type(obj)(cleaned) if isinstance(obj, tuple) else cleaned
        else:
            return obj


def write_json(json_obj, file_name):
    """Write JSON data to file with optimized sanitization strategy.

    Two-stage approach:
    1. Fast path: Try direct serialization (works for clean data ~99% of time)
    2. Slow path: Use custom encoder that sanitizes during serialization

    Returns:
        bool: True if sanitization was applied (caller should reload data),
              False if direct write succeeded (no reload needed)
    """
    dir_name = os.path.dirname(os.path.abspath(file_name))
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, file_name)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return False

    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        logger.debug(f"Direct JSON write failed, using sanitizing encoder: {e}")

    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False, cls=SanitizingJSONEncoder)
        os.replace(tmp_path, file_name)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info(f"JSON sanitization applied during write: {file_name}")
    return True


# ---------------------------------------------------------------------------
# Text sanitization and normalization
# ---------------------------------------------------------------------------


def sanitize_text_for_encoding(text: str, replacement_char: str = "") -> str:
    """Sanitize text to ensure safe UTF-8 encoding.

    Handles surrogates, invalid Unicode sequences, control characters, HTML escapes,
    and whitespace trimming.
    """
    if not text:
        return text

    text = text.strip()
    if not text:
        return text

    text = html.unescape(text)
    text = _SURROGATE_PATTERN.sub(replacement_char, text)
    text = _CONTROL_CHAR_PATTERN_ALL.sub(replacement_char, text)

    return text.strip()


def normalize_extracted_info(name: str, remove_inner_quotes=False) -> str:
    """Normalize entity/relation names and descriptions.

    Rules applied:
    - Clean HTML paragraph/line-break tags
    - Convert full-width CJK letters/numbers/symbols to half-width
    - Remove spaces between Chinese characters (and between CJK and Latin)
    - Replace Chinese parentheses/dash with ASCII equivalents
    - Remove outer quotation marks (English and Chinese)
    - Filter short numeric-only text (length < 3 or mixed dots < 6)
    - Optionally remove inner Chinese quotes and non-breaking spaces
    """
    # Clean HTML tags - remove paragraph and line break tags
    name = re.sub(r"</p\s*>|<p\s*>|<p/>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*>|<br/>", "", name, flags=re.IGNORECASE)

    # Chinese full-width letters to half-width (A-Z, a-z)
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )

    # Chinese full-width numbers to half-width
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    # Chinese full-width symbols to half-width
    name = name.replace("－", "-")  # Chinese minus
    name = name.replace("＋", "+")  # Chinese plus
    name = name.replace("／", "/")  # Chinese slash
    name = name.replace("＊", "*")  # Chinese asterisk

    # Replace Chinese parentheses with English parentheses
    name = name.replace("（", "(").replace("）", ")")

    # Replace Chinese dash with English dash (additional patterns)
    name = name.replace("—", "-").replace("－", "-")

    # Chinese full-width space to regular space (after other replacements)
    name = name.replace("　", " ")

    # Use regex to remove spaces between Chinese characters
    name = re.sub(r"(?<=[一-龥])\s+(?=[一-龥])", "", name)

    # Remove spaces between Chinese and English/numbers/symbols
    name = re.sub(
        r"(?<=[一-龥])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[一-龥])", "", name
    )

    # Remove outer quotes
    if len(name) >= 2:
        # Handle double quotes
        if name.startswith('"') and name.endswith('"'):
            inner_content = name[1:-1]
            if '"' not in inner_content:  # No double quotes inside
                name = inner_content

        # Handle single quotes
        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content:  # No single quotes inside
                name = inner_content

        # Handle Chinese-style double quotes
        if name.startswith("“") and name.endswith("”"):
            inner_content = name[1:-1]
            if "“" not in inner_content and "”" not in inner_content:
                name = inner_content
        if name.startswith("‘") and name.endswith("’"):
            inner_content = name[1:-1]
            if "‘" not in inner_content and "’" not in inner_content:
                name = inner_content

        # Handle Chinese-style book title mark
        if name.startswith("《") and name.endswith("》"):
            inner_content = name[1:-1]
            if "《" not in inner_content and "》" not in inner_content:
                name = inner_content

    if remove_inner_quotes:
        # Remove Chinese quotes
        name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        # Remove English quotes in and around Chinese
        name = re.sub(r"['\"]+(?=[一-龥])", "", name)
        name = re.sub(r"(?<=[一-龥])['\"]+" , "", name)
        # Convert non-breaking space to regular space
        name = name.replace(" ", " ")
        # Convert narrow non-breaking space to regular space when after non-digits
        name = re.sub(r"(?<=[^\d]) ", " ", name)

    # Remove spaces from the beginning and end of the text
    name = name.strip()

    # Filter out pure numeric content with length < 3
    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""

    def should_filter_by_dots(text):
        return all(c.isdigit() or c == "." for c in text) and "." in text

    if len(name) < 6 and should_filter_by_dots(name):
        return ""

    return name


def sanitize_and_normalize_extracted_text(
    input_text: str, remove_inner_quotes=False
) -> str:
    """Sanitize and normalize extracted entity/relation text."""
    safe_input_text = sanitize_text_for_encoding(input_text)
    if safe_input_text:
        return normalize_extracted_info(
            safe_input_text, remove_inner_quotes=remove_inner_quotes
        )
    return ""


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags and their content from the text.

    Handles two cases:
    1. Complete <think>...</think> blocks anywhere in the text.
    2. Orphaned </think> at the very start (from streaming that begins mid-think-block).
    """
    text = re.sub(r"^((?!<think>).)*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def safe_unicode_decode(content):
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    def replace_unicode_escape(match):
        return chr(int(match.group(1), 16))

    return unicode_escape_pattern.sub(replace_unicode_escape, content.decode("utf-8"))


def get_content_summary(content: str, max_length: int = 250) -> str:
    """Truncate content to max_length with ellipsis if needed."""
    content = content.strip()
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def fix_tuple_delimiter_corruption(
    record: str, delimiter_core: str, tuple_delimiter: str
) -> str:
    """Fix various forms of tuple_delimiter corruption from LLM output."""
    if not record or not delimiter_core or not tuple_delimiter:
        return record

    escaped_delimiter_core = re.escape(delimiter_core)

    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|*?{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )
    record = re.sub(
        rf"<\|\\{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )
    record = re.sub(r"<\|+>", tuple_delimiter, record)
    record = re.sub(
        rf"<.?\|{escaped_delimiter_core}\|.?>",
        tuple_delimiter,
        record,
    )
    record = re.sub(
        rf"<\|?{escaped_delimiter_core}\|?>",
        tuple_delimiter,
        record,
    )
    record = re.sub(
        rf"<[^|]{escaped_delimiter_core}\|>|<\|{escaped_delimiter_core}[^|]>",
        tuple_delimiter,
        record,
    )
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|+(?!>)",
        tuple_delimiter,
        record,
    )
    record = re.sub(
        rf"<\|{escaped_delimiter_core}:(?!>)",
        tuple_delimiter,
        record,
    )
    record = re.sub(
        rf"<\|+{escaped_delimiter_core}>",
        tuple_delimiter,
        record,
    )
    record = re.sub(r"<\|\|(?!>)", tuple_delimiter, record)
    record = re.sub(
        rf"(?<!<)\|{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|>\|",
        tuple_delimiter,
        record,
    )
    record = re.sub(
        rf"\|\|{escaped_delimiter_core}\|\|",
        tuple_delimiter,
        record,
    )

    return record


# ---------------------------------------------------------------------------
# Miscellaneous string helpers
# ---------------------------------------------------------------------------


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))
