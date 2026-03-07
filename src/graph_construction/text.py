from __future__ import annotations

"""Build similarity matrices from *text* list columns using sentence
transformer embeddings and cosine similarity.
"""

import re
from html import unescape
from typing import Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .common import (
    parse_string_list,
    special_print,
)

__all__ = ["build_similarity_matrix"]

#DEFAULT_MODEL = "sentence-transformers/average_word_embeddings_glove.6B.300d"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def _clean_html_tags(text: str) -> str:
    """Remove HTML tags and decode HTML entities from text."""
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text)

    text = unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _clean_text_list(lst: list | None) -> list[str]:
    """Clean a list of strings: remove None/empty elements and HTML tags."""
    if lst is None:
        return []

    cleaned = []
    for item in lst:
        if item is None:
            continue
        if not isinstance(item, str):
            cleaned.append(str(item))
            continue

        clean_item = _clean_html_tags(item)
        if clean_item and clean_item.strip():
            cleaned.append(clean_item)

    return cleaned


def _join_text_list(lst: list[str]) -> str:
    """Join a list of strings into a single space-separated string."""
    cleaned = _clean_text_list(lst)
    return " ".join(cleaned) if cleaned else ""


# ---------------------------------------------------------------------------
# Text embedder class
# ---------------------------------------------------------------------------

class TextEmbedder:
    """Wrapper around SentenceTransformer for text embedding and similarity."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into normalized embeddings."""
        return self.model.encode(texts, normalize_embeddings=True)

    def similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix from embeddings."""
        return self.model.similarity(embeddings, embeddings).numpy()


# ---------------------------------------------------------------------------
# Similarity matrix computation
# ---------------------------------------------------------------------------

def _create_similarity_matrix(
    texts: Sequence[str],
    model_name: str,
    device: str | None,
    verbose: bool,
) -> np.ndarray:
    """Compute similarity matrix from list of text strings."""
    if verbose:
        print(f"Loading model: {model_name}")

    embedder = TextEmbedder(model_name=model_name, device=device)

    if verbose:
        print(f"Encoding {len(texts)} texts...")

    embeddings = embedder.encode(texts)

    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")
        print("Computing similarity matrix...")

    similarity_matrix = embedder.similarity(embeddings)

    return similarity_matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_similarity_matrix(
    df: pd.DataFrame,
    *,
    label_column: str = "churn",
    item_list_column: str,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute similarity matrix from text list column using sentence embeddings.

    Parameters
    ----------
    df
        Input DataFrame – one row per node. ``item_list_column`` must contain
        an *iterable* (list/array) of strings.
    label_column
        Name of the column that holds the node labels (`y`).
    item_list_column
        Column with the list of text strings.
    model_name
        SentenceTransformer model name/path. Default is GloVe-based model.
    device
        Device for model inference ('cuda', 'cpu', or None for auto-detect).
    verbose
        Whether to print progress information.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (similarity_matrix, y) where similarity_matrix is shape [N, N] and
        y is the label vector of shape [N].
    """
    df = df.copy()

    # Parse string representations if needed
    if isinstance(df[item_list_column].iloc[0], str):
        df[item_list_column] = df[item_list_column].apply(parse_string_list)

    # Join each list of strings into a single string per row
    df[item_list_column] = df[item_list_column].apply(_join_text_list)

    texts = df[item_list_column].tolist()

    if verbose:
        special_print(df.head(), "df.head()")

    similarity_matrix = _create_similarity_matrix(
        texts=texts,
        model_name=model_name,
        device=device,
        verbose=verbose,
    )

    if verbose:
        special_print(similarity_matrix.shape, "similarity_matrix.shape")

    # Extract labels
    y = df[label_column].values if label_column in df.columns else None

    return similarity_matrix, y
