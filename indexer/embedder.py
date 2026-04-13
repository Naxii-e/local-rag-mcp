"""Embedding module using intfloat/multilingual-e5-large."""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import config

logger = logging.getLogger(__name__)


def _average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling (recommended for multilingual-e5-large)."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class Embedder:
    """multilingual-e5-large embedder (FP16, mean pooling, L2-normalized)."""

    QUERY_INSTRUCTION: str = "query: "
    DOCUMENT_INSTRUCTION: str = "passage: "

    def __init__(self) -> None:
        self.model_name: str = config.EMBED_MODEL
        self.batch_size: int = config.EMBED_BATCH_SIZE

        # device: EMBED_DEVICE env var or CUDA auto-detect
        embed_device_env = os.environ.get("EMBED_DEVICE")
        if embed_device_env:
            self.device: str = embed_device_env
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if self.device == "cpu":
            logger.warning("CUDA unavailable, running on CPU (slow).")
        else:
            logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))

        logger.info("Loading embedding model: %s", self.model_name)

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model: AutoModel = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        logger.info("Embedding model loaded.")

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts (instruction prefix already applied)."""
        if not texts:
            return []

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        embeddings = _average_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        result = embeddings.cpu().float().tolist()
        del encoded, outputs, embeddings
        if self.device != "cpu":
            torch.cuda.empty_cache()
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of code chunks with DOCUMENT_INSTRUCTION prefix."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_texts = [self.DOCUMENT_INSTRUCTION + t for t in batch]
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            logger.debug(
                "Embedded batch %d/%d (%d texts)",
                i // self.batch_size + 1,
                (len(texts) + self.batch_size - 1) // self.batch_size,
                len(batch),
            )

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with QUERY_INSTRUCTION prefix."""
        prefixed = self.QUERY_INSTRUCTION + query
        results = self._encode_batch([prefixed])
        return results[0]
