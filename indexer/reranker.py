"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3."""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


class Reranker:
    """Score query-document pairs and return top-k by relevance."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.RERANKER_MODEL
            ).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize reranker (model: {config.RERANKER_MODEL}): {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Return top-k candidates sorted by rerank_score (desc). Defaults to config.FINAL_TOP_K."""
        if not candidates:
            return []
        # validate required key
        if missing := [i for i, c in enumerate(candidates) if "text" not in c]:
            raise ValueError(f"candidates[{missing}] missing required 'text' key")

        effective_top_k = top_k if top_k is not None else config.FINAL_TOP_K

        pairs = [[query, c["text"]] for c in candidates]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=config.RERANKER_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits
            scores = logits.view(-1).float().tolist()
            del inputs, logits
        if self.device != "cpu":
            torch.cuda.empty_cache()

        scored = [
            {**c, "rerank_score": score}
            for c, score in zip(candidates, scores)
        ]
        return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:effective_top_k]
