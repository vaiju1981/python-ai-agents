"""RAG seams: retriever, embedder, chunk store, and retrieval-augmented agent.

Core owns the protocols; implementations wrap ecosystem libraries
(LlamaIndex, LangChain, chromadb) as optional adapters. A zero-dependency
in-memory vector store is provided for tests and prototyping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse

__all__ = [
    "ChunkStore",
    "Document",
    "Embedder",
    "InMemoryVectorStore",
    "Ingestor",
    "RetrievalAugmentedAgent",
    "RetrievedChunk",
    "Retriever",
]


@dataclass(frozen=True, slots=True)
class Document:
    """A unit of text to be split and indexed."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieved passage with its similarity score and source metadata."""

    text: str
    score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


class Embedder(Protocol):
    """Turns text into a dense vector for semantic similarity."""

    async def embed(self, text: str) -> list[float]:
        ...


class Retriever(Protocol):
    """Retrieves the most relevant chunks for a query, scoped to a tenant."""

    async def retrieve(self, tenant: str, query: str, limit: int) -> list[RetrievedChunk]:
        ...


class ChunkStore(Protocol):
    """Stores embedded chunks for later retrieval."""

    async def add(self, tenant: str, chunks: list[tuple[str, list[float], dict[str, str]]]) -> None:
        ...

    async def search(self, tenant: str, query_vector: list[float], limit: int) -> list[RetrievedChunk]:
        ...


@dataclass
class InMemoryVectorStore:
    """Zero-dependency in-memory vector store using cosine similarity."""

    _store: dict[str, list[tuple[str, list[float], dict[str, str]]]] = field(
        default_factory=dict, init=False
    )

    async def add(
        self, tenant: str, chunks: list[tuple[str, list[float], dict[str, str]]]
    ) -> None:
        self._store.setdefault(tenant, []).extend(chunks)

    async def search(
        self, tenant: str, query_vector: list[float], limit: int
    ) -> list[RetrievedChunk]:
        entries = self._store.get(tenant, [])
        scored = [
            (text, _cosine(query_vector, vec), meta)
            for text, vec, meta in entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(text=text, score=score, metadata=meta)
            for text, score, meta in scored[:limit]
        ]


@dataclass(slots=True)
class Ingestor:
    """Splits documents into chunks, embeds them, and stores them."""

    embedder: Embedder
    store: ChunkStore
    chunk_size: int = 500
    chunk_overlap: int = 50

    async def ingest(self, tenant: str, documents: list[Document]) -> int:
        count = 0
        for doc in documents:
            chunks = _split_text(doc.text, self.chunk_size, self.chunk_overlap)
            stored: list[tuple[str, list[float], dict[str, str]]] = []
            for chunk in chunks:
                vec = await self.embedder.embed(chunk)
                stored.append((chunk, vec, dict(doc.metadata)))
                count += 1
            await self.store.add(tenant, stored)
        return count


@dataclass(slots=True)
class RetrievalAugmentedAgent:
    """Grounds an agent in retrieved context.

    Before delegating, retrieves the top-k chunks for the user's input and
    prepends them to the prompt. When nothing is retrieved, delegates the
    request unchanged.
    """

    delegate: Agent
    retriever: Retriever
    top_k: int = 4

    async def run(self, request: AgentRequest) -> AgentResponse:
        chunks = await self.retriever.retrieve(
            request.context.tenant, request.input, self.top_k
        )
        if not chunks:
            return await self.delegate.run(request)
        context = "\n".join(f"- {c.text}" for c in chunks)
        augmented = (
            "Answer using the retrieved context below. If it does not contain "
            f"the answer, say so rather than guessing.\n\nContext:\n{context}\n\n"
            f"Question: {request.input}"
        )
        return await self.delegate.run(AgentRequest(input=augmented, context=request.context))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
