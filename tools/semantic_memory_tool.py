"""
Semantic Memory Tool — Pure Python TF-IDF Implementation

Provides vector similarity search over memory files (MEMORY.md, USER.md)
without requiring C extensions or external APIs.

Architecture:
    Memory files (MEMORY.md, USER.md)
        → chunk_text() [paragraphs → sentences, ~256 tokens/chunk]
        → TfidfEmbedder [fixed vocab, L2-normalized TF-IDF vectors]
        → SQLite (vectors as JSON blobs, dedup by SHA-256 hash)
        → _search() [cosine similarity via np.matmul]
"""

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ==============================================================================
# Configuration
# ==============================================================================

MEMORY_DIR = Path.home() / ".hermes"
MEMORY_FILES = [
    MEMORY_DIR / "SOUL.md",
    MEMORY_DIR / "USER.md",
]
DB_PATH = MEMORY_DIR / "semantic_memory.db"

# Fixed vocabulary — curated ~200 terms with pre-set IDF weights
# Covers: AI/ML, medical/neurology, developer terms, general context
FIXED_VOCAB = [
    # AI/ML terms
    "model", "llm", "embedding", "rag", "vector", "semantic", "search",
    "retrieval", "context", "prompt", "inference", "training", "fine-tuning",
    "provider", "api", "token", "chunk", "index", "cosine", "similarity",
    # Developer terms
    "docker", "container", "kubernetes", "server", "client", "endpoint",
    "database", "sqlite", "query", "cache", "redis", "memory", "storage",
    "file", "path", "directory", "folder", "script", "python", "javascript",
    "terminal", "cli", "command", "run", "execute", "build", "test", "deploy",
    "config", "environment", "variable", "secret", "key", "auth", "token",
    "http", "https", "request", "response", "json", "xml", "yaml", "toml",
    # Medical/neurology
    "patient", "doctor", "medical", "medicine", "treatment", "diagnosis",
    "sclerosis", "ms", "mri", "scan", "neurologist", "brain", "spinal",
    "therapy", "drug", "medication", "clinical", "trial", "study", "research",
    # Project/team terms
    "project", "team", "member", "collaborator", "contributor", "review",
    "pr", "pull", "request", "issue", "bug", "feature", "task", "sprint",
    "milestone", "roadmap", "priority", "status", "progress", "update",
    # User preferences
    "user", "preference", "setting", "option", "choice", "default", "custom",
    "personal", "private", "profile", "account", "subscription", "plan",
    # Security
    "security", "secure", "encrypt", "decrypt", "password", "credential",
    "permission", "access", "firewall", "vpn", "2fa", "mfa", "oauth",
    # Infrastructure
    "cloud", "aws", "azure", "gcp", "serverless", "function", "lambda",
    "compute", "instance", "vm", "network", "ip", "domain", "dns", "ssl",
    # Communication
    "email", "message", "notification", "alert", "slack", "discord", "telegram",
    "chat", "bot", "webhook", "integration", "connect", "sync", "export", "import",
    # General
    "error", "warning", "info", "debug", "log", "trace", "exception", "crash",
    "performance", "speed", "latency", "throughput", "optimization", "bottleneck",
    "documentation", "docs", "tutorial", "guide", "example", "sample", "test",
    "production", "staging", "development", "dev", "debug", "release", "version",
]

# Pre-set IDF weights for fixed vocabulary (blended with corpus IDF at fit time)
FIXED_IDF = {term: 3.0 for term in FIXED_VOCAB}  # Default: rare terms

# Default IDF for terms not in vocab
DEFAULT_IDF = 3.0

# Threshold for search results
SCORE_THRESHOLD = 0.05
TOP_K_DEFAULT = 5

# ==============================================================================
# Database schema
# ==============================================================================

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS passages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,          -- 'memory', 'user', 'manual'
    source_file TEXT,                   -- filename if applicable
    chunk_hash  TEXT UNIQUE NOT NULL,   -- SHA-256 for dedup
    content     TEXT NOT NULL,          -- original text
    tokens      INTEGER,                -- token count estimate
    vector      TEXT,                   -- JSON-encoded numpy array
    indexed_at  TEXT NOT NULL           -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_source ON passages(source);
CREATE INDEX IF NOT EXISTS idx_hash   ON passages(chunk_hash);
"""


def _get_db() -> sqlite3.Connection:
    """Get a database connection, initializing schema if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SQL_SCHEMA)
    return conn


# ==============================================================================
# Text processing
# ==============================================================================

def chunk_text(text: str, max_tokens: int = 256) -> list[str]:
    """
    Split text into chunks of ~max_tokens, preserving sentence boundaries.
    Uses Unicode-aware regex for proper French/European language support.
    """
    # Split into sentences (Unicode-aware)
    sentence_pattern = r'[\u00C0-\u024F\w][^\.!?]*[\.!?]?\s*'
    sentences = re.findall(sentence_pattern, text, re.UNICODE)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Rough token estimate: ~0.75 words per token
        sent_tokens = len(sent.split())
        
        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(sent)
        current_tokens += sent_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def _compute_hash(text: str) -> str:
    """SHA-256 hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ==============================================================================
# TF-IDF Embedder
# ==============================================================================

class TfidfEmbedder:
    """
    TF-IDF embedder with fixed vocabulary and L2-normalized vectors.
    
    Uses a curated fixed vocabulary (~200 terms) with pre-set IDF weights,
    blended 70/30 with corpus IDF at fit() time. This guarantees stable
    vector dimensions across all operations.
    """
    
    def __init__(self):
        self.vocab: dict[str, int] = {term: i for i, term in enumerate(FIXED_VOCAB)}
        self.vocab_size = len(self.vocab)
        self.idf: np.ndarray = np.ones(self.vocab_size) * DEFAULT_IDF
        self.fitted = False
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using Unicode-aware regex."""
        # Unicode letter pattern: \w + accented chars
        return re.findall(r'[\u00C0-\u024F\w]+', text.lower(), re.UNICODE)
    
    def _text_to_tf(self, tokens: list[str]) -> np.ndarray:
        """Convert tokens to term frequency vector."""
        tf = np.zeros(self.vocab_size)
        for token in tokens:
            if token in self.vocab:
                tf[self.vocab[token]] += 1
        return tf
    
    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        """
        Fit the embedder on a corpus of texts.
        Blends 70% fixed IDF with 30% corpus IDF.
        """
        # Compute corpus IDF
        doc_count = len(texts)
        df = np.zeros(self.vocab_size)  # document frequency
        
        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                if token in self.vocab:
                    df[self.vocab[token]] += 1
        
        # Corpus IDF: log(N / df)
        corpus_idf = np.zeros(self.vocab_size)
        for i, dfi in enumerate(df):
            if dfi > 0:
                corpus_idf[i] = max(1.0, np.log(doc_count / dfi))
        
        # Blend fixed IDF (3.0) with corpus IDF: 70/30
        self.idf = 0.7 * self.idf + 0.3 * corpus_idf
        self.fitted = True
        return self
    
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors, L2-normalized."""
        if not self.fitted:
            self.fit(texts)
        
        vectors = []
        for text in texts:
            tokens = self._tokenize(text)
            tf = self._text_to_tf(tokens)
            
            # TF-IDF
            tfidf = tf * self.idf
            
            # L2 normalize
            norm = np.linalg.norm(tfidf)
            if norm > 0:
                tfidf = tfidf / norm
            else:
                tfidf = tf  # Fallback to raw TF if all IDF weights were zero
            
            vectors.append(tfidf)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(texts)
        return self.transform(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.transform([query])[0]


# ==============================================================================
# Semantic memory functions
# ==============================================================================

def _reindex_all() -> dict[str, Any]:
    """
    Rebuild entire semantic index from MEMORY.md and USER.md files.
    Returns stats about the indexing operation.
    """
    conn = _get_db()
    cursor = conn.cursor()
    
    # Clear existing passages from memory files
    cursor.execute("DELETE FROM passages WHERE source IN ('memory', 'user')")
    
    # Collect all texts
    all_texts = []
    file_sources = []
    
    for memory_file in MEMORY_FILES:
        if memory_file.exists():
            source_name = "memory" if "MEMORY" in memory_file.name else "user"
            text = memory_file.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            for chunk in chunks:
                all_texts.append(chunk)
                file_sources.append((source_name, str(memory_file.name)))
    
    if not all_texts:
        conn.close()
        return {"indexed": 0, "duplicates": 0, "error": "No memory files found"}
    
    # Fit embedder and transform
    embedder = TfidfEmbedder()
    vectors = embedder.fit_transform(all_texts)  # shape: (n_chunks, vocab_size)
    
    # Store in database
    indexed = 0
    duplicates = 0
    now = datetime.now().isoformat()
    
    for i, (text, (source, filename)) in enumerate(zip(all_texts, file_sources)):
        chunk_hash = _compute_hash(text)
        vector_json = json.dumps(vectors[i].tolist())
        tokens_est = len(text.split())
        
        try:
            cursor.execute(
                """INSERT INTO passages (source, source_file, chunk_hash, content, tokens, vector, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (source, filename, chunk_hash, text, tokens_est, vector_json, now)
            )
            indexed += 1
        except sqlite3.IntegrityError:
            duplicates += 1  # Already exists (dedup by hash)
    
    conn.commit()
    conn.close()
    
    return {
        "indexed": indexed,
        "duplicates": duplicates,
        "vocab_size": embedder.vocab_size,
        "n_chunks": len(all_texts),
    }


def _search(query: str, top_k: int = TOP_K_DEFAULT) -> list[dict[str, Any]]:
    """
    Cosine similarity search over indexed passages.
    Returns top_k results above score threshold.
    """
    conn = _get_db()
    cursor = conn.cursor()
    
    # Get all passages
    cursor.execute("SELECT id, source, source_file, content, vector FROM passages")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
    
    # Fit embedder on-the-fly ( lightweight — uses fixed vocab)
    texts = [row[3] for row in rows]
    embedder = TfidfEmbedder()
    embedder.fit(texts)  # For IDF weights
    
    # Embed query
    query_vec = embedder.embed_query(query)  # shape: (vocab_size,)
    
    # Compute similarities
    results = []
    for row in rows:
        passage_id, source, filename, content, vector_json = row
        passage_vec = np.array(json.loads(vector_json))
        
        # Cosine similarity = dot product (vectors are L2-normalized)
        score = float(np.dot(query_vec, passage_vec))
        
        if score >= SCORE_THRESHOLD:
            results.append({
                "id": passage_id,
                "source": source,
                "file": filename,
                "text": content,
                "score": round(score, 4),
            })
    
    # Sort by score descending, take top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _add_passages(texts: list[str], source: str = "manual") -> int:
    """
    Index new text passages (auto-chunked, deduped by hash).
    Returns number of passages indexed.
    """
    conn = _get_db()
    cursor = conn.cursor()
    
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        conn.close()
        return 0
    
    # Fit embedder and transform
    embedder = TfidfEmbedder()
    vectors = embedder.fit_transform(all_chunks)
    
    indexed = 0
    now = datetime.now().isoformat()
    
    for i, chunk in enumerate(all_chunks):
        chunk_hash = _compute_hash(chunk)
        vector_json = json.dumps(vectors[i].tolist())
        tokens_est = len(chunk.split())
        
        try:
            cursor.execute(
                """INSERT INTO passages (source, chunk_hash, content, tokens, vector, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (source, chunk_hash, chunk, tokens_est, vector_json, now)
            )
            indexed += 1
        except sqlite3.IntegrityError:
            pass  # Deduplicated
    
    conn.commit()
    conn.close()
    return indexed


def _stats() -> dict[str, Any]:
    """Return index statistics."""
    conn = _get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*), SUM(tokens) FROM passages WHERE source = 'memory'")
    memory_count, memory_tokens = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*), SUM(tokens) FROM passages WHERE source = 'user'")
    user_count, user_tokens = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*), SUM(tokens) FROM passages WHERE source = 'manual'")
    manual_count, manual_tokens = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*) FROM passages")
    total_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT indexed_at FROM passages ORDER BY indexed_at DESC LIMIT 1")
    last_indexed = cursor.fetchone()
    
    conn.close()
    
    return {
        "total_passages": total_count or 0,
        "by_source": {
            "memory": {"count": memory_count or 0, "tokens": memory_tokens or 0},
            "user": {"count": user_count or 0, "tokens": user_tokens or 0},
            "manual": {"count": manual_count or 0, "tokens": manual_tokens or 0},
        },
        "vocab_size": len(FIXED_VOCAB),
        "last_indexed": last_indexed[0] if last_indexed else None,
    }


# ==============================================================================
# Tool schema and handler
# ==============================================================================

SCHEMA = {
    "name": "semantic_memory",
    "description": (
        "Search or index persistent semantic memory. "
        "'search' finds relevant passages by cosine similarity. "
        "'reindex' rebuilds from MEMORY.md/USER.md. "
        "'add' indexes new text. 'stats' shows index info."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "add", "reindex", "stats"],
                "description": "Action to perform",
            },
            "query": {
                "type": "string",
                "description": "Search query (for 'search' action)",
            },
            "text": {
                "type": "string",
                "description": "Text to index (for 'add' action)",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "description": "Max results to return (for 'search' action)",
            },
        },
        "required": ["action"],
    },
}


def _handle_semantic_memory(args: dict, **kwargs) -> dict:
    """
    Tool handler for semantic_memory.
    All Hermes tool handlers use (args, **kw) signature.
    """
    action = args.get("action")
    
    if action == "search":
        query = args.get("query", "")
        top_k = args.get("top_k", TOP_K_DEFAULT)
        if not query:
            return {"error": "query is required for search action"}
        results = _search(query, top_k=top_k)
        return {"results": results, "query": query, "count": len(results)}
    
    elif action == "add":
        text = args.get("text", "")
        if not text:
            return {"error": "text is required for add action"}
        indexed = _add_passages([text], source="manual")
        return {"indexed": indexed}
    
    elif action == "reindex":
        result = _reindex_all()
        return result
    
    elif action == "stats":
        return _stats()
    
    else:
        return {"error": f"Unknown action: {action}"}


# Alias for Hermes tool discovery
handle_function_call = _handle_semantic_memory
