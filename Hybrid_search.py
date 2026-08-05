"""
Phase 5: Hybrid search = semantic (Chroma) + keyword (BM25) combined.
Run this on your own machine, in the same folder as law_chunks.json and law_vector_db/
(the vector database folder created by build_vector_db.py in Phase 4).

Install first (if not already):
    pip install rank_bm25
"""

import json
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# 1. Load chunks (needed to build BM25 index — it doesn't use the embedding model)
with open("law_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

documents = [
    Document(
        page_content=c["text"],
        metadata={
            "law": c["law"],
            "chapter": c.get("chapter") or "",
            "section": c.get("section") or "",
            "article_ordinal": c["article_ordinal"],
            "status": c["status"],
            "chunk_id": c["chunk_id"],
        }
    )
    for c in chunks
]
log(f"Loaded {len(documents)} chunks")

# 2. Load the existing semantic (Chroma) vector store you already built in Phase 4
log("Loading embedding model + existing vector database...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vector_db = Chroma(
    persist_directory="./law_vector_db",
    embedding_function=embedding_model,
    collection_name="saudi_labor_social_law",
)
semantic_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# 3. Build the keyword (BM25) retriever
log("Building BM25 keyword retriever...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 4. Combine both using Reciprocal Rank Fusion (RRF) - a simple, standard way to merge
#    two ranked lists into one, without needing extra library imports that keep moving around.
def hybrid_search(query, k=5, rrf_k=60, weights=(0.5, 0.5)):
    semantic_results = semantic_retriever.invoke(query)
    keyword_results = bm25_retriever.invoke(query)

    scores = {}
    docs_by_id = {}

    for rank, doc in enumerate(semantic_results):
        cid = doc.metadata["chunk_id"]
        docs_by_id[cid] = doc
        scores[cid] = scores.get(cid, 0) + weights[0] * (1 / (rrf_k + rank + 1))

    for rank, doc in enumerate(keyword_results):
        cid = doc.metadata["chunk_id"]
        docs_by_id[cid] = doc
        scores[cid] = scores.get(cid, 0) + weights[1] * (1 / (rrf_k + rank + 1))

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs_by_id[cid] for cid in ranked_ids]

log("Hybrid retriever ready.\n")

# 5. Compare semantic-only vs hybrid on a few test questions
test_queries = [
    "هل يحق لصاحب العمل فصل الموظف دون إشعار؟",           # can employer fire without notice?
    "متى يستحق العامل مكافأة نهاية الخدمة؟",              # end-of-service gratuity
    "المادة 80",                                          # exact article number - good BM25 test
]

for q in test_queries:
    log(f"QUERY: {q}")
    print("  -- Semantic only --")
    for r in semantic_retriever.invoke(q)[:3]:
        print("   ", r.metadata["law"], "| Article:", r.metadata["article_ordinal"], "|", r.page_content[:60])
    print("  -- Hybrid (semantic + keyword, via RRF) --")
    for r in hybrid_search(q, k=3):
        print("   ", r.metadata["law"], "| Article:", r.metadata["article_ordinal"], "|", r.page_content[:60])
    print()