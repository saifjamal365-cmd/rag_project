"""
Phase 5 Evaluation: measure Recall@10 for semantic-only vs hybrid search.
Run this in the same folder as law_chunks.json, law_vector_db/, and the test file.

Usage:
    python eval_recall.py test_questions.json        (easy set - legal wording)
    python eval_recall.py test_questions_hard.json    (hard set - casual, realistic wording)
"""

import json, time, sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

TEST_FILE = sys.argv[1] if len(sys.argv) > 1 else "test_questions.json"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# 1. Load chunks + build retrievers (same as hybrid_search.py)
with open("law_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

documents = [
    Document(
        page_content=c["text"],
        metadata={"law": c["law"], "article_number": c.get("article_number"), "chunk_id": c["chunk_id"]}
    )
    for c in chunks
]

log("Loading embedding model + vector database...")
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
semantic_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

def hybrid_search(query, k=10, rrf_k=60, weights=(0.8, 0.2)):
    semantic_results = semantic_retriever.invoke(query)
    keyword_results = bm25_retriever.invoke(query)
    scores, docs_by_id = {}, {}
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

# 2. Load the test set
with open(TEST_FILE, encoding="utf-8") as f:
    test_questions = json.load(f)

log(f"Testing with: {TEST_FILE}")
log(f"Loaded {len(test_questions)} test questions\n")

# 3. Run both retrieval methods on every question, check if correct article appears
#    within the top 1 / top 3 / top 10 results (three difficulty levels of the same metric)
K_LEVELS = [1, 3, 10]
hits = {("semantic", k): 0 for k in K_LEVELS}
hits.update({("hybrid", k): 0 for k in K_LEVELS})
results_log = []

for q in test_questions:
    semantic_results = semantic_retriever.invoke(q["question"])  # already fetches top 10
    hybrid_results = hybrid_search(q["question"], k=10)

    row = {"id": q["id"], "question": q["question"], "expected": q["expected_article"]}
    for k in K_LEVELS:
        sem_found = any(
            r.metadata.get("law") == q["law"] and r.metadata.get("article_number") == q["expected_article"]
            for r in semantic_results[:k]
        )
        hyb_found = any(
            r.metadata.get("law") == q["law"] and r.metadata.get("article_number") == q["expected_article"]
            for r in hybrid_results[:k]
        )
        hits[("semantic", k)] += sem_found
        hits[("hybrid", k)] += hyb_found
        row[f"sem@{k}"] = sem_found
        row[f"hyb@{k}"] = hyb_found
    results_log.append(row)

# 4. Print detailed + summary results
header = f"{'ID':<4}{'Expected':<10}" + "".join(f"{'S@'+str(k):<7}{'H@'+str(k):<7}" for k in K_LEVELS) + "Question"
print(header)
for row in results_log:
    line = f"{row['id']:<4}{row['expected']:<10}"
    for k in K_LEVELS:
        line += f"{'Y' if row[f'sem@{k}'] else '.':<7}{'Y' if row[f'hyb@{k}'] else '.':<7}"
    line += row["question"][:35]
    print(line)

n = len(test_questions)
print(f"\n=== RECALL RESULTS ({TEST_FILE}) ===")
print(f"{'Metric':<12}{'Semantic':<15}{'Hybrid':<15}")
for k in K_LEVELS:
    s = hits[('semantic', k)] / n * 100
    h = hits[('hybrid', k)] / n * 100
    print(f"Recall@{k:<5}{s:>6.1f}%        {h:>6.1f}%")