"""
Phase 4: Embed the law chunks and store them in a Chroma vector database.
Run this on your own machine (needs internet access to download the embedding model).

Install first:
    pip install sentence-transformers chromadb langchain langchain-community langchain-huggingface
"""

import json
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

start_time = time.time()

# 1. Load your chunked law data (the file we built in Phase 3)
with open("law_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

log(f"Loaded {len(chunks)} chunks")

# 2. Turn each chunk into a LangChain Document (text + metadata together)
documents = []
for c in chunks:
    doc = Document(
        page_content=c["text"],
        metadata={
            "law": c["law"],
            "chapter": c.get("chapter") or "",
            "section": c.get("section") or "",
            "article_ordinal": c["article_ordinal"],
            "status": c["status"],
            "chunk_id": c["chunk_id"],
            "part": c.get("part", "1/1"),
            "article_number": c.get("article_number") or 0,
        }
    )
    documents.append(doc)

# 3. Load the Arabic-capable embedding model
log("Loading embedding model (downloads ~2GB the first time - be patient)...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
    encode_kwargs={"normalize_embeddings": True},
)
log(f"Model ready. Elapsed so far: {time.time()-start_time:.0f}s")

# 4. Embed everything and store it in Chroma (this may take a few minutes)
log(f"Embedding {len(documents)} chunks and storing in Chroma...")
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./law_vector_db",  # saved to disk here
    collection_name="saudi_labor_social_law",
)

log(f"Done. Vector database saved to ./law_vector_db")
log(f"Total vectors stored: {vector_db._collection.count()}")
log(f"Total time: {time.time()-start_time:.0f}s")

# 5. Quick sanity test — try a real question
test_query = "هل يحق لصاحب العمل فصل الموظف دون إشعار؟"  # "can employer fire without notice?"
results = vector_db.similarity_search(test_query, k=3)
log(f"Test search results for: {test_query}")
for r in results:
    print("-", r.metadata["law"], "| Article:", r.metadata["article_ordinal"], "|", r.page_content[:80])