"""
Phase 7: FastAPI backend for Haqqi (the legal assistant).
Wraps the retrieval + generation pipeline (Phases 4-6) behind a REST API,
adds conversation memory per session, and logs every query.

Run this in the same folder as law_chunks.json and law_vector_db/.

Install first (one-time):
    pip install fastapi uvicorn

Get a free Groq API key from: https://console.groq.com/keys
Then set it in your terminal before running:
    set GROQ_API_KEY=your_key_here

Run the server:
    set OMP_NUM_THREADS=1
    set MKL_NUM_THREADS=1
    set KMP_DUPLICATE_LIB_OK=TRUE
    set GROQ_API_KEY=your_key_here
    uvicorn api:app --reload
"""

import json, time, os, logging
from datetime import datetime
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()  # reads GROQ_API_KEY (and anything else) from a local .env file automatically

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --- Logging setup (Phase 7 requirement R6: log queries, chunks, latency) ---
logging.basicConfig(
    filename="query_log.jsonl",
    level=logging.INFO,
    format="%(message)s",
)
query_logger = logging.getLogger("queries")

# --- 1. Load chunks + retrievers (same pipeline proven working in Phase 6) ---
log("Loading law_chunks.json...")
with open("law_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

documents = [
    Document(
        page_content=c["text"],
        metadata={
            "law": c["law"], "article_number": c.get("article_number") or 0,
            "article_ordinal": c["article_ordinal"], "status": c["status"], "chunk_id": c["chunk_id"],
        }
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
semantic_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5
log(f"Retrieval ready. {vector_db._collection.count()} vectors loaded.")

def hybrid_search(query, k=5, rrf_k=60, weights=(0.8, 0.2)):
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

# --- 2. Load the LLM (imported here on purpose - avoids native library conflict) ---
log("Importing Groq connector...")
from langchain_groq import ChatGroq
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not set. Run: set GROQ_API_KEY=your_key_here")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
log("LLM ready.")

SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في نظام العمل السعودي ونظام التأمينات الاجتماعية.

القواعد الملزمة:
1. أجب فقط بالاعتماد على النصوص القانونية المرفقة أدناه. لا تستخدم أي معلومة من خارجها.
2. إذا لم تجد إجابة واضحة في النصوص المرفقة، قل بوضوح: "لم أجد إجابة واضحة لهذا السؤال في المواد المتاحة لدي، يُفضل استشارة مختص."
3. اذكر دائماً رقم المادة التي استندت إليها في إجابتك، مثل: (المادة 80 من نظام العمل).
4. إذا كانت المادة المسترجعة "ملغاة" (status: repealed)، نبّه المستخدم صراحة أن هذه المادة لم تعد سارية، ولا تقدمها على أنها القانون الحالي.
5. لا تخترع أي رقم مادة أو نص لم يرد في السياق المرفق.
6. اكتب بلغة عربية واضحة وبسيطة يفهمها أي شخص، وليس بلغة قانونية معقدة.
7. إذا كان هناك سياق من محادثة سابقة مع المستخدم، استخدمه لفهم أسئلة المتابعة، لكن استمر بالاعتماد فقط على النصوص القانونية المرفقة للإجابة.
"""

def build_context(retrieved_docs):
    parts = []
    for doc in retrieved_docs:
        status_note = " [ملاحظة: هذه المادة ملغاة ولم تعد سارية]" if doc.metadata["status"] == "repealed" else ""
        law_name = "نظام العمل" if doc.metadata["law"] == "labor_law" else "نظام التأمينات الاجتماعية"
        parts.append(f"[{law_name} - المادة {doc.metadata['article_number']}]{status_note}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

# --- 3. Conversation memory (simple in-memory store, per session_id) ---
# NOTE: this resets if the server restarts. For a class project this is fine;
# a production version would use a real database (e.g. Redis or Postgres).
conversation_memory = defaultdict(list)  # session_id -> list of {role, content}
MAX_HISTORY_TURNS = 5  # keep last 5 exchanges to avoid the prompt growing forever

# --- 4. FastAPI app ---
app = FastAPI(title="Haqqi Legal Assistant API")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class SourceInfo(BaseModel):
    law: str
    article_number: int
    status: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    latency_ms: int

@app.get("/health")
def health():
    return {"status": "ok", "vectors_loaded": vector_db._collection.count()}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    # Retrieve relevant articles
    retrieved = hybrid_search(request.message, k=5)
    context = build_context(retrieved)

    # Build the message list: system prompt + past conversation (memory) + new question
    history = conversation_memory[request.session_id][-(MAX_HISTORY_TURNS * 2):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"النصوص القانونية ذات الصلة:\n\n{context}\n\n---\n\nسؤال المستخدم: {request.message}"
    })

    response = llm.invoke(messages)
    answer = response.content

    # Update memory (store the plain question, not the retrieved-context version, to keep it light)
    conversation_memory[request.session_id].append({"role": "user", "content": request.message})
    conversation_memory[request.session_id].append({"role": "assistant", "content": answer})

    latency_ms = int((time.time() - start_time) * 1000)

    # Log the query (Phase 7 requirement R6)
    query_logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "session_id": request.session_id,
        "question": request.message,
        "retrieved_chunks": [d.metadata["chunk_id"] for d in retrieved],
        "latency_ms": latency_ms,
    }, ensure_ascii=False))

    return ChatResponse(
        answer=answer,
        sources=[
            SourceInfo(law=d.metadata["law"], article_number=d.metadata["article_number"], status=d.metadata["status"])
            for d in retrieved
        ],
        latency_ms=latency_ms,
    )

@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    conversation_memory.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}