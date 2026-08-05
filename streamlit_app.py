"""
Phase 8: Streamlit chat frontend for Haqqi.
Talks to the FastAPI backend (api.py) over HTTP - does not touch the
database or LLM directly, exactly as required by the assignment.

Run this in a SEPARATE terminal, while api.py (uvicorn) is already running.

Install first (one-time):
    pip install streamlit requests

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="حقّي - المساعد القانوني", page_icon="⚖️", layout="centered")

# --- Right-to-left styling for Arabic ---
st.markdown("""
<style>
    .stApp { direction: rtl; text-align: right; }
    .stChatMessage { direction: rtl; text-align: right; }
    .stChatInput textarea { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ حقّي - مساعدك القانوني")
st.caption("مساعد ذكي يجيب على أسئلتك حول نظام العمل السعودي ونظام التأمينات الاجتماعية، بالاستناد إلى النصوص الرسمية.")

# --- Session setup: each browser tab gets its own persistent session_id ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": ..., "sources": [...]}

# --- Sidebar: new conversation button + backend health check ---
with st.sidebar:
    st.header("الإعدادات")
    if st.button("🗑️ محادثة جديدة"):
        try:
            requests.delete(f"{API_URL}/chat/{st.session_state.session_id}", timeout=5)
        except requests.exceptions.RequestException:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"✅ متصل بالخادم ({health.get('vectors_loaded', '?')} مادة قانونية)")
    except requests.exceptions.RequestException:
        st.error("❌ لا يمكن الاتصال بالخادم. تأكد أن api.py يعمل (uvicorn api:app).")

# --- Display past messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            law_names = {"labor_law": "نظام العمل", "social_insurance_law": "نظام التأمينات الاجتماعية"}
            citations = ", ".join(
                f"{law_names.get(s['law'], s['law'])} - المادة {s['article_number']}"
                + (" ⚠️ ملغاة" if s["status"] == "repealed" else "")
                for s in msg["sources"]
            )
            st.caption(f"📚 المصادر: {citations}")

# --- Chat input ---
user_input = st.chat_input("اكتب سؤالك القانوني هنا...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في النصوص القانونية..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": user_input},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])

                st.write(answer)
                if sources:
                    law_names = {"labor_law": "نظام العمل", "social_insurance_law": "نظام التأمينات الاجتماعية"}
                    citations = ", ".join(
                        f"{law_names.get(s['law'], s['law'])} - المادة {s['article_number']}"
                        + (" ⚠️ ملغاة" if s["status"] == "repealed" else "")
                        for s in sources
                    )
                    st.caption(f"📚 المصادر: {citations}")

                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

            except requests.exceptions.RequestException as e:
                error_msg = "عذراً، حدث خطأ في الاتصال بالخادم. تأكد أن الخادم يعمل بشكل صحيح."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})