import time
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st
import torch
from groq import Groq
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================================
# 1. PAGE SETTINGS
# ==========================================================
st.set_page_config(
    page_title="Malaysia ICH CPG RAG Chat",
    page_icon="🧠",
    layout="wide",
)


# ==========================================================
# 2. FINAL RAG SETTINGS FROM THE DISSERTATION
# ==========================================================
CHUNKING_METHOD = "recursive"
EMBEDDING_MODEL_NAME = "FremyCompany/BioLORD-2023-C"
TOP_K = 5
RRF_CONSTANT = 60
GENERATOR_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.0
MAX_COMPLETION_TOKENS = 500
COLLECTION_NAME = "stage4_recursive_biolord_dense_v1"
EXPECTED_CHUNK_COUNT = 350

BASE_DIR = Path(__file__).resolve().parent
CHUNK_PATH = BASE_DIR / "data" / "recursive_chunks.csv"
VECTOR_DIR = BASE_DIR / "vector_db" / "stage4_safe"


# ==========================================================
# 3. EXACT FINAL RAG PROMPTS
# ==========================================================
RAG_SYSTEM_PROMPT = """
You are a clinical assistant answering questions about spontaneous intracerebral haemorrhage.
Use only the provided guideline excerpts.
Do not add information from outside the excerpts.
If the answer is not supported by the excerpts, state that the guideline excerpts provided are insufficient.
""".strip()


def build_rag_user_prompt(question, context):
    return f"""
Clinical question:

{question}

Guideline excerpts:

{context}

Task:

Provide a concise clinical answer based only on the guideline excerpts above.

Include recommendation strength or level of evidence where it is provided in the excerpts.

Do not fabricate recommendations, drug doses, thresholds, or other clinical details that are not present in the excerpts.
""".strip()


# ==========================================================
# 4. LOAD THE FINAL RETRIEVAL SYSTEM ONCE
# ==========================================================
@st.cache_resource(show_spinner="Loading the final RAG system...")
def load_retrieval_stack():
    if not CHUNK_PATH.is_file():
        raise FileNotFoundError(
            "recursive_chunks.csv was not found. Put it inside the data folder."
        )

    if not VECTOR_DIR.is_dir():
        raise FileNotFoundError(
            "The vector_db/stage4_safe folder was not found."
        )

    chunk_df = pd.read_csv(CHUNK_PATH)

    required_columns = {"chunk_id", "text"}
    missing = required_columns - set(chunk_df.columns)
    if missing:
        raise ValueError(
            f"recursive_chunks.csv is missing columns: {sorted(missing)}"
        )

    if len(chunk_df) != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CHUNK_COUNT} chunks, but found {len(chunk_df)}."
        )

    documents = [
        Document(
            page_content=str(row["text"]),
            metadata={"chunk_id": str(row["chunk_id"])},
        )
        for _, row in chunk_df.iterrows()
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collections = chroma_client.list_collections()
    collection_names = [
        c.name if hasattr(c, "name") else str(c) for c in collections
    ]

    if COLLECTION_NAME not in collection_names:
        raise RuntimeError(
            f"Expected Chroma collection '{COLLECTION_NAME}' was not found."
        )

    vector_db = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    if vector_db._collection.count() != len(documents):
        raise RuntimeError(
            "The Chroma database does not match recursive_chunks.csv."
        )

    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = TOP_K

    dense = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    return bm25, dense, device


# ==========================================================
# 5. HYBRID RETRIEVAL + RECIPROCAL RANK FUSION
# ==========================================================
def reciprocal_rank_fusion(bm25_docs, dense_docs, rrf_k=60, top_k=5):
    scores = {}
    lookup = {}

    for rank, doc in enumerate(bm25_docs, start=1):
        chunk_id = str(doc.metadata["chunk_id"])
        lookup[chunk_id] = doc
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rrf_k + rank)

    for rank, doc in enumerate(dense_docs, start=1):
        chunk_id = str(doc.metadata["chunk_id"])
        lookup[chunk_id] = doc
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rrf_k + rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [lookup[cid] for cid in ranked_ids[:top_k]]


def hybrid_retrieve(question, bm25, dense):
    bm25_docs = bm25.invoke(question)
    dense_docs = dense.invoke(question)
    return reciprocal_rank_fusion(
        bm25_docs,
        dense_docs,
        rrf_k=RRF_CONSTANT,
        top_k=TOP_K,
    )


def format_context(docs):
    sections = []
    for rank, doc in enumerate(docs, start=1):
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        sections.append(
            f"[Excerpt {rank} | Chunk {chunk_id}]\n{doc.page_content}"
        )
    return "\n\n".join(sections)


# ==========================================================
# 6. GROQ / GPT-OSS GENERATION
# ==========================================================
@st.cache_resource
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception as exc:
        raise RuntimeError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        ) from exc
    return Groq(api_key=api_key)


def call_groq(client, user_prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            completion = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            latency = time.perf_counter() - start
            answer = completion.choices[0].message.content or ""
            return answer.strip(), latency
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code in {400, 401, 403, 404} or attempt == max_retries - 1:
                raise
            time.sleep(min(2**attempt, 20))


# ==========================================================
# 7. SIMPLE CHAT INTERFACE
# ==========================================================
st.title("🧠 Malaysia ICH CPG RAG Chat")
st.caption("Research prototype grounded in Malaysia's ICH CPG 2025")

st.warning(
    "Research use only. Not clinically validated. Do not enter identifiable "
    "patient information. Always verify against the source guideline."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This chat uses the final dissertation pipeline: "
        "Recursive chunking → BioLORD-2023-C → Hybrid BM25 + Dense → "
        "RRF → Top 5 → GPT-OSS-120B."
    )
    st.info(
        "Each question is retrieved independently. The visible chat history is "
        "for convenience; previous answers are not used as medical context for "
        "the next question."
    )
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


try:
    bm25_retriever, dense_retriever, embedding_device = load_retrieval_stack()
    groq_client = get_groq_client()
except Exception as startup_error:
    st.error("The app could not load its RAG files.")
    st.exception(startup_error)
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info(
        "Ask a question about spontaneous intracerebral haemorrhage, for example: "
        "'What is the recommended acute blood pressure management in ICH?'"
    )


# Re-draw previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("evidence"):
            with st.expander("View retrieved guideline evidence"):
                for item in message["evidence"]:
                    st.markdown(
                        f"**Rank {item['rank']} — Chunk {item['chunk_id']}**"
                    )
                    st.write(item["text"])
                    if item["rank"] < len(message["evidence"]):
                        st.divider()

        if message["role"] == "assistant" and message.get("latency") is not None:
            st.caption(f"Generated in {message['latency']:.2f} seconds")


question = st.chat_input("Ask the Malaysia ICH guideline...")

if question:
    clean_question = question.strip()

    st.session_state.messages.append(
        {"role": "user", "content": clean_question}
    )

    with st.chat_message("user"):
        st.markdown(clean_question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the guideline..."):
                docs = hybrid_retrieve(
                    clean_question,
                    bm25_retriever,
                    dense_retriever,
                )

            if not docs:
                raise RuntimeError("No guideline evidence was retrieved.")

            context = format_context(docs)
            user_prompt = build_rag_user_prompt(clean_question, context)

            with st.spinner("Generating a guideline-grounded answer..."):
                answer, latency = call_groq(groq_client, user_prompt)

            st.markdown(answer)

            evidence = []
            with st.expander("View retrieved guideline evidence"):
                for rank, doc in enumerate(docs, start=1):
                    chunk_id = str(doc.metadata.get("chunk_id", "unknown"))
                    st.markdown(f"**Rank {rank} — Chunk {chunk_id}**")
                    st.write(doc.page_content)
                    if rank < len(docs):
                        st.divider()
                    evidence.append(
                        {
                            "rank": rank,
                            "chunk_id": chunk_id,
                            "text": doc.page_content,
                        }
                    )

            st.caption(f"Generated in {latency:.2f} seconds")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "evidence": evidence,
                    "latency": latency,
                }
            )

        except Exception as request_error:
            error_message = "I could not process this question. Please check the app logs."
            st.error(error_message)
            st.exception(request_error)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )


with st.expander("System configuration"):
    st.markdown(
        f"""
- **Knowledge source:** Malaysia ICH CPG 2025
- **Chunking:** {CHUNKING_METHOD}
- **Embedding:** {EMBEDDING_MODEL_NAME}
- **Embedding device:** {embedding_device}
- **Retrieval:** Hybrid BM25 + Dense
- **Fusion:** Reciprocal Rank Fusion (RRF)
- **RRF constant:** {RRF_CONSTANT}
- **Top-k:** {TOP_K}
- **Generator:** {GENERATOR_MODEL}
- **Temperature:** {TEMPERATURE}
- **Max completion tokens:** {MAX_COMPLETION_TOKENS}
"""
    )
