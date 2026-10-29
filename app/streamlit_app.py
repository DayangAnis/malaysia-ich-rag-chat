"""Streamlit interface for the ICH guideline RAG research prototype."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.retrieval import reciprocal_rank_fusion, tokenize

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "local" / "recursive_chunks.csv"
EMBEDDING_MODEL = "FremyCompany/BioLORD-2023-C"
DEFAULT_GENERATOR = "openai/gpt-oss-120b"
TOP_K = 5
RRF_CONSTANT = 60

SYSTEM_PROMPT = """You are answering a research question using excerpts from a clinical practice guideline.
Use only the supplied excerpts as evidence. Do not add clinical claims that are not supported by them.
If the excerpts do not contain enough information, clearly say that the provided excerpts are insufficient.
Keep the answer concise. Do not present yourself as a doctor or claim to make clinical decisions.
This is a research prototype, not a source of medical advice."""


@st.cache_resource(show_spinner="Loading corpus and embedding model (first run may take a while)…")
def load_retrieval_resources(corpus_path: str, modified_time: float) -> dict[str, Any]:
    """Load corpus, create lexical index and embed passages for dense retrieval."""
    del modified_time  # included in cache key so a changed file triggers reload
    frame = pd.read_csv(corpus_path)
    required = {"chunk_id", "text"}
    if not required.issubset(frame.columns):
        raise ValueError("Corpus CSV must contain 'chunk_id' and 'text' columns.")
    frame = frame.dropna(subset=["text"]).copy()
    frame["text"] = frame["text"].astype(str)
    frame = frame[frame["text"].str.strip().ne("")].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Corpus CSV contains no non-empty passages.")

    tokenized = [tokenize(text) for text in frame["text"]]
    bm25 = BM25Okapi(tokenized)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        frame["text"].tolist(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return {"frame": frame, "bm25": bm25, "model": model, "embeddings": embeddings}


def retrieve(query: str, resources: dict[str, Any], top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Hybrid BM25 + dense retrieval, fused with RRF."""
    frame = resources["frame"]
    query_tokens = tokenize(query)
    lexical_scores = resources["bm25"].get_scores(query_tokens)
    lexical_rank = np.argsort(-lexical_scores)[:top_k].tolist()

    query_embedding = resources["model"].encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    dense_scores = resources["embeddings"] @ query_embedding
    dense_rank = np.argsort(-dense_scores)[:top_k].tolist()

    fused = reciprocal_rank_fusion([lexical_rank, dense_rank], k=RRF_CONSTANT)[:top_k]
    results = []
    for idx, score in fused:
        row = frame.iloc[idx]
        results.append({
            "chunk_id": str(row["chunk_id"]),
            "text": str(row["text"]),
            "rrf_score": float(score),
            "bm25_rank": lexical_rank.index(idx) + 1 if idx in lexical_rank else None,
            "dense_rank": dense_rank.index(idx) + 1 if idx in dense_rank else None,
        })
    return results


def generate_answer(question: str, passages: list[dict[str, Any]], api_key: str, model_name: str) -> str:
    """Generate an answer constrained to the retrieved passages."""
    client = Groq(api_key=api_key)
    context = "\n\n".join(
        f"[Passage {i} | chunk_id={p['chunk_id']}]\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_completion_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nGuideline excerpts:\n{context}"},
        ],
    )
    return response.choices[0].message.content or "The model returned an empty response."


def get_secret(name: str, fallback: str = "") -> str:
    try:
        return str(st.secrets.get(name, fallback))
    except Exception:
        return os.getenv(name, fallback)


def main() -> None:
    load_dotenv(ROOT / ".env")
    st.set_page_config(page_title="ICH Guideline RAG", page_icon="📚", layout="wide")
    st.title("Clinical ICH Guideline RAG")
    st.caption("MSc Data Science research prototype · Retrieval-augmented question answering")

    st.warning(
        "Research demonstration only — not medical advice and not for clinical decision-making. "
        "Answers may be wrong or incomplete. Verify against the original guideline and qualified clinicians."
    )
    st.markdown(
        "This prototype retrieves passages from an authorised local copy of the 2025 Malaysian "
        "ICH clinical practice guideline, then asks an LLM to answer using those passages."
    )

    if not CORPUS_PATH.exists():
        st.error("Local corpus not found.")
        st.markdown(
            "To run this app, place an authorised CSV at "
            "`data/local/recursive_chunks.csv` with columns `chunk_id` and `text`. "
            "The corpus is excluded from this repository pending copyright/redistribution checks. "
            "See `docs/data_and_copyright.md`."
        )
        st.stop()

    api_key = get_secret("GROQ_API_KEY")
    model_name = get_secret("GROQ_MODEL", DEFAULT_GENERATOR)
    with st.sidebar:
        st.header("Configuration")
        st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
        st.write(f"**Generator:** `{model_name}`")
        st.write(f"**Retrieval:** BM25 + dense embeddings → RRF")
        st.write(f"**Passages retrieved:** {TOP_K}")
        st.write(f"**RRF constant:** {RRF_CONSTANT}")
        st.caption("The first launch downloads the embedding model and embeds all passages.")

    question = st.text_area(
        "Enter a question about spontaneous intracerebral haemorrhage",
        placeholder="Ask a question answerable from the source guideline…",
        height=100,
    )
    run = st.button("Retrieve evidence and generate answer", type="primary")
    if run:
        if not question.strip():
            st.info("Enter a question first.")
            st.stop()
        if not api_key:
            st.error("Missing GROQ_API_KEY. Set it in `.env` locally or Streamlit Secrets when deployed.")
            st.stop()
        try:
            with st.spinner("Preparing retrieval resources…"):
                resources = load_retrieval_resources(
                    str(CORPUS_PATH), CORPUS_PATH.stat().st_mtime
                )
            with st.spinner("Retrieving passages…"):
                passages = retrieve(question.strip(), resources)
            if not passages:
                st.warning("No passages were retrieved.")
                st.stop()
            with st.spinner("Generating a guideline-grounded response…"):
                answer = generate_answer(question.strip(), passages, api_key, model_name)
            st.subheader("Generated response")
            st.write(answer)
            st.caption("Generated content is not independently verified. Inspect the evidence below.")
            st.subheader("Retrieved evidence")
            for i, passage in enumerate(passages, start=1):
                with st.expander(
                    f"{i}. Chunk {passage['chunk_id']} · RRF score {passage['rrf_score']:.4f}"
                ):
                    st.caption(
                        f"BM25 rank: {passage['bm25_rank'] or 'not in top 5'} · "
                        f"Dense rank: {passage['dense_rank'] or 'not in top 5'}"
                    )
                    st.write(passage["text"])
        except Exception as exc:
            st.error(f"The request could not be completed: {exc}")
            st.caption("Check corpus format, internet access, installed dependencies, API key, and provider model availability.")


if __name__ == "__main__":
    main()
