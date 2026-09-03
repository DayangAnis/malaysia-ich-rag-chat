# Malaysia ICH CPG RAG Chat

A beginner-friendly Streamlit chat interface for the final dissertation RAG pipeline.

## Final pipeline

Recursive chunking → BioLORD-2023-C → Hybrid BM25 + dense retrieval → Reciprocal Rank Fusion (k=60) → Top 5 chunks → `openai/gpt-oss-120b`.

## Before deployment

Add:

- `data/recursive_chunks.csv`
- all final Chroma files under `vector_db/stage4_safe/`

The Chroma collection must be named:

`stage4_recursive_biolord_dense_v1`

The chunk CSV must contain 350 chunks with `chunk_id` and `text` columns.

## Streamlit secret

In Streamlit Community Cloud, add:

```toml
GROQ_API_KEY = "YOUR_REAL_GROQ_API_KEY"
```

Never commit the real API key to GitHub.

## Important

This is a research prototype and is not clinically validated for patient care.
