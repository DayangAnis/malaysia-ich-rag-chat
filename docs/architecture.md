# Architecture

```mermaid
flowchart TD
    A[Authorised CPG text] --> B[Preprocessing and recursive chunking]
    B --> C[Passage corpus]
    C --> D[BM25 lexical retrieval]
    C --> E[BioLORD-2023-C dense embeddings]
    Q[User question] --> F[Question tokenisation and embedding]
    F --> D
    F --> E
    D --> G[Reciprocal Rank Fusion, k=60]
    E --> G
    G --> H[Top 5 passages]
    H --> I[LLM constrained by retrieved context]
    I --> J[Answer plus inspectable evidence]
```

The Streamlit implementation included in this repository builds a local lexical index and dense embeddings from `data/local/recursive_chunks.csv` at runtime. It uses RRF to combine the BM25 and dense rankings. This implementation is intended to make the core design inspectable; re-run the original notebooks and compare outputs before claiming exact equivalence to every experimental run.
