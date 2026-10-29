# Clinical ICH Guideline RAG — Research Prototype

A research prototype for clinical question answering grounded in Malaysia's **Clinical Practice Guideline on Management of Spontaneous Intracerebral Haemorrhage (2025)**.

The project evaluates retrieval-augmented generation (RAG) against a non-RAG LLM baseline. It is an MSc Data Science research project, **not a medical device, clinical decision-support product, or source of medical advice**.

## Research question

Can retrieving evidence from a specific clinical practice guideline help an LLM produce answers that are more faithful to the guideline and contain fewer unsupported claims?

## Final evaluated configuration

`Recursive chunking → BioLORD-2023-C embeddings → BM25 + dense retrieval → Reciprocal Rank Fusion (RRF, k=60) → top 5 passages → Groq-hosted openai/gpt-oss-120b`

- **Source:** Malaysia's CPG on Management of Spontaneous Intracerebral Haemorrhage, 2025.
- **Evaluation set:** 29 guideline-derived clinical questions.
- **Retrieval metrics:** Recall@5 and MRR@5.
- **Answer evaluation:** 58 responses (29 RAG and 29 baseline), scored by one medically qualified evaluator using a 7-point rubric.
- **Interface:** Streamlit.

## Main results

| Outcome | RAG | Non-RAG baseline |
|---|---:|---:|
| Mean answer-quality score (0–7) | **5.552** | 1.414 |
| Responses with unsupported claims | **4/29 (13.8%)** | 28/29 (96.6%) |

The RAG answer-quality score was higher on 24 of the 29 paired questions. These are results from a small, single-guideline evaluation and should not be interpreted as evidence of clinical safety or generalisability.

The uploaded experiment log recorded final-stage retrieval results around Recall@5 = 0.607 and MRR@5 = 0.868 for hybrid retrieval. The thesis abstract reports mean Recall@5 = 0.600 and mean MRR@5 = 0.833 across its selected final configuration. **Please reconcile which aggregation/analysis version you intend to cite before publication**; the values differ between supplied artefacts. The results file in `evaluation/summary.json` records the figures reported in the final analysis workbook where available.

## Repository contents

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── local/                  # Local-only corpus file; intentionally not committed
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── limitations.md
│   └── data_and_copyright.md
├── evaluation/
│   ├── summary.json
│   └── README.md
├── figures/
│   └── architecture.svg
├── notebooks/                  # Original research notebooks; outputs cleared
├── tests/
│   └── test_rrf.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Run locally

### 1. Create an environment

Python 3.11 is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Provide the source corpus

For copyright and redistribution safety, the guideline PDF, extracted full text, derived chunk corpus, and prebuilt vector databases are **not included** in this public-ready package. Before running the app, ensure you have permission to use the source and place an authorized, locally generated CSV at:

`data/local/recursive_chunks.csv`

It must have these columns:

- `chunk_id`: stable identifier for the passage
- `text`: passage text

The app builds embeddings from this file on first launch. This can take time and requires internet access to download the Hugging Face embedding model.

### 3. Configure the model API key

Copy `.env.example` to `.env` for local use and set `GROQ_API_KEY`. Never commit `.env` or paste API keys into source code. In Streamlit Community Cloud, set the key through **App → Settings → Secrets**.

### 4. Start the app

```bash
streamlit run app/streamlit_app.py
```

The first run downloads `FremyCompany/BioLORD-2023-C` and creates a local vector index. The app uses the configured Groq model `openai/gpt-oss-120b`; availability may change, so check the provider's current model list if generation fails.

## Reproducibility and research artefacts

The `notebooks/` folder contains the supplied research notebooks with cell outputs removed to avoid publishing potentially sensitive or copyrighted extracted text in outputs. The notebooks may require local input files and package versions from the original research environment. They are research records, not guaranteed one-click pipelines.

Aggregated final answer-evaluation metrics are in `evaluation/summary.json`. Raw response sheets, unblinding keys, full extracted guideline text, and vector databases have been intentionally excluded. Do not publish those files without checking privacy, copyright, ethics, and supervisor/university requirements.

## Safety notice

This project is for research and demonstration only. It must not be used to diagnose, triage, or treat patients. The retrieval layer can return incomplete or irrelevant passages, and the model can still produce incorrect, incomplete, or unsupported answers. Always consult the original guideline and qualified clinicians for clinical decisions.

## Feedback

Constructive feedback on retrieval design, evaluation methodology, reproducibility, and failure analysis is welcome. Please open an issue with a clear description and reproducible details; do not submit patient-identifiable information.

## Citation

If you use or discuss the project, cite the associated MSc Data Science report when it is publicly available. Add the final bibliographic citation here after university approval/publication.
