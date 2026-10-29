# Methodology overview

## Research design

The project studies retrieval-augmented generation for clinical question answering grounded in Malaysia's 2025 Clinical Practice Guideline (CPG) on spontaneous intracerebral haemorrhage.

## Experimental stages

1. **Chunking:** compare fixed-size, recursive, and semantic chunking while holding other retrieval settings constant as far as practical.
2. **Embedding model:** compare `sentence-transformers/all-mpnet-base-v2` with `FremyCompany/BioLORD-2023-C`.
3. **Retrieval strategy:** compare dense, BM25 lexical, and hybrid retrieval.
4. **Answer evaluation:** compare 29 RAG answers against 29 non-RAG baseline answers (58 responses total) using a blinded scoring workflow.

The experiments were designed as one-factor-at-a-time (OFAT) comparisons. This is interpretable, but interactions between parameters may not be captured.

## Retrieval metrics

- **Recall@5:** proportion of required gold evidence considered retrieved within the top five passages.
- **MRR@5:** mean reciprocal rank of the first relevant passage, truncated at rank five.

Metric values depend on the precise relevance-ground-truth and aggregation implementation. Consult the original evaluation notebooks and the final analysis workbook before reproducing or quoting results. The supplied experiment log and report abstract contain slightly different final retrieval summaries; the README flags this discrepancy rather than silently treating them as identical.

## Answer rubric

Each response was scored on:
- Faithfulness to the supplied guideline evidence: 0–2
- Clinical correctness: 0–2
- Completeness: 0–2
- Conciseness: 0–1

Total: 0–7. Unsupported claims were separately recorded as a yes/no outcome.

## Statistical analysis

The supplied analysis artefacts include paired Wilcoxon signed-rank tests for score outcomes and an exact McNemar analysis for paired unsupported-claim outcomes. Check the final report for the exact hypotheses, test settings, and reported p-values.
