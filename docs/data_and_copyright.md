# Data, privacy, and redistribution

The source archive supplied for repository preparation included the CPG PDF, extracted text, derived chunk CSVs, Chroma vector databases, raw generated answers, evaluation workbooks, and an unblinding key.

This public-ready repository intentionally does **not** include:
- the source guideline PDF;
- extracted full-text or chunk files derived from the guideline;
- prebuilt vector databases;
- raw generated answer/evaluation workbooks;
- the unblinding key.

These materials may contain copyrighted source text or detailed derivative excerpts. Before publishing any of them, confirm the applicable redistribution permissions and follow university/supervisor requirements. The exclusion is a conservative publication choice, not a legal determination that any specific item cannot be shared.

To run the app, use a corpus you are authorised to process and place a CSV with `chunk_id` and `text` columns at `data/local/recursive_chunks.csv`. Keep it local unless you have permission to redistribute it.

Before publishing:
- inspect all notebooks and Git history for API keys, tokens, credentials, private paths, and personal information;
- do not publish the unblinding key or raw evaluation sheets unless explicitly cleared;
- avoid patient-identifiable information;
- if a credential was ever committed, rotate it rather than only deleting it from the latest version;
- verify source-document attribution and any licence conditions.
