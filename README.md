# ManualLens

A typed Python RAG application that answers questions from one selected product PDF, verifies citations against retrieved evidence, displays exact page images, and restores attached safety warnings verbatim.

## Architecture

~~~mermaid
flowchart LR
  PDF[Product PDF] --> I[PyMuPDF ingest]
  I --> P[Page PNGs and exact bboxes]
  I --> C[Structure-aware chunks and tables]
  C --> S[(SQLite)]
  C --> B[BM25]
  C --> V[Local BGE embeddings]
  Q[Question and history] --> R[Standalone rewrite]
  R --> H[Exact code lookup and hybrid retrieval]
  B --> H
  V --> H
  S --> H
  H --> W[Warnings and adjacent steps]
  W --> G[Sufficiency gate]
  G --> A[JSON answer or abstention]
  A --> X[Citation verifier]
  X --> U[Streamlit chat and highlighted pages]
  P --> U
~~~

## Windows setup

Python 3.11 or newer is required. Run these PowerShell commands from this folder:

~~~powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-tested.txt
python -m pip install -e ".[test]"
Copy-Item .env.example .env
~~~

If the Windows py launcher reports no installed Python but python --version works, use python -m venv .venv. If PowerShell blocks activation, invoke .\.venv\Scripts\python.exe directly instead of changing system execution policy.

Add your Anthropic key to .env locally. Never paste a key into source or commit .env. The requested model IDs are configurable; unavailable models fail explicitly rather than being silently replaced. The first semantic search downloads BAAI/bge-small-en-v1.5 to data/cache/models; subsequent calls use local weights. There are no Poppler/Tesseract/system package requirements.

~~~powershell
python -m pytest -q --basetemp=data/cache/pytest
python -m streamlit run app/streamlit_app.py
~~~

## Synthetic demo

The original fixture describes a **fictional ML-100 printer**. It must not be used as advice for a real device.

~~~powershell
python -m manualrag synthetic
python -m manualrag ingest data/manuals/synthetic_ml100.pdf --name "ManualLens Lab Printer" --model "ML-100"
~~~

Copy the returned id into $productId:

~~~powershell
$productId = "PASTE_RETURNED_PRODUCT_ID"
python -m manualrag search "How do I clear a paper jam?" --product $productId --mode bm25
python -m manualrag ask "How do I clear a paper jam?" --product $productId
~~~

The ask result includes conversation_id. Reuse it with --conversation for follow-ups.

## Manual UI acceptance script

1. Start Streamlit. In **Add a product manual**, choose the PDF, enter a product name and model, and select **Process manual**.
2. Confirm upload progress completes and the selected manual has the expected page count.
3. Ask **How do I clear a paper jam?** Verify that a source citation accompanies every factual step.
4. Confirm the unplugging warning and roller caution are displayed verbatim with page citations.
5. Click a **p.4** citation. The right viewer must select physical PDF page 4 and highlight only source regions.
6. Ask **Which cover should I open?** Verify that the rewritten query includes the previous context and that retrieval ran again.
7. Ask **What does E-04 mean?** It must retrieve the E04 row for the active ML-100 and supporting jam instructions.
8. Ask **What does E99 mean?** Expect **Not in manual**, no invented instructions, and related pages labelled only as possibly related.
9. Ask about a different model. Expect a clear mismatch response; never reuse another manual's answer.
10. Use the page selector, accessible page text, graphic crops, and **Debug trace**. Confirm citation page numbers match the physical PDF pages.

Without an API key, ingestion, page browsing, and the search CLI work; the UI explicitly disables generated answers.

## Configuration and privacy

config.yaml contains models, retrieval sizes, thresholds, geometry, cache versions, and feature flags. .env can override model names. PDFs, embeddings, and traces remain local; API answering sends the question, recent conversation context, selected chunks and (when needed) at most two page images to Anthropic. Vision page cards send each page when explicitly enabled.

The default gate threshold is **uncalibrated** until DEV calibration. Model confidence is not a calibrated probability. Code citation checks prove source membership, not semantic support; the evaluation and human audit measure faithfulness.

Scanned PDFs with no text layer are rejected in ordinary Stage 1 ingestion. Fully scanned manuals require a separately validated OCR/transcription pipeline; cached vision page cards do not silently replace missing safety text. Stage 2 remains gated until a real-manual Stage 1 evaluation and audit pass.

See DECISIONS.md for implementation tradeoffs.

## Evaluation and build status

See [eval/README.md](eval/README.md) for A–E runs, real-manual dataset setup, metrics, cost configuration, and the human audit. The bundled report is [synthetic retrieval only](eval/results/synthetic-retrieval/report.md).

P0–P5 are implemented and tested locally. P6 provides the dataset, calibration, judge, reports, and audit workflow; real-manual/live-model validation is pending. P7 is deliberately not started until Stage 1 qualifies. A real printer PDF and an API key configured locally are the remaining required inputs.

The tested dependency list pins direct and core ML packages, not every transitive dependency. A clean virtual environment is recommended. Development reused host packages; unrelated pre-existing host dependency conflicts were not modified.
