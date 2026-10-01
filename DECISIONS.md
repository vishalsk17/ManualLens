# ManualLens decisions

- The deliverable repository is in `outputs/ManualLens`. All generated/private data stays inside its ignored `data` directories.
- The Windows `py` launcher is broken on this host. Development uses the installed Python 3.12 executable, with a venv using existing system packages to avoid duplicating large dependencies. README setup uses the requested clean `py -m venv .venv` workflow and documents the fallback.
- Stage 1 uses PDF text and page images. VLM page cards have a cached implementation but are disabled by default. Enabling production Stage 2 requires a real-manual Stage 1 evaluation and manual audit; synthetic smoke results cannot unlock it.
- The supplied category counts sum to 48 when unanswerable has 6 questions. The synthetic development dataset therefore uses 48 questions, including 6 unanswerable, with a fixed stratified ~30/70 split.
- PDF page citations are one-based physical PDF pages, which may differ from printed page labels. Geometry is in the unrotated PDF coordinate system; page rendering uses an unrotated copy so coordinates remain exact.
- API credentials are never stored in source or traces. There is no silent model substitution or fake dense embedding fallback. Offline tests inject deterministic doubles and are labelled synthetic.
- Warning text is shown verbatim even when longer than the normal 25-word quote limit; the user's safety requirement takes precedence for attached warnings.
- An LLM's uncited summary is never displayed as a product fact. The verifier replaces it with a neutral status sentence; factual content is displayed only in cited steps. Citation validation establishes provenance, not semantic entailment; faithfulness still requires evaluation and human audit.
- Similarity thresholds start uncalibrated. Calibration uses DEV only, is stored separately by ablation and dataset hash, and never claims production validity from synthetic tests.

- The end-to-end test exposed a heading-only retrieval failure. Enrichment now attaches numbered procedure chunks from retrieved sections, not just immediate ordinal neighbors. All same-section warnings are attached afterward; excess evidence fails with a request to narrow the question.
- Fully scanned manuals are rejected in Stage 1 even if page cards are enabled. A VLM card is not a validated OCR replacement for safety-critical warnings. Mixed text/diagram PDFs remain supported; handwritten/scanned warning completeness is not guaranteed and needs later OCR validation.
- Threshold calibration uses lexical query-token coverage, comparable across A–E, plus deterministic wrong-model and unknown-code rejection. This is deliberately simple and must be recalibrated on real DEV data.
- Evaluation warning requirements are inferred conservatively from sections on gold pages. A real manual with multiple procedures per page needs annotation review to avoid overcounting.
- Stage 1 acceptance thresholds are configurable engineering choices: Recall@5 >= .8, abstention F1 >= .8, gold-page citation precision 1, warning recall 1, claim/human faithfulness >= .95. These are not statistical guarantees.
- Live model calls, real-manual results, human/LLM agreement, and API cost are not claimed without actual execution and manual audit. P7 is deferred until those prerequisites pass.
- The BGE model revision is pinned to the downloaded snapshot. requirements-tested.txt records direct and core ML library versions. This development venv inherits host packages; a clean venv is recommended for reproduction.
