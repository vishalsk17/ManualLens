# Evaluation protocol

The included questions.jsonl is an original **synthetic** fixture. It contains 48 questions: setup 8, buttons/lights 8, error codes 8, paper jams 8, troubleshooting 6, follow-ups 4, unanswerable 6. The fixed split is 14 DEV / 34 TEST. Do not describe these results as real-product accuracy.

## Run retrieval without an API key

~~~powershell
$productId = "ID_FROM_INGEST"
python -m manualrag eval --product $productId --config A..E --synthetic --retrieval-only --output eval/results/synthetic-retrieval
~~~

A = BM25 text; B = dense text; C = hybrid text; D = hybrid plus cached page-card/figure-description chunks; E = D plus reranker. D/E are marked unavailable when visual chunks are absent. No fake cards or embeddings are substituted. Ingestion and local weights are cached.

The retrieval-only rewrite baseline concatenates supplied history; it is explicitly different from the full FAST_MODEL query rewrite. Gate abstention is reported separately from full generated-answer abstention. Ranking metrics use distinct pages in final ranked retrieval evidence, excluding attached warning/neighbor chunks.

DEV thresholds maximize abstention F1. TEST labels are not used to select thresholds or tune parameters. Configurations, exact DEV IDs, data hashes, and thresholds are saved per ablation. Do not tune configuration based on TEST outcomes; use a new unseen test set if you do.

## Live evaluation

Add ANTHROPIC_API_KEY to .env. First run Stage 1 with text-only A,B,C:

~~~powershell
python -m manualrag eval --product $productId --config A,B,C --synthetic --output eval/results/synthetic-full
~~~

For a real manual, create a separate questions JSONL and matching questions.manifest.json:

~~~json
{
  "source_type": "real",
  "model": "EXACT_MODEL_FROM_INGEST",
  "pdf_sha256": "FULL_PDF_SHA256",
  "title": "Human-annotated evaluation"
}
~~~

Use the same record schema as the synthetic dataset. Keep gold facts and page references specific to that exact PDF. Annotate 40–50 questions and fix the DEV/TEST assignment before experimenting. History entries contain question, answer, and pages. Run without --synthetic and with --dataset pointing to this file.

## Metrics and auditing

The report includes Recall@1/3/5, MRR, category sample counts, and Wilson confidence intervals for the binomial event “at least one gold page retrieved by page rank 5.” Recall itself is macro-averaged across answerable questions; the Wilson interval is not incorrectly applied to fractional recall.

Full runs add citation precision/recall against gold pages, LLM-judged atomic-claim support, gold-point coverage, visual component accuracy, abstention precision/recall/F1, false-refusal rate, required-warning recall, latency p50/p95, and per-query cost. Warning gold requirements are conservatively inferred from all sections represented on gold pages; inspect this approximation on manuals with many procedures per page.

The judge is instructed to assess only each claim's cited evidence, and its response must cover every answer step. This is a heuristic judgment, not proof of truth. Inspect results.json for claims and reasons.

Populate model_prices in config.yaml with exact model IDs and [input, output] USD per million tokens. Unconfigured prices are reported as unknown. User-query cost and latency exclude the evaluator judge and one-time ingestion/indexing; judge cost/tokens are separate CSV fields. Local CPU costs are not estimated.

Review at least 20 **distinct C answers** against the manual. Fill human_audit.csv with faithful=0 or 1, reviewer, and notes. No audit labels are prefilled. Existing filled audit worksheets are preserved.

~~~powershell
python -m manualrag audit --report-dir eval/results/real-full --audit-file eval/results/real-full/human_audit.csv
~~~

The command reports human/LLM agreement. Stage 1 qualification additionally requires a real-manual full run, the configured retrieval/abstention/citation/warning/faithfulness thresholds, and at least 20 reviewed answers. Synthetic and retrieval-only reports can never qualify. The thresholds are engineering acceptance choices in config.yaml, not statistically established guarantees.

## Stage 2 dependency

P7 (figure-level retrieval, photo hypotheses, optional API) remains pending until Stage 1 live evaluation and audit pass. P2 provides the cached page-card interface and PDF figure crops, but no photo diagnosis is exposed.

After Stage 1 qualification, enable and evaluate diagram-aware evidence with explicitly requested page-card ingestion. The optional page-image retrieval ablation F is not implemented. Small sample counts limit comparisons; report negative and inconclusive outcomes, not only improvements.
