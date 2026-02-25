# Graph-based-Uncertainty — Project Memory

## Project Summary
Extends NeurIPS 2024 graph-based LLM uncertainty paper with:
1. Weighted edge construction (prior-grounded bipartite graphs)
2. Iterative knowledge expansion with expert verification
3. QASPER experiments (NLP paper QA with progressive section priors)

## Key Source Files
- `src/weighted_edge_construction.py` — Phase 1: 4 edge weight types (anchor 3.0, agree 2.0, neutral 1.0, contra 0.5)
- `src/weighted_centrality.py` — Weighted closeness centrality via Dijkstra
- `src/iterative_expansion.py` — Phase 2: priority/cost/utility claim selection (828 lines)
- `src/expert_verification.py` — LLM/human/mock verifier interfaces
- `src/graph_update.py` — Post-verification graph updates
- `src/qasper_experiment.py` — QASPER orchestrator (progressive section expansion)
- `src/qasper_data.py` — QASPER dataset utilities
- `src/claims_qa.py` — Claims-based QA (4 conditions incl. graph_guided_retrieval)
- `src/embeddings.py` — Sentence-transformer embedding wrapper
- `run_verified_claims_qa.py` — QA experiment runner (main entry point for QA evaluation)

## Main Entry Points
- `run_phase1_weighted_graph.py` — Weighted graph construction
- `run_iterative_expansion.py` — Iterative claim selection
- `run_qasper_experiment.py` — QASPER experiment
- `run_verified_claims_qa.py` — QA experiment (4 conditions)
- `main.py` — Original 4-stage pipeline

## QASPER experiment_gpt4o
- 50 papers, 3 iterations, 10 claims/iter
- Generator: gpt-4o-mini, Verifier: gpt-4.1 (LLM)
- Results in `experiments/qasper/experiment_gpt4o/`
- PCR improved 30% → 68.2% across iterations
- Python env: `/opt/anaconda3/envs/autogen/bin/python`

## Edge Weight Defaults
w_anchor=3.0, w_agree=2.0, w_neutral=1.0, w_contra=0.5
delta_true=0.6, delta_false=0.2

## Current Branch
`weighted` — active development branch for weighted edge + iterative expansion work

## QA Experiment (graph_guided_retrieval) — Status
See `qa_experiment_design.md` for full details.

### Four conditions
1. `verified_claims` — flat list of grounded/accepted claims as context
2. `baseline_abstract` — title + abstract only
3. `baseline_full` — full paper text
4. `graph_guided_retrieval` — per-question passage retrieval from prior-revealed sections

### Prior sections per iteration (QASPER experiment)
- Iteration 0: Abstract only (P_0)
- Iteration 1: + Introduction (identified by `INTRODUCTION_VARIANTS`)
- Iteration 2: + Methods (identified by `METHODS_VARIANTS`)

### Key design constraint (user-confirmed)
graph_guided_retrieval MUST only retrieve from **prior-revealed sections** (Abstract, Introduction, Methods) — NOT the full paper. Using full paper would make it indistinguishable from baseline_full.

### Implementation details
- Provenance (source_section, source_paragraph) stored on claims and graph nodes in `qasper_experiment.py`
- `_get_prior_sections_from_results()` in `run_verified_claims_qa.py` reads graph snapshot prior nodes to identify which sections were used as priors, then builds paragraph corpus restricted to those sections
- `_enrich_claims_with_paper_sections()` maps claims without provenance to best-matching prior-section paragraph via word overlap (Jaccard)
- Centrality scores loaded from last iteration's graph_snapshot nodes (closeness centrality)
- Retrieval: 60% embedding similarity + 40% normalized centrality (or 100% centrality if no embeddings)
- `load_verified_claims_with_provenance()` in `claims_qa.py` handles both new results (provenance on claims) and old results (provenance reconstructed from graph anchor edges)

### Results location
- QA results: `experiments/qasper/verified_claims_qa/graph_guided_retrieval/qa_results.json` (user opened this)
- Knowledge expansion: `experiments/qasper/experiment_gpt4o/results.json`

### Remaining work
- Analyze qa_results.json to produce: knowledge expansion metrics table (PCR, KF, VUR by iteration) and QA performance comparison table (accuracy, answerability, acc@answerable per condition)
