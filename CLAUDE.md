# Graph-based Uncertainty — Project Guide

## Overview

This project implements graph-based uncertainty metrics for long-form LLM outputs, extending the NeurIPS 2024 paper by Jiang et al. The core idea: generate multiple diverse responses, decompose them into atomic claims, build a bipartite claim-source graph, and compute uncertainty via graph centrality and consistency measures.

**Key extensions in this codebase (beyond the original paper):**
1. **Weighted edge construction** grounded in external knowledge priors (WikiData / paper sections)
2. **Iterative knowledge expansion** with expert-in-the-loop verification and graph updates
3. **QASPER experiments** using NLP paper sections as progressive priors

---

## Architecture

```
Input Question
    ↓
Stage 1: Generate N diverse outputs  (model_generate.py / model_generate_weighted.py)
    ↓
Stage 2: Decompose → atomic claims, merge duplicates  (break_and_merge.py)
    ↓
Stage 3: Build bipartite graph + uncertainty metrics  (edge_construction.py)
    ↓
Stage 3b [NEW]: Weighted graph with prior grounding  (weighted_edge_construction.py)
    ↓
Stage 4: Uncertainty-aware decoding / output filtering  (uad.py)
    ↓
Phase 2 [NEW]: Iterative claim selection → expert verification → graph update
              (iterative_expansion.py → expert_verification.py → graph_update.py)
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/models.py` | LLM wrappers (OpenAI, Llama-3) |
| `src/model_generate.py` | Stage 1: diverse generation (1 greedy + N stochastic) |
| `src/model_generate_weighted.py` | All-temperature generation (no greedy sample) |
| `src/break_and_merge.py` | Stage 2: LLM-based claim decomposition and cross-generation merging |
| `src/edge_construction.py` | Stage 3: unweighted bipartite graph + centrality/consistency metrics |
| `src/weighted_edge_construction.py` | **Phase 1**: weighted graph with prior grounding |
| `src/weighted_centrality.py` | Weighted closeness centrality via Dijkstra |
| `src/wikidata_prior.py` | Fetch structured knowledge from WikiData as ground truth anchors |
| `src/iterative_expansion.py` | **Phase 2**: priority/cost/utility-based claim selection |
| `src/expert_verification.py` | Expert verifier interfaces (LLM, human, mock) |
| `src/graph_update.py` | Graph updates after verification (promotes claims to priors) |
| `src/qasper_experiment.py` | QASPER progressive section expansion orchestrator |
| `src/qasper_data.py` | QASPER dataset loader |
| `src/embeddings.py` | Sentence embeddings (sentence-transformers) |
| `src/uad.py` | Stage 4: uncertainty-aware decoding |
| `src/visualization.py` | Bipartite graph visualization |

---

## Weighted Edge Construction (Phase 1)

**File**: [src/weighted_edge_construction.py](src/weighted_edge_construction.py)

Builds a weighted bipartite graph between sources (responses + priors) and claims. Each edge gets a weight based on whether the claim is grounded in prior knowledge.

### Edge Weight Schema

| Edge Type | Weight | Condition |
|-----------|--------|-----------|
| `anchor` | 3.0 | Prior source → claim entailed by prior |
| `agreement` | 2.0 | Response → claim that is also in the prior |
| `neutral` | 1.0 | Response → novel claim (not in prior, not contradictory) |
| `contradictory` | 0.5 | Response → claim that contradicts prior knowledge |

### Processing Steps

1. **Prior → Claims (Anchor Links)**: LLM faithfulness check — "Is this claim supported by the prior context?" If yes, `w=3.0`.
2. **Contradiction Detection**: For non-prior claims, check via LLM if any claim contradicts known priors.
3. **Response → Claims (Weighted Links)**: For each (response, claim) pair, assign weight based on faithfulness + claim category.

### Claim Categories (by closeness centrality `CC`)

- **Grounded** (`CC ≥ delta_true = 0.6`): High consensus, anchored to priors
- **Boundary** (`0.2 < CC < 0.6`): Novel or contested
- **Contradictory** (`CC ≤ delta_false = 0.2`): Conflicts with priors

### Weighted Closeness Centrality

```
CC(v) = (N-1) / Σ dW(v,u)  ×  |Vv| / N
where dW uses edge costs = 1/weight
```

Implemented in [src/weighted_centrality.py](src/weighted_centrality.py) via Dijkstra's algorithm.

### Output JSON Structure

```json
{
  "edge_weights": [[3.0, 2.0, 1.0, ...]],
  "edge_types": [["anchor", "agreement", "neutral", ...]],
  "claims_in_prior": [0, 1, 5],
  "contradictory_claims": [3, 7],
  "edge_type_counts": {"anchor": 10, "agreement": 25, "neutral": 45, "contradictory": 3, "none": 17},
  "num_priors": 1, "num_responses": 10, "num_claims": 100
}
```

---

## Iterative Knowledge Expansion with Expert Feedback (Phase 2)

**File**: [src/iterative_expansion.py](src/iterative_expansion.py)

Selects the most informative contested claims for expert (LLM or human) verification, updates the graph, and repeats.

### Claim Selection Algorithm

**Step 1 — Priority Score**
```
Priority(c) = λ1 × S_centrality(c) + λ2 × S_novelty(c)
```
- `S_centrality`: Betweenness centrality on the contested subgraph (responses + contested claims only)
- `S_novelty`: Shortest path distance to nearest prior-entailed claim (farther = more novel = higher priority)
- Both are z-score normalized with positive shift

**Step 2 — Cost**
```
Cost(c) = 0.4 × min_dist + 0.6 × k_nearest_avg
```
Where distances are cosine distances from claim embeddings to prior embeddings. Lower cost = easier to verify.

**Step 3 — Utility**
```
Utility(c) = Priority(c) × (1 - σ(CC(c))) - Cost(c)
```
- `σ(CC)` converts closeness centrality to [0,1] confidence
- `(1 - σ(CC))` gives higher utility to uncertain claims
- Top-k claims by utility are selected for verification

**Step 4 — Graph Update** ([src/graph_update.py](src/graph_update.py))
- Accepted claims promoted to prior-entailed status (new anchor edges, `w=3.0`)
- Contested set recomputed for next iteration
- `update_after_verification()` prepares state for the next round

### Main Class

```python
class IterativeKnowledgeExpansion:
    def __init__(self, graph, claim_embeddings, prior_embeddings,
                 prior_entailed_claim_ids, response_node_ids,
                 closeness_centrality, contested_claim_ids=None,
                 lambda1=1.0, lambda2=1.0, k_neighbors=5)

    def select_claims(self, k: int) -> Tuple[Dict, Dict]:
        # Returns: {'verify': [claim_ids], 'discard': [claim_ids]}, diagnostics

    def update_after_verification(self, verified_claim_ids, accepted_claim_ids)
```

### Expert Verifier Interfaces ([src/expert_verification.py](src/expert_verification.py))

- `LLMVerifier`: Uses an LLM (e.g., GPT-4.1) to judge claim correctness against provided context
- `HumanVerifier`: Interactive CLI prompts for manual review
- `MockVerifier`: Accepts claims at a configurable rate (for testing)

---

## QASPER Experiment

**Orchestrator**: [src/qasper_experiment.py](src/qasper_experiment.py)
**Runner**: [run_qasper_experiment.py](run_qasper_experiment.py)
**Results**: [experiments/qasper/experiment_gpt4o/](experiments/qasper/experiment_gpt4o/)

### Design

QASPER is a QA dataset over 1,585 NLP papers. The experiment uses paper **sections as progressive priors**:

- **Iteration 0**: No priors — baseline graph
- **Iteration 1**: Add Abstract + Introduction as priors
- **Iteration 2**: Add Methods section as priors
- **Iteration 3**: Add Results + Conclusion as priors

Responses are generated once (fixed) in Iteration 0. Only the prior set expands across iterations, isolating the effect of grounding on claim categorization.

### experiment_gpt4o Configuration ([experiments/qasper/experiment_gpt4o/config.json](experiments/qasper/experiment_gpt4o/config.json))

```json
{
  "split": "test",
  "num_samples": 50,
  "seed": 42,
  "iterations": 3,
  "claims_per_iter": 10,
  "response_samples": 3,
  "generator_model": "gpt-4o-mini",
  "verifier_type": "llm",
  "openai_model": "gpt-4.1",
  "temperature": 0.7,
  "w_anchor": 3.0, "w_agree": 2.0, "w_neutral": 1.0, "w_contra": 0.5,
  "delta_true": 0.6, "delta_false": 0.2
}
```

### Key Results (from [analysis/ANALYSIS_REPORT.md](experiments/qasper/experiment_gpt4o/analysis/ANALYSIS_REPORT.md))

| Metric | Iter 1 | Iter 2 | Iter 3 | Change |
|--------|--------|--------|--------|--------|
| Prior Coverage Ratio (PCR) | 0.300 | 0.635 | 0.682 | **+38.1%** |
| Knowledge Frontier (KF) | 56.9 | 29.4 | 25.4 | **-55.3%** |
| Grounded Claims | 18.2 | 45.8 | 49.7 | +31.5 |
| Verification Accept Rate | 60.2% | 43.9% | 26.8% | decreasing |

- PCR: fraction of claims grounded to priors
- KF: number of contested (unverified) claims remaining
- Accept rate decreases across iterations — easy claims are verified first

### Experiment Output Structure

```
experiments/qasper/experiment_gpt4o/
├── config.json
├── results.json                   # Full results for all 50 papers
├── results_intermediate.json      # Checkpoints during run
├── analysis/
│   ├── ANALYSIS_REPORT.md
│   ├── metric_progression.png
│   ├── verification_results.png
│   ├── pcr_distribution.png
│   ├── claims_grounding.png
│   └── per_paper_improvement.png
└── vis/
    └── <arxiv_id>/
        ├── iteration_0.png
        ├── iteration_1.png
        ├── iteration_2.png
        └── final_graph.png
```

---

## Running Experiments

### Phase 1: Weighted Graph Construction

```bash
python run_phase1_weighted_graph.py \
  --model gpt-3.5-turbo \
  --dataset pop_qa \
  --data_size 100 \
  --num_generations_per_prompt 10 \
  --w_anchor 3.0 --w_agree 2.0 --w_neutral 1.0 --w_contra 0.5
```

### Phase 2: Iterative Expansion

```bash
python run_iterative_expansion.py \
  --model gpt-3.5-turbo \
  --dataset pop_qa \
  --k 20 \
  --lambda1 1.0 --lambda2 1.0 --k_neighbors 5
```

### QASPER Experiment

```bash
python run_qasper_experiment.py \
  --num_samples 50 \
  --iterations 3 \
  --claims_per_iter 10 \
  --verifier_type llm \
  --openai_model gpt-4.1
```

### Original 4-Stage Pipeline

```bash
python main.py \
  --model gpt-3.5-turbo \
  --dataset pop_qa \
  --data_size 10
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_anchor` | 3.0 | Edge weight: prior → prior-entailed claim |
| `w_agree` | 2.0 | Edge weight: response → prior-grounded claim |
| `w_neutral` | 1.0 | Edge weight: response → novel claim |
| `w_contra` | 0.5 | Edge weight: response → contradictory claim |
| `delta_true` | 0.6 | Closeness centrality threshold for "grounded" |
| `delta_false` | 0.2 | Closeness centrality threshold for "contradictory" |
| `lambda1` | 1.0 | Weight for centrality in priority score |
| `lambda2` | 1.0 | Weight for novelty in priority score |
| `k_neighbors` | 5 | k-nearest priors for cost computation |
| `claims_per_iter` | 10 | Claims selected per iteration for verification |
| `response_samples` | 3 | Number of LLM responses per question |

---

## Other Datasets

| Dataset | Location | Description |
|---------|----------|-------------|
| FactScore | `data/factscore/` | Biography generation (Wikipedia grounding) |
| PopQA | `data/pop_qa_filtered/` | General knowledge QA |
| QASPER | `data/qasper/` | NLP paper QA (1,585 papers, 5,049 questions) |
| WikiData cache | `data/wikidata_cache/` | Cached entity lookups |

---

## Documentation Files

| File | Contents |
|------|---------|
| `README.md` | Main usage guide |
| `PHASE1_README.md` | Weighted graph pipeline guide |
| `PHASE1_IMPLEMENTATION_SUMMARY.md` | Implementation status and test results |
| `graph_instructions.txt` | Mathematical formulations for weighted graph |
| `iterative_knowledge_expansion_instructions.txt` | Phase 2 algorithm specification |
| `expert_verification_graph_update_instructions.txt` | Verification and graph update spec |
| `BIPARTITE_JSON_STRUCTURE.md` | Output JSON format details |
