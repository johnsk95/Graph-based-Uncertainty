# Phase 1: Weighted Bipartite Graph Construction with Knowledge Priors

This document describes the implementation of Phase 1 of the "Certainty Collapsing Boundary Expansion" framework.

## Overview

Phase 1 extends the baseline unweighted bipartite graph structure into a weighted, grounded knowledge graph using **Knowledge Priors** (WikiData) for factual verification.

### Key Innovations

1. **All responses generated with temperature** (no greedy generation)
2. **WikiData priors** as ground truth anchors
3. **Weighted edges** based on proximity to priors:
   - **Anchor Links** (w=3.0): Prior → Claim from prior
   - **Agreement Links** (w=2.0): Response → Claim also in prior
   - **Neutral Links** (w=1.0): Response → Novel claim
   - **Contradictory Links** (w=0.5): Response → Contradictory claim
4. **Weighted centrality metrics** using edge costs (1/weight)
5. **Claim categorization**: Grounded / Boundary / Contradictory

## File Structure

```
src/
├── wikidata_prior.py              # Fetches WikiData priors
├── model_generate_weighted.py     # Generates all responses with temperature
├── weighted_edge_construction.py  # Builds weighted bipartite graph
├── weighted_centrality.py         # Computes weighted centrality metrics
└── break_and_merge.py             # Existing claim decomposition (reused)

run_phase1_weighted_graph.py       # Main pipeline script
PHASE1_README.md                   # This file
graph_instructions.txt             # Technical specification
```

## Pipeline Stages

### Stage 1: Fetch WikiData Priors

```python
from src.wikidata_prior import WikiDataPriorFetcher

fetcher = WikiDataPriorFetcher(cache_dir="data/wikidata_cache")
prior = fetcher.get_prior_claims(wikidata_uri, subject_name)

# Returns:
# {
#   'uri': 'http://www.wikidata.org/entity/Q123',
#   'entity_id': 'Q123',
#   'subject': 'Albert Einstein',
#   'claims': [
#     'Albert Einstein is a physicist.',
#     'Albert Einstein was born on 1879.',
#     ...
#   ],
#   'raw_data': {...}
# }
```

### Stage 2: Generate Responses (All with Temperature)

```python
from src.model_generate_weighted import WeightedGeneration

generator = WeightedGeneration(args, dataset, llm_model)
sequences = generator.generate()

# All N responses generated with temperature=0.7
# No distinction between "greedy" and "stochastic"
```

### Stage 3: Decompose into Claims

```python
from src.break_and_merge import Break_And_Merge

breakdown = Break_And_Merge(args, sequences, llm_model)
sequences = breakdown.break_down_match()

# Decomposes both responses AND priors into atomic claims
```

### Stage 4: Construct Weighted Bipartite Graph

```python
from src.weighted_edge_construction import WeightedEdgeConstruction

graph_constructor = WeightedEdgeConstruction(args, sequences, llm_model)
sequences = graph_constructor.construct_all_weighted_graphs()

# For each claim:
# 1. Check if it's entailed by prior (faithfulness)
# 2. Check if it contradicts prior (LLM judge)
# 3. Assign edge weights:
#    - Anchor (3.0): Prior → Prior claim
#    - Agreement (2.0): Response → Prior claim
#    - Neutral (1.0): Response → Novel claim
#    - Contradictory (0.5): Response → Contradictory claim
```

### Stage 5: Compute Weighted Centrality

```python
from src.weighted_centrality import compute_all_weighted_metrics

metrics = compute_all_weighted_metrics(
    edge_weights, num_priors, num_responses, num_claims,
    delta_true=0.6, delta_false=0.2
)

# Computes:
# - Weighted closeness centrality (primary metric)
# - Weighted betweenness, eigenvector, pagerank
# - Claim categories (grounded/boundary/contradictory)
```

## Usage

### Basic Usage

```bash
python run_phase1_weighted_graph.py \
  --model gpt-3.5-turbo \
  --dataset pop_qa \
  --data_size 10 \
  --num_generations_per_prompt 5 \
  --temperature 0.7
```

### Advanced Usage

```bash
python run_phase1_weighted_graph.py \
  --model llama-3-8b-instruct \
  --dataset pop_qa \
  --data_size 50 \
  --num_generations_per_prompt 10 \
  --temperature 0.7 \
  --w_anchor 3.0 \
  --w_agree 2.0 \
  --w_neutral 1.0 \
  --w_contra 0.5 \
  --delta_true 0.6 \
  --delta_false 0.2 \
  --seed 42
```

### Pipeline Control

Control which stages to run:

```bash
python run_phase1_weighted_graph.py \
  --fetch_priors True \
  --generate True \
  --breakdown True \
  --construct_graph True \
  --compute_centrality True
```

## Output Files

All outputs are saved to `experiments/{dataset}/{model}/`:

```
experiments/pop_qa/gpt-3.5-turbo/
├── args_phase1.json                    # Arguments used
├── priors.json                         # WikiData priors
├── pop_qa_gpt-3.5-turbo_weighted_generations.json  # Generated responses
├── pop_qa_raw_return_weighted.json     # Raw generation cache
├── weighted_graph_raw_return.json      # Faithfulness/contradiction cache
├── pop_qa_gpt-3.5-turbo_weighted_graph.json       # Weighted graph
└── phase1_weighted_graph_final.json    # Final results with centrality
```

## Output JSON Structure

### Final Output (`phase1_weighted_graph_final.json`)

```json
[
  {
    "prompt": "Answer the following question: What is Albert Einstein's occupation?",
    "entity": "What is Albert Einstein's occupation?",
    "subject": "Albert Einstein",
    "ground_truth": "physicist",
    "s_uri": "http://www.wikidata.org/entity/Q937",

    "all_generations": [
      "Albert Einstein's occupation is physicist.",
      "He was a theoretical physicist who developed the theory of relativity.",
      "Albert Einstein worked as a physicist and professor.",
      ...
    ],

    "prior_claims": [
      "Albert Einstein is a physicist.",
      "Albert Einstein was born on 1879.",
      ...
    ],

    "breakdown": [
      "Albert Einstein is a physicist",
      "Albert Einstein developed the theory of relativity",
      "Albert Einstein was a professor",
      ...
    ],

    "weighted_graph": {
      "edge_weights": [
        [3.0, 3.0, 0.0, ...],  // Prior → Claims
        [2.0, 1.0, 1.0, ...],  // Response 0 → Claims
        [2.0, 2.0, 1.0, ...],  // Response 1 → Claims
        ...
      ],
      "edge_types": [
        ["anchor", "anchor", "none", ...],
        ["agreement", "neutral", "neutral", ...],
        ...
      ],
      "claims_in_prior": [0],
      "contradictory_claims": [],
      "num_priors": 1,
      "num_responses": 5,
      "num_claims": 10
    },

    "weighted_metrics": {
      "weighted_closeness": [0.85, 0.45, 0.38, ...],
      "avg_distance_to_sources": [0.42, 0.89, 1.23, ...],
      "betweenness_centrality": [0.12, 0.05, 0.02, ...],
      "eigenvector_centrality": [0.34, 0.18, 0.12, ...],
      "pagerank": [0.08, 0.05, 0.03, ...],
      "weighted_degree": [12.0, 5.0, 3.0, ...],
      "claim_categories": {
        "grounded": [0, 1],
        "boundary": [2, 3, 4],
        "contradictory": []
      }
    },

    "pointwise_dict": [
      {
        "claim": "Albert Einstein is a physicist",
        "weighted_closeness": 0.85,
        "avg_distance_to_sources": 0.42,
        "weighted_betweenness": 0.12,
        "weighted_eigenvector": 0.34,
        "weighted_pagerank": 0.08,
        "weighted_degree": 12.0,
        "category": "grounded",
        "in_prior": true,
        "is_contradictory": false,
        "edge_type_counts": {
          "anchor": 1,
          "agreement": 5,
          "neutral": 0,
          "contradictory": 0,
          "none": 0
        },
        "avg_edge_weight": 2.17
      },
      ...
    ]
  }
]
```

## Weighted Centrality Calculation

### Formula

```
CC(v) = (N-1) / Σ dW(v,u) · |Vv| / N
```

Where:
- `N`: Total number of nodes (priors + responses + claims)
- `dW(v,u)`: Shortest weighted path distance using costs = 1/weight
- `|Vv|`: Size of connected component containing node v

### Edge Cost Mapping

```
Anchor (w=3.0)        → cost = 1/3.0 = 0.33 (low cost, short path)
Agreement (w=2.0)     → cost = 1/2.0 = 0.50
Neutral (w=1.0)       → cost = 1/1.0 = 1.00
Contradictory (w=0.5) → cost = 1/0.5 = 2.00 (high cost, long path)
```

**Claims with high centrality** are closer to priors (shorter weighted paths) → more grounded.

**Claims with low centrality** are distant from priors or contradictory → hallucinations.

## Claim Categorization

Based on weighted closeness centrality:

| Category | Centrality Range | Description |
|----------|------------------|-------------|
| **Grounded** | CC ≥ δ_true (0.6) | High consensus, grounded to priors |
| **Boundary** | δ_false < CC < δ_true | Novel or contested claims (knowledge boundary) |
| **Contradictory** | CC ≤ δ_false (0.2) | Contradicts prior, consistent hallucination |

## Key Differences from Baseline

| Aspect | Baseline (Jiang et al.) | Phase 1 (This Implementation) |
|--------|------------------------|-------------------------------|
| **Priors** | None | WikiData entities |
| **Edge Weights** | Unweighted (binary) | Weighted (3.0, 2.0, 1.0, 0.5) |
| **Centrality** | Unweighted closeness | Weighted closeness (Dijkstra) |
| **Generation** | 1 greedy + N-1 stochastic | All N with temperature |
| **Contradiction Detection** | None | LLM-based judge |
| **Ground Truth** | Self-consistency only | Grounded to external knowledge |

## Example: Distinguishing Shared Hallucination

### Scenario

**Question**: "What is George Rankin's occupation?"

**WikiData Prior Claims**:
- "George Rankin is a politician"
- "George Rankin served in Parliament"

**LLM Responses** (all 5 generations):
1. "George Rankin is a politician and lawyer"
2. "George Rankin worked as a lawyer and politician"
3. "He was a politician and practiced law"
4. "George Rankin is a politician and lawyer"
5. "He served as a politician and attorney"

### Baseline Behavior

All responses consistently mention "lawyer" → **High self-consistency** → Claim "George Rankin is a lawyer" gets **high centrality** → Incorrectly classified as **TRUE**.

### Phase 1 Behavior

1. **Faithfulness Check**: Prior does NOT mention "lawyer"
2. **Contradiction Check**: LLM judge determines "is a lawyer" contradicts absence in prior
3. **Edge Assignment**: All Response → "is a lawyer" edges get **w=0.5** (contradictory)
4. **Weighted Centrality**: Claim "is a lawyer" gets **LOW centrality** (high cost paths)
5. **Categorization**: "is a lawyer" → **CONTRADICTORY** (or **BOUNDARY** if uncertainty)

Result: **Correctly identified as shared hallucination**, not true claim!

## Testing

### Test WikiData Fetcher

```bash
python -c "from src.wikidata_prior import test_wikidata_fetcher; test_wikidata_fetcher()"
```

### Test Weighted Centrality

```bash
python src/weighted_centrality.py
```

## Next Steps (Phase 2)

Phase 2 will implement:
1. **Claim clustering** and boundary identification
2. **Active learning** selection (centrality + novelty scores)
3. **Human-in-the-loop** verification simulation
4. **Knowledge expansion** (add verified claims as new priors)
5. **Iterative refinement** of the graph

## Dependencies

- `datasets`
- `requests`
- `networkx`
- `numpy`
- `torch`
- `tqdm`
- `openai` (for GPT models)
- `transformers` (for Llama models)

## Citation

This implementation extends:

```
Jiang et al. (2024). "Graph-based Uncertainty Metrics for Long-form Language Model Outputs."
NeurIPS 2024.
```

With the proposed framework:

```
Yi & Lee (2025). "Coevolving Human–LLM Organizational Knowledge Copilot:
Certainty Collapsing Boundary Expansion."
```

## Contact

For questions or issues, please refer to the project repository.
