# Bipartite Graph Visualization for Uncertainty Quantification

## Summary

This document describes the fixes and enhancements made to the Graph-based Uncertainty codebase to support **Llama 3.1 8B Instruct** and create visualizations of bipartite graphs with uncertainty annotations.

## Changes Made

### 1. Fixed NoneType Error in Break-and-Merge Phase

**Problem**: The `_parse_json_lines()` function returned `None` when JSON parsing failed, causing a crash when trying to iterate over the result.

**Solution** ([src/break_and_merge.py](src/break_and_merge.py)):
- Changed line 97: Return empty list `[]` instead of `None` when parsing fails
- Added `continue` instead of `return None` to skip failed lines gracefully
- Added null check in `_clean_breakdown_dicts()` at line 54 to handle empty inputs

**Result**: The pipeline now handles malformed LLM outputs gracefully and continues processing.

---

### 2. Bipartite Graph Generation

The pipeline generates bipartite graphs connecting **generations** (responses) and **claims** (atomic facts):

```
Generations (N samples)  ←──edges──→  Claims (M unique facts)
```

**Key Components:**

1. **Generation Stage** (already implemented):
   - Produces 1 "most likely" generation (temperature ≈ 0)
   - Produces N diverse samples (default N=10, temperature=0.7)

2. **Break-and-Merge Stage** (`--breakdown True`):
   - Extracts atomic claims from each generation
   - Merges semantically equivalent claims across samples
   - Result: M unique claims from N+1 generations

3. **Edge Construction Stage** (`--sc_samples 4`):
   - Creates bipartite graph with `sc_samples + 1` generations (5 total)
   - For each (generation, claim) pair: **faithfulness check** → "Is claim supported by generation?"
   - Produces N×M adjacency matrix stored in `sc_match_4samples`

4. **Uncertainty Metrics** (calculated per claim):
   - **Self-Consistency (SC) Score**: `mean(adjacency_matrix[:, claim_j])`
     - Proportion of generations supporting the claim
     - Range: [0, 1], higher = more consistent

   - **Closeness Centrality**: Graph-based metric
     - Measures how "close" a claim node is to all generation nodes
     - Range: [0, 1], higher = more certain
     - Formula: `reachable_nodes / sum(shortest_path_lengths)`

   - **Other metrics**: Eigenvector, Betweenness, PageRank centrality

**Output File**: `experiments/<dataset>/<model>/factscore_llama-3-8b-instruct_bipartite_sc_5samples_4matches.json`

---

### 3. Visualization Tool

Created `visualize_bipartite.py` to visualize bipartite graphs with uncertainty annotations.

#### Features:

1. **Two Visualization Styles**:
   - **Full view**: Shows all nodes, edges, claim text, and annotations
   - **Compact view**: Graph + uncertainty metrics table side-by-side

2. **Color Coding**:
   - **Generation nodes**: Blue squares (left side)
   - **Claim nodes**: Circles colored by closeness centrality (right side)
     - Red = Low certainty (low closeness)
     - Yellow = Medium certainty
     - Green = High certainty (high closeness)

3. **Annotations on Each Claim**:
   - **Closeness**: Closeness centrality score (0-1)
   - **SC Score**: Self-consistency score (0-1)

4. **Summary Statistics**:
   - Histogram of closeness centrality distribution
   - Histogram of self-consistency score distribution

#### Usage:

```bash
python visualize_bipartite.py \
  --input_file experiments/factscore/llama-3-8b-instruct/factscore_llama-3-8b-instruct_bipartite_sc_5samples_4matches.json \
  --output_dir visualizations/factscore_10examples \
  --max_instances 10
```

```bash
python visualize_bipartite.py \
  --input_file experiments/factscore/llama-3-70b-instruct/factscore_llama-3-70b-instruct_bipartite_sc_5samples_4matches.json \
  --output_dir visualizations/factscore_10examples \
  --max_instances 10
```

```
python visualize_bipartite.py \
  --input_file experiments/freshqa_false_premise/llama-3-8b-instruct/freshqa_false_premise_llama-3-8b-instruct_bipartite.json \
  --output_dir visualizations/freshqa_10examples \
  --max_instances 10
```

**Arguments**:
- `--input_file`: Path to bipartite JSON file (required)
- `--output_dir`: Output directory for visualizations (default: `visualizations`)
- `--max_instances`: Limit number of instances to visualize (optional)

**Output Files**:
- `bipartite_<idx>_<entity>_full.png`: Full visualization with edges and annotations
- `bipartite_<idx>_<entity>_compact.png`: Compact graph + metrics table
- `uncertainty_distribution.png`: Histogram of uncertainty across all instances (if > 1 instance)

---

## Running the Full Pipeline

### Recommended Command for Bipartite Graph Generation:

```bash
conda run -n debate python main.py \
  --model llama-3-8b-instruct \
  --dataset factscore \
  --data_size 10 \
  --breakdown True \
  --sc_samples 4 \
  --num_generations_per_prompt 10 \
  --temperature 0.7
```

**Parameters Explained**:
- `--model llama-3-8b-instruct`: Use Llama 3.1 8B Instruct
- `--dataset factscore`: FactScore biographical dataset
- `--data_size 10`: Process 10 entities
- `--breakdown True`: Enable claim extraction and merging (required for graphs!)
- `--sc_samples 4`: Use 5 generations (1 most likely + 4 diverse) for graph construction
- `--num_generations_per_prompt 10`: Generate 10 diverse samples total
- `--temperature 0.7`: Temperature for diverse sampling

**Note**: You need `--breakdown True` for bipartite graphs. Without it, no claims are extracted, so no graph can be built.

---

## Understanding the Metrics

### Self-Consistency (SC) Score

**Definition**: Fraction of sampled generations that support a given claim.

**Formula**:
```
SC(claim_j) = (# of generations supporting claim_j) / (total # of generations)
```

**Example**:
- 5 generations (1 most likely + 4 diverse)
- Claim: "Billy Snedden was born in 1926"
- 3 out of 5 generations mention this → SC = 0.6

**Interpretation**:
- **SC = 1.0**: All generations agree on this claim (high confidence)
- **SC = 0.5**: Half of generations support it (medium confidence)
- **SC = 0.2**: Only 20% support it (low confidence, likely hallucination)

---

### Closeness Centrality

**Definition**: Graph-based metric measuring how "close" a claim node is to all generation nodes in the bipartite graph.

**Formula**:
```
Closeness(claim_j) = reachable_nodes / Σ(shortest_path_length(claim_j, gen_i))
```

**Intuition**:
- If a claim is supported by many generations, it has short paths to many generation nodes
- High closeness → claim is well-connected → more certain

**Example**:
- Claim supported by 4/5 generations → many edges → high closeness
- Claim supported by 1/5 generations → few edges → low closeness

**Comparison to SC**:
- SC only counts direct support
- Closeness considers graph structure (indirect connections, network topology)
- In practice, they correlate but closeness can capture more nuanced uncertainty

---

## File Structure

```
Graph-based-Uncertainty/
├── main.py                          # Main pipeline entry point
├── visualize_bipartite.py           # Bipartite graph visualization tool (NEW)
├── BIPARTITE_VISUALIZATION_README.md # This file (NEW)
│
├── src/
│   ├── models.py                    # LLM wrappers (added Llama 3.1 8B support)
│   ├── utils.py                     # Utility functions (added load_llama3_8b_model_and_tokenizer)
│   ├── break_and_merge.py           # Claim extraction & merging (FIXED NoneType error)
│   ├── edge_construction.py         # Bipartite graph construction & uncertainty metrics
│   └── ...
│
├── experiments/
│   └── factscore/
│       └── llama-3-8b-instruct/
│           ├── factscore_llama-3-8b-instruct_generations.json           # Raw generations
│           ├── factscore_llama-3-8b-instruct_bipartite.json            # Claims after merging
│           └── factscore_llama-3-8b-instruct_bipartite_sc_5samples_4matches.json  # Final graph + metrics
│
└── visualizations/                  # Visualization output directory (NEW)
    └── factscore_10examples/
        ├── bipartite_000_Billy_Snedden_full.png
        ├── bipartite_000_Billy_Snedden_compact.png
        ├── bipartite_001_Bobby_Fischer_full.png
        ├── ...
        └── uncertainty_distribution.png
```

---

## Example Workflow

### Step 1: Run the Pipeline

```bash
conda run -n debate python main.py \
  --model llama-3-8b-instruct \
  --dataset factscore \
  --data_size 10 \
  --breakdown True \
  --sc_samples 4
```

**Expected Duration**: ~15-30 minutes for 10 examples (depends on LLM speed)

**Output**: `experiments/factscore/llama-3-8b-instruct/factscore_llama-3-8b-instruct_bipartite_sc_5samples_4matches.json`

---

### Step 2: Visualize the Bipartite Graphs

```bash
conda run -n debate python visualize_bipartite.py \
  --input_file experiments/factscore/llama-3-8b-instruct/factscore_llama-3-8b-instruct_bipartite_sc_5samples_4matches.json \
  --output_dir visualizations/factscore_10examples
```

**Expected Duration**: ~30 seconds for 10 examples

**Output**:
- 20 PNG files (2 per entity: full + compact)
- 1 histogram PNG

---

### Step 3: Analyze the Results

**Look at the visualizations**:

1. **Bipartite Graphs** (`*_full.png`, `*_compact.png`):
   - Are there many red (low certainty) claims? → Model is uncertain
   - Are there many green (high certainty) claims? → Model is confident
   - Do edges connect most claims to most generations? → High agreement

2. **Uncertainty Distribution** (`uncertainty_distribution.png`):
   - Is the closeness distribution skewed left (many low values)? → Many uncertain claims
   - Is the closeness distribution skewed right (many high values)? → Model is generally confident
   - Is the SC distribution bimodal? → Some claims have full agreement, others have disagreement

**Example Findings**:
- Claims about birth dates: Often have **low SC** (0.2-0.4) → models hallucinate dates
- Claims about basic facts (profession, nationality): **High SC** (0.8-1.0) → models agree
- Claims about specific details (exact positions, years): **Medium SC** (0.4-0.6) → partial agreement

---

## Key Insights

### 1. Self-Consistency vs. Closeness Centrality

Both metrics measure uncertainty, but from different angles:

| Metric | What it measures | Pros | Cons |
|--------|------------------|------|------|
| **Self-Consistency (SC)** | Direct support from generations | Simple, interpretable | Ignores graph structure |
| **Closeness Centrality** | Graph connectivity | Captures network effects | Less intuitive |

**Recommendation**: Use **both**! They often agree, but closeness can catch edge cases where SC misses nuances.

---

### 2. When to Use Which Metric

- **For filtering claims**: Use SC (simple threshold: keep claims with SC > 0.5)
- **For ranking claims**: Use closeness (smoother distribution, better for ordering)
- **For uncertainty-aware decoding**: Combine both (e.g., `score = SC + 0.1 * closeness`)

---

### 3. Bipartite Graphs Reveal Hallucinations

**Visual Inspection**:
- **Isolated claims** (few edges) = likely hallucinations
- **Well-connected claims** (many edges) = likely true
- **Clusters of connected claims** = coherent sub-narratives

**Example**:
```
Generation 1: "Billy Snedden was born in 1926"
Generation 2: "Billy Snedden was born in 1927"
Generation 3: "Billy Snedden was born on July 31, 1926"

→ Two competing claims about birth year
→ Visualization shows two separate claim nodes with different edge patterns
→ Helps identify conflicting information
```

---

## Troubleshooting

### Issue 1: NoneType Error in Break-and-Merge

**Error Message**: `TypeError: 'NoneType' object is not iterable`

**Cause**: LLM returned malformed JSON that couldn't be parsed

**Fix**: Already applied! The code now returns empty list instead of None.

---

### Issue 2: No Claims Generated

**Error Message**: `Empty breakdown for Xth generation`

**Cause**: LLM failed to generate valid JSONL claims

**Solution**:
- Check LLM prompt in `src/break_and_merge.py:24`
- Try different model (GPT-4 is more reliable than Llama 3.1 8B for structured output)
- Increase temperature slightly (0.7 → 0.8) for more diverse outputs

---

### Issue 3: Visualization Crashes

**Error Message**: `KeyError: 'breakdown_closeness_centrality_4samples'`

**Cause**: Ran visualization on file without `--sc_samples` parameter

**Solution**: Make sure you run with `--sc_samples > 0` and `--breakdown True`

---

## Next Steps

### 1. Run on More Examples

```bash
conda run -n debate python main.py \
  --model llama-3-8b-instruct \
  --dataset factscore \
  --data_size 50 \
  --breakdown True \
  --sc_samples 4
```

---

### 2. Compare Models

Run the same pipeline with different models:

```bash
# GPT-3.5-turbo
python main.py --model gpt-3.5-turbo --dataset factscore --data_size 10 --breakdown True --sc_samples 4

# GPT-4
python main.py --model gpt-4 --dataset factscore --data_size 10 --breakdown True --sc_samples 4

# Llama 3.1 70B
python main.py --model llama-3-70b-instruct --dataset factscore --data_size 10 --breakdown True --sc_samples 4
```

Then compare uncertainty distributions across models.

---

### 3. Uncertainty-Aware Filtering

Use the bipartite graph to filter claims before presenting to users:

```python
import json

# Load bipartite data
with open('factscore_llama-3-8b-instruct_bipartite_sc_5samples_4matches.json', 'r') as f:
    data = json.load(f)

# Filter high-confidence claims
for instance in data:
    high_confidence_claims = []
    for claim_dict in instance['pointwise_dict']:
        closeness = claim_dict['breakdown_closeness_centrality_4samples']
        sc_score = claim_dict['sc_score_4samples']

        # Keep only claims with high certainty
        if closeness > 0.5 and sc_score > 0.6:
            high_confidence_claims.append(claim_dict['claim'])

    print(f"Entity: {instance['entity']}")
    print(f"High-confidence claims: {len(high_confidence_claims)}/{len(instance['pointwise_dict'])}")
    for claim in high_confidence_claims:
        print(f"  ✓ {claim}")
```

---

## References

- **Paper**: "Graph-based Uncertainty Metrics for Long-form Language Model Outputs" (NeurIPS 2024)
  - Link: https://arxiv.org/pdf/2410.20783

- **Key Concepts**:
  - Self-consistency: Measuring agreement across multiple model samples
  - Bipartite graphs: Modeling generation-claim relationships
  - Centrality metrics: Quantifying uncertainty via graph structure

---

## Acknowledgments

- Original codebase: Graph-based Uncertainty repository
- Model: Meta Llama 3.1 8B Instruct
- Visualization: NetworkX + Matplotlib
