# Bipartite JSON File Structure

## Overview

The `*_bipartite_sc_*samples_*matches.json` file contains the complete bipartite graph data structure connecting **generations** (model responses) and **claims** (atomic facts), along with uncertainty metrics for each claim.

**File naming convention**: `<dataset>_<model>_bipartite_sc_<N>samples_<M>matches.json`
- `N samples`: Number of generations used (e.g., `5samples` = 1 most likely + 4 diverse)
- `M matches`: Number of generations used for claim extraction

---

## Top-Level Structure

```json
[
  {instance_1},
  {instance_2},
  ...
  {instance_N}
]
```

The file is a **list** where each element represents one entity/question processed by the pipeline.

---

## Instance Structure

Each instance is a **dictionary** with the following fields:

### 1. **Metadata Fields**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `prompt` | string | The input prompt given to the LLM | `"Tell me a paragraph bio of Billy Snedden.\n"` |
| `entity` | string | The entity/subject being queried | `"Billy Snedden"` |
| `wiki_title` | string | Wikipedia article title (for evaluation) | `"Billy Snedden"` |

---

### 2. **Generation Fields** (Model Responses)

| Field | Type | Description |
|-------|------|-------------|
| `most_likely_generation` | string | **Deterministic generation** (temperature ≈ 0)<br>This is the model's "best guess" answer |
| `more_generations` | list[string] | **List of diverse samples** (temperature = 0.7)<br>Default: 10 samples<br>Used to measure self-consistency across multiple responses |

**Example**:
```json
{
  "most_likely_generation": "Billy Snedden was an Australian politician who served...",
  "more_generations": [
    "Billy Snedden (1926-1987) was an Australian politician and lawyer...",
    "Sir Billy Snedden was a member of the Liberal Party...",
    ...
  ]
}
```

**Purpose**: Multiple generations allow measuring **consistency** - claims that appear across many samples are more likely to be true.

---

### 3. **Claim Fields** (Atomic Facts)

| Field | Type | Description |
|-------|------|-------------|
| `most_likely_breakdown` | list[string] | Claims extracted from `most_likely_generation` only |
| `most_likely_breakdown_len` | int | Number of claims in `most_likely_breakdown` |
| `breakdown` | list[string] | **UNION of all unique claims** from multiple generations<br>This is the merged set after removing duplicates |
| `breakdown_len` | int | Total number of unique claims |

**Example**:
```json
{
  "most_likely_breakdown": [
    "Billy Snedden was an Australian politician.",
    "He was born on 31 July 1926.",
    "He died in 1987."
  ],
  "most_likely_breakdown_len": 3,
  "breakdown": [
    "Billy Snedden was an Australian politician.",
    "He was born on 31 July 1926.",
    "He died in 1987.",
    "Billy Snedden served as Attorney-General.",  // From another generation
    "He was a member of the Liberal Party."       // From another generation
  ],
  "breakdown_len": 5
}
```

**Note**: `breakdown` contains **all unique claims** merged from multiple generations. Some claims appear in `most_likely_generation`, others only in the diverse samples.

---

### 4. **Bipartite Graph: Adjacency Matrix**

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `sc_match_4samples` | list[list[float]] | (N_gen × M_claims) | **Binary adjacency matrix**<br>`[i][j] = 1` if generation i supports claim j<br>`[i][j] = 0` otherwise |

**Example** (5 generations × 10 claims):
```python
sc_match_4samples = [
  [1, 1, 0, 1, 0, 1, 1, 0, 0, 1],  # Gen 0 supports claims 0,1,3,5,6,9
  [1, 1, 1, 1, 0, 0, 1, 0, 0, 0],  # Gen 1 supports claims 0,1,2,3,6
  [1, 0, 0, 1, 1, 1, 1, 0, 1, 1],  # Gen 2 supports claims 0,3,4,5,6,8,9
  [1, 1, 0, 0, 0, 1, 1, 1, 0, 1],  # Gen 3 supports claims 0,1,5,6,7,9
  [1, 1, 0, 1, 0, 1, 1, 0, 0, 0],  # Gen 4 supports claims 0,1,3,5,6
]
```

**Visual Representation**:

```
              Claim0  Claim1  Claim2  Claim3  Claim4  ...
Generation 0:   1       1       0       1       0     ...
Generation 1:   1       1       1       1       0     ...
Generation 2:   1       0       0       1       1     ...
Generation 3:   1       1       0       0       0     ...
Generation 4:   1       1       0       1       0     ...
```

**Interpretation**:
- **Row i**: Which claims are supported by generation i
- **Column j**: Which generations support claim j
- **1 (edge exists)**: The generation mentions/supports this claim
- **0 (no edge)**: The generation does not mention this claim

---

### 5. **Uncertainty Metrics: `pointwise_dict`**

| Field | Type | Description |
|-------|------|-------------|
| `pointwise_dict` | list[dict] | **List of dicts (one per claim)** containing all uncertainty metrics |

Each dict in `pointwise_dict` corresponds to **one claim** and contains:

#### 5.1 **Basic Fields**

| Field | Type | Description |
|-------|------|-------------|
| `claim` | string | The actual claim text |
| `correctness` | string | Gold label from FactScore (`"Y"` or `"N"`) - empty if not evaluated |
| `gpt-score` | string | GPT's evaluation of correctness |
| `gpt_annotation_result` | string | Full GPT annotation response |

#### 5.2 **Verbalized Confidence** (LLM's Own Confidence)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `inline_verbalized_confidence` | float | [0, 1] | Confidence extracted during claim breakdown<br>The LLM assigns confidence when generating claims |
| `verbalized_confidence_with_options` | float | [0, 1] | Confidence from explicit question with categorical options<br>Asked: "How confident are you this claim is true?"<br>Options: No chance (0), Little chance (0.2), ..., Almost certain (1.0) |

**Example**:
```json
{
  "claim": "Billy Snedden was born on 31 July 1926.",
  "inline_verbalized_confidence": 0.9,
  "verbalized_confidence_with_options": 0.8
}
```

#### 5.3 **Self-Consistency Score**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `sc_score_4samples` | float | [0, 1] | **Fraction of generations that support this claim**<br>Formula: `mean(sc_match[:, claim_j])` |

**Calculation**:
```python
# For claim j, count how many generations support it
sc_score = sum(sc_match_4samples[:, j]) / num_generations

# Example: If 4 out of 5 generations support the claim
sc_score = 4 / 5 = 0.8
```

**Interpretation**:
- **1.0**: All generations agree on this claim (high confidence)
- **0.5**: Half of generations mention it (medium confidence)
- **0.2**: Only 20% support it (low confidence, likely hallucination)

#### 5.4 **Graph Centrality Metrics**

All centrality metrics are computed on the bipartite graph.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `breakdown_closeness_centrality_4samples` | float | [0, 1] | **How "close" the claim is to all generations**<br>High value = well-connected = more certain |
| `breakdown_eigenvector_centrality_4samples` | float | [0, 1] | **Importance based on connected nodes**<br>Claims connected to important generations score higher |
| `breakdown_betweenness_centrality_4samples` | float | [0, 1] | **Broker role in graph**<br>Claims on shortest paths between generations |
| `breakdown_pagerank_4samples` | float | [0, 1] | **Authority-based ranking**<br>Random walk probability of reaching this claim |
| `breakdown_closeness_centrality_with_node_confidence_4samples` | float | [0, 1] | **Closeness weighted by verbalized confidence**<br>Combines graph structure + LLM confidence |

**Formulas**:

**Closeness Centrality**:
```
closeness(claim_j) = reachable_nodes / Σ(shortest_path_length(claim_j, gen_i))
```

**Self-Consistency** (for comparison):
```
sc_score(claim_j) = Σ(adjacency[i][j]) / num_generations
```

#### 5.5 **Composite Metrics**

| Field | Type | Description |
|-------|------|-------------|
| `sc_plus_vc` | float | `sc_score + verbalized_confidence_with_options`<br>Simple addition |
| `sc_based_vc` | float | `sc_score + 0.1 * verbalized_confidence_with_options`<br>SC dominates, VC breaks ties |
| `sc_based_ilvc` | float | `sc_score + 0.1 * inline_verbalized_confidence`<br>Uses inline confidence instead |

**Purpose**: Combine graph-based metrics (SC) with model confidence (VC) for better uncertainty estimation.

---

## Complete Example

```json
[
  {
    "prompt": "Tell me a paragraph bio of Billy Snedden.\n",
    "entity": "Billy Snedden",
    "wiki_title": "Billy Snedden",

    "most_likely_generation": "Billy Snedden was an Australian politician who served as a member of the Liberal Party...",

    "more_generations": [
      "Billy Snedden (1926-1987) was an Australian politician...",
      "Sir Billy Snedden was a member of the Liberal Party...",
      ...
    ],

    "most_likely_breakdown": [
      "Billy Snedden was an Australian politician.",
      "He served as a member of the Liberal Party.",
      "Billy Snedden was born on 31 July 1926."
    ],
    "most_likely_breakdown_len": 3,

    "breakdown": [
      "Billy Snedden was an Australian politician.",
      "He served as a member of the Liberal Party.",
      "Billy Snedden was born on 31 July 1926.",
      "He was born in Melbourne, Victoria.",
      "Billy Snedden died in 1987."
    ],
    "breakdown_len": 5,

    "sc_match_4samples": [
      [1.0, 1.0, 0.0, 1.0, 0.0],  // Gen 0
      [1.0, 1.0, 0.0, 0.0, 1.0],  // Gen 1
      [1.0, 0.0, 1.0, 1.0, 1.0],  // Gen 2
      [1.0, 1.0, 0.0, 1.0, 0.0],  // Gen 3
      [1.0, 1.0, 1.0, 1.0, 1.0]   // Gen 4
    ],

    "pointwise_dict": [
      {
        "claim": "Billy Snedden was an Australian politician.",
        "correctness": "Y",
        "inline_verbalized_confidence": 0.9,
        "verbalized_confidence_with_options": 0.8,
        "sc_score_4samples": 1.0,
        "breakdown_closeness_centrality_4samples": 0.5376,
        "breakdown_eigenvector_centrality_4samples": 0.2335,
        "breakdown_betweenness_centrality_4samples": 0.0244,
        "breakdown_pagerank_4samples": 0.0560,
        "breakdown_closeness_centrality_with_node_confidence_4samples": 0.4373,
        "sc_plus_vc": 1.8,
        "sc_based_vc": 1.08,
        "sc_based_ilvc": 1.09
      },
      {
        "claim": "Billy Snedden was born on 31 July 1926.",
        "correctness": "",
        "inline_verbalized_confidence": 0.9,
        "verbalized_confidence_with_options": 0.8,
        "sc_score_4samples": 0.4,
        "breakdown_closeness_centrality_4samples": 0.4322,
        "breakdown_eigenvector_centrality_4samples": 0.1520,
        "breakdown_betweenness_centrality_4samples": 0.0016,
        "breakdown_pagerank_4samples": 0.0237,
        "breakdown_closeness_centrality_with_node_confidence_4samples": 0.3590,
        "sc_plus_vc": 1.2,
        "sc_based_vc": 0.48,
        "sc_based_ilvc": 0.49
      }
    ]
  }
]
```

---

## How to Use This Data

### 1. **Access Claims for an Entity**

```python
import json

with open('bipartite_file.json', 'r') as f:
    data = json.load(f)

# Get first instance
instance = data[0]
entity = instance['entity']
claims = instance['breakdown']

print(f"Entity: {entity}")
print(f"Total claims: {len(claims)}")
for i, claim in enumerate(claims[:5]):
    print(f"  {i+1}. {claim}")
```

### 2. **Compute Self-Consistency Score Manually**

```python
import numpy as np

# Get adjacency matrix
adjacency = np.array(instance['sc_match_4samples'])  # Shape: (5, M)

# Compute SC score for each claim
sc_scores = adjacency.mean(axis=0)  # Average over generations

# Compare with stored values
for i, (claim_dict, computed_sc) in enumerate(zip(instance['pointwise_dict'], sc_scores)):
    stored_sc = claim_dict['sc_score_4samples']
    print(f"Claim {i}: Computed SC = {computed_sc:.2f}, Stored SC = {stored_sc:.2f}")
```

### 3. **Filter High-Confidence Claims**

```python
high_confidence_claims = []

for claim_dict in instance['pointwise_dict']:
    sc_score = claim_dict['sc_score_4samples']
    closeness = claim_dict['breakdown_closeness_centrality_4samples']

    # Keep only high-confidence claims
    if sc_score >= 0.6 and closeness >= 0.45:
        high_confidence_claims.append(claim_dict['claim'])

print(f"High-confidence claims: {len(high_confidence_claims)}/{len(instance['pointwise_dict'])}")
for claim in high_confidence_claims:
    print(f"  ✓ {claim}")
```

### 4. **Visualize Bipartite Graph**

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# Add generation nodes
num_gens = len(instance['sc_match_4samples'])
for i in range(num_gens):
    G.add_node(f"Gen{i}", bipartite=0)

# Add claim nodes
for j, claim_dict in enumerate(instance['pointwise_dict']):
    G.add_node(f"C{j}", bipartite=1,
               closeness=claim_dict['breakdown_closeness_centrality_4samples'])

# Add edges from adjacency matrix
for i in range(num_gens):
    for j in range(len(instance['breakdown'])):
        if instance['sc_match_4samples'][i][j] == 1:
            G.add_edge(f"Gen{i}", f"C{j}")

# Visualize
nx.draw(G, with_labels=True)
plt.show()
```

---

## Key Insights

### **Self-Consistency (SC) vs. Closeness Centrality**

| Metric | What it measures | When it's high | When it's low |
|--------|------------------|----------------|---------------|
| **SC Score** | Direct agreement across generations | Claim appears in most samples | Claim appears in few samples |
| **Closeness** | Graph connectivity | Claim is well-connected to generations | Claim is isolated |

**Example**:
- Claim: "Billy Snedden was born in 1926"
- If **SC = 0.2**: Only 1 out of 5 generations mention this
- If **Closeness = 0.3**: Even though few generations mention it, it's not well-connected in the graph structure

### **Typical Values**

Based on empirical observations:

| Uncertainty Level | SC Score | Closeness | Interpretation |
|-------------------|----------|-----------|----------------|
| **Very Certain** | > 0.8 | > 0.5 | Basic facts, widely agreed upon |
| **Certain** | 0.6 - 0.8 | 0.45 - 0.5 | General information, mostly consistent |
| **Uncertain** | 0.4 - 0.6 | 0.4 - 0.45 | Conflicting or rare information |
| **Very Uncertain** | < 0.4 | < 0.4 | Likely hallucination or very specific details |

### **Verbalized Confidence Interpretation**

- **High VC + High SC**: Model is confident AND consistent → Likely true
- **High VC + Low SC**: Model is confident BUT inconsistent → Overconfident hallucination
- **Low VC + High SC**: Model is uncertain BUT consistent → Likely true but model unsure
- **Low VC + Low SC**: Model is uncertain AND inconsistent → Definitely uncertain

---

## File Locations

Typical file paths:
```
experiments/<dataset>/<model>/<dataset>_<model>_bipartite_sc_<N>samples_<M>matches.json
```

Examples:
- `experiments/factscore/llama-3-8b-instruct/factscore_llama-3-8b-instruct_bipartite_sc_5samples_4matches.json`
- `experiments/pop_qa_filtered/llama-3-70b-instruct/pop_qa_filtered_llama-3-70b-instruct_bipartite_sc_5samples_4matches.json`

---

## Summary

The bipartite JSON file is the **core output** of the uncertainty quantification pipeline. It contains:

1. **Input**: Prompts and entity information
2. **Generations**: Multiple model responses (deterministic + diverse)
3. **Claims**: Atomic facts extracted and merged
4. **Graph**: Bipartite adjacency matrix connecting generations ↔ claims
5. **Metrics**: Comprehensive uncertainty scores for each claim

This structure enables:
- **Uncertainty quantification**: SC, closeness, betweenness, etc.
- **Claim filtering**: Keep only high-confidence claims
- **Visualization**: Bipartite graphs showing model consistency
- **Evaluation**: Compare against ground truth (FactScore)
