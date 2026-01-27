# FACTS Dataset Integration

This document describes how to use the FACTS grounding dataset with the weighted bipartite graph framework.

## Dataset Overview

**FACTS (Factuality Assessment of Claims and Text Summaries)** is a grounding dataset from Google that contains:
- `user_request`: User questions or prompts
- `context_document`: Reference document with factual information
- `system_instruction`: Template for combining the above
- `full_prompt`: Complete prompt with all components

## Integration Details

### Prior Knowledge Source
For FACTS dataset, we use the `context_document` as the knowledge prior instead of WikiData. The context document is processed to extract atomic factual claims using an LLM.

### Prompt Format
The `user_request` field is used directly as the prompt for LLM response generation.

### Entity Identification
Since FACTS doesn't have entity IDs like PopQA, we use a truncated version of the `user_request` (first 50 characters) as the entity identifier.

## Usage

### Basic Command

```bash
python3 run_phase1_weighted_graph.py \
  --dataset facts \
  --data_size 10 \
  --num_generations_per_prompt 5 \
  --temperature 0.7 \
  --model gpt-3.5-turbo
```

### With Visualization

```bash
python3 run_phase1_weighted_graph.py \
  --dataset facts \
  --data_size 10 \
  --num_generations_per_prompt 5 \
  --temperature 0.7 \
  --model gpt-3.5-turbo \
  --vis True \
  --vis_max_instances 10 \
  --vis_max_claims 20
```

## Output Structure

The pipeline produces the same output structure as PopQA:

- `priors.json`: Extracted claims from context documents
- `facts_gpt-3.5-turbo_weighted_generations.json`: LLM responses
- `facts_gpt-3.5-turbo_weighted_graph.json`: Weighted bipartite graphs
- `phase1_weighted_graph_final.json`: Final results with centrality metrics
- `visualizations/`: Graph visualizations (if --vis True)

## Example Results

Running on 3 FACTS examples with 5 generations each:

```
Total examples processed: 3
Total claims: 53
Grounded claims: 36 (67.9%)
Boundary claims: 17 (32.1%)
Contradictory claims: 0 (0.0%)
```

### Example 1: Dementia Risk Factors
- **User Request**: "I'm middle-aged, never smoked, had my ears blown out in the war, get a case of the sads pretty regular, and eat mostly garbage. What are my risk factors for dementia?"
- **Prior Claims**: 18 (extracted from health information document)
- **LLM Claims**: 15
- **Result**: All claims grounded or boundary (no contradictions with medical facts)

### Example 2: Knife Sharpening Angles
- **User Request**: "Can you list all the knife brands that sell knives suitable for sharpening at a 10-15 degree angle?"
- **Prior Claims**: 15 (extracted from knife manufacturer specifications)
- **LLM Claims**: 16
- **Result**: 11 contradictory claims (LLM hallucinated brands/angles not in the document)

## Key Differences from PopQA

| Aspect | PopQA | FACTS |
|--------|-------|-------|
| Prior Source | WikiData API | Context document (extracted claims) |
| Prompt | Question field | User request field |
| Entity ID | WikiData URI | Truncated user request |
| Prior Format | Structured KB facts | Free-text document |

## Implementation Files Modified

1. `run_phase1_weighted_graph.py`:
   - Added `load_facts_dataset()` function
   - Added `extract_claims_from_text()` for prior extraction
   - Modified `fetch_all_priors()` to handle both datasets

2. `src/model_generate_weighted.py`:
   - Updated entity name extraction to use `subj` field (populated by FACTS loader)

## Notes

- Prior extraction from context documents uses temperature=0.0 for consistency
- Claim extraction quality depends on LLM's ability to parse unstructured text
- Longer context documents may produce many prior claims (e.g., 55 claims in the money-saving example)
- The contradiction detection works well for factual discrepancies (as shown in the knife example)
