# Graph-based Uncertainty Metrics for Long-form Language Model Outputs

Implementation of the paper ["Graph-based Uncertainty Metrics for Long-form Language Model Outputs"](https://arxiv.org/pdf/2410.20783) presented at NeurIPS 2024.

**Authors:** Mingjian Jiang, Yangjun Ruan, Prasanna Sattigeri, Salim Roukos, Tatsunori Hashimoto

## Overview

This repository provides a comprehensive framework for measuring and improving uncertainty in long-form LLM outputs through graph-based analysis of generation consistency and claim verification.

**Key Innovation:** Combines graph-based metrics (centrality), consistency scores (sentence matching), and verbalized confidence to provide interpretable uncertainty estimates for long-form LLM outputs.

## Architecture

The pipeline consists of 4 main stages:

```
Input Question
    ↓
Stage 1: Generate 10 diverse outputs
    ↓
Stage 2: Decompose into atomic claims + merge across generations
    ↓
Stage 3: Build bipartite graph + calculate uncertainty metrics
    ↓
Stage 4: Filter by uncertainty + reconstruct high-confidence output
```

## Repository Structure

```
Graph-based-Uncertainty/
├── main.py                          # Entry point and pipeline orchestrator
├── requirements.txt                 # Python dependencies
├── src/                            # Core implementation
│   ├── models.py                   # LLM model wrappers (OpenAI, Llama)
│   ├── model_generate.py           # Stage 1: Generation pipeline
│   ├── break_and_merge.py          # Stage 2: Claim breakdown and merging
│   ├── edge_construction.py        # Stage 3: Graph construction & metrics
│   ├── uad.py                      # Stage 4: Uncertainty-aware decoding
│   ├── utils.py                    # Utility functions
│   └── factscore_utils.py          # Fact-checking evaluation
├── data/                           # Datasets
│   ├── factscore/                  # FactScore dataset
│   └── pop_qa_filtered/            # PopQA filtered dataset
└── experiments/                    # Results and outputs
```

## Key Components

### Stage 1: Generation (`model_generate.py`)

Generates multiple diverse outputs per prompt:
- **One deterministic output** (greedy decoding, temperature ≈ 0)
- **N stochastic samples** (temperature = 0.7, default N = 10)
- Filters out uncertainty-expressing phrases ("I don't know", etc.)
- Implements caching to avoid recomputation

**Supported Models:**
- OpenAI: GPT-3.5-turbo, GPT-4
- Llama-3-70B-Instruct (via Hugging Face)

### Stage 2: Breakdown & Merge (`break_and_merge.py`)

The core claim processing module with multiple components:

**BreakdownProcessor:**
- Decomposes long-form text into atomic, standalone claims
- Extracts inline verbalized confidence for each claim
- Returns JSONL format: `{claim: str, gpt-confidence: float}`

**MatchProcessor:**
- Iteratively merges claims from multiple generations
- Uses bipartite matching to identify novel claims
- Filters out claims entailed by existing ones

**AutoEvalWrapper:**
- Fact-checks claims against Wikipedia or FactScore
- Integrates GPT-4-based verification
- Returns labels: Y (true), N (false), S (subjective)

**Output:** Merged claim list with per-claim metrics:
- `inline_verbalized_confidence` - From breakdown step
- `verbalized_confidence_with_options` - Explicit confidence scale
- `gpt-score` - Fact-check result
- `correctness` - Ground truth label

### Stage 3: Edge Construction (`edge_construction.py`)

Builds bipartite graphs and calculates uncertainty metrics:

**Bipartite Graph:**
- **Nodes:** All generations + all claims
- **Edges:** Claim-generation faithfulness connections

**Faithfulness Checking:**
- For each (claim, generation) pair: "Is the claim supported by the generation?"
- Aggregates into matching matrix: shape `(num_generations, num_claims)`

**Calculated Metrics:**

1. **Sentence Consistency (SC)**
   ```python
   SC_score[claim] = mean(match_matrix[:, claim])
   ```
   Average probability that claim appears in sampled generations

2. **Centrality Scores** (via NetworkX):
   - Eigenvector Centrality
   - Betweenness Centrality
   - Closeness Centrality
   - PageRank
   - Closeness with Node Confidence (custom weighted implementation)

3. **Hybrid Metrics:**
   - `sc_plus_vc` = SC + verbalized_confidence
   - `sc_based_vc` = SC + 0.1 × verbalized_confidence
   - `sc_based_ilvc` = SC + 0.1 × inline_verbalized_confidence

### Stage 4: Uncertainty-Aware Decoding (`uad.py`)

Filters claims and reconstructs coherent outputs:

**Threshold Calibration:**
```python
# Based on percentile of training set (first 20 examples)
threshold = np.percentile(training_scores, percentile)
```

**Claim Filtering:**
- Only keeps claims from "most-likely" generation that pass threshold
- Generates multiple output versions at different confidence percentiles (10%, 20%, ..., 90%)

**Output Merging:**
- Uses LLM to synthesize filtered claims into coherent paragraphs
- Preserves factual accuracy while maintaining readability

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Graph-based-Uncertainty.git
cd Graph-based-Uncertainty

# Install dependencies
pip install -r requirements.txt

# For FactScore evaluation, you may need:
python -m spacy download en_core_web_sm
```

### Requirements

**Core Dependencies:**
- `transformers==4.41.0` - Model loading and tokenization
- `torch==2.2.0` - GPU acceleration
- `networkx==3.2.1` - Graph algorithms
- `openai==1.30.5` - OpenAI API client
- `pandas==2.2.3` - Data manipulation
- `numpy==1.24.1` - Numerical operations

See `requirements.txt` for full dependency list.

## Usage

### Basic Generation

Generate multiple outputs for a dataset:

```bash
python main.py \
    --model gpt-4 \
    --num_generations_per_prompt 10 \
    --dataset factscore_m \
    --data_size 50
```

### With Breakdown & Metrics

Run generation with claim breakdown and uncertainty metrics:

```bash
python main.py \
    --model gpt-4 \
    --breakdown True \
    --sc_samples 4 \
    --gpt_annotate True \
    --dataset factscore_m \
    --data_size 50
```

### Full Pipeline with Uncertainty-Aware Decoding

Run all stages including UAD:

```bash
python main.py \
    --model llama-3-70b-instruct \
    --breakdown True \
    --sc_samples 4 \
    --gpt_annotate True \
    --uad True \
    --fs_eval True \
    --temperature 0.7 \
    --seed 10 \
    --data_size 100
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `gpt-3.5-turbo` | Model name (gpt-3.5-turbo, gpt-4, llama-3-70b-instruct) |
| `--num_generations_per_prompt` | int | 10 | Number of diverse outputs per prompt |
| `--temperature` | float | 0.7 | Sampling temperature |
| `--data_size` | int | 50 | Number of examples to process |
| `--dataset` | str | `factscore_m` | Dataset (factscore_m, pop_qa_filtered, nq) |
| `--breakdown` | bool | False | Enable text decomposition into claims |
| `--sc_samples` | int | 4 | Number of samples for sentence consistency |
| `--gpt_annotate` | bool | False | Enable fact verification |
| `--uad` | bool | False | Enable uncertainty-aware decoding |
| `--fs_eval` | bool | False | Enable FactScore evaluation |
| `--seed` | int | 10 | Random seed |

## Datasets

The pipeline supports three datasets:

### 1. FactScore (`factscore_m`)
- **Task:** Biography generation
- **Prompt:** "Tell me a paragraph bio of {entity}"
- **Evaluation:** FactScore framework with Wikipedia verification
- **Location:** `data/factscore/`

### 2. PopQA (`pop_qa_filtered`)
- **Task:** General knowledge QA
- **Prompt:** "Provide facts related to {topic}"
- **Evaluation:** Wikipedia-based
- **Location:** `data/pop_qa_filtered/`

### 3. Natural Questions (`nq`)
- **Task:** Long-form QA
- **Evaluation:** General Wikipedia evaluation

## Output Structure

### Generation Output

After Stage 1 (Generation):

```json
{
  "prompt": "Tell me a paragraph bio of Albert Einstein",
  "entity": "Albert Einstein",
  "most_likely_generation": "...",
  "more_generations": ["...", "...", ...]
}
```

### Breakdown Output

After Stage 2 (Break & Merge):

```json
{
  "most_likely_breakdown": ["claim1", "claim2", ...],
  "breakdown": ["all_merged_claims", ...],
  "pointwise_dict": [
    {
      "claim": "Albert Einstein was a physicist",
      "inline_verbalized_confidence": 0.95,
      "verbalized_confidence_with_options": 0.8,
      "sc_score_4samples": 0.75,
      "breakdown_closeness_centrality_4samples": 0.62,
      "sc_based_ilvc": 0.755,
      "gpt-score": "Y"
    }
  ]
}
```

### UAD Output

After Stage 4 (Uncertainty-Aware Decoding):

```json
{
  "merged_output_closeness_percentile40": "Filtered paragraph (closeness metric)...",
  "merged_output_sc_percentile40": "Filtered paragraph (SC metric)...",
  "merged_output_sc+ilvc_percentile40": "Filtered paragraph (hybrid metric)...",
  "merged_output_most_likely_generation": "Original unfiltered output"
}
```

## Caching Strategy

The system uses multi-level caching to avoid expensive recomputation:

1. **Generation Cache:** `{dataset}_{model}_generations.json`
   - Caches raw LLM responses

2. **Breakdown Cache:** `{dataset}_{model}_bipartite.json`
   - Stores breakdown and matching results

3. **Faithfulness Cache:** `sc_match_raw_return.json`
   - Stores faithfulness evaluations

4. **Fact Verification Cache:** `annotate_raw_all.json`
   - Stores FactScore/Wikipedia evaluations

5. **Verbalized Confidence Cache:** `vc_raw.json`
   - Stores confidence extraction results

Cache files are stored in `experiments/{dataset}/{model}/`.

## Uncertainty Metrics

### Sentence Consistency (SC)

Measures how consistently a claim appears across multiple generations:

```python
SC_score[claim] = mean(match_matrix[:, claim])
```

Higher SC indicates the model is more confident about the claim.

### Graph-Based Centrality

Uses NetworkX to compute:
- **Eigenvector Centrality:** Importance based on connected important nodes
- **Betweenness Centrality:** How often node lies on shortest paths
- **Closeness Centrality:** Average distance to other nodes
- **PageRank:** Network importance metric

### Verbalized Confidence

Extracts explicit confidence from LLM using:

**Option-based scale:**
- No chance (0%)
- Little chance (20%)
- Less than even (40%)
- Fairly possible (60%)
- Very good chance (80%)
- Almost certain (100%)

**Percentage-based:** Direct extraction of percentage values

### Hybrid Metrics

Combines multiple signals:
- `sc_based_ilvc = SC + 0.1 × inline_verbalized_confidence`
- Minimal computation, strong performance

## Evaluation

The framework includes calibration and evaluation tools:

- **Expected Calibration Error (ECE):** Measures calibration quality
- **AUROC:** Area under ROC curve for uncertainty ranking
- **Brier Score:** Probabilistic prediction accuracy
- **Calibration Plots:** Confidence vs. accuracy visualization

Evaluation functions are available in `src/utils.py`:

```python
from src.utils import calibration_measure

calibration_measure(
    df=results_df,
    label_column='correctness',
    confidence_columns=['sc_score', 'sc_based_ilvc'],
    output_path='calibration_plot.pdf'
)
```

## Model Support

### OpenAI Models (`src/models.py`)

Supports any OpenAI Chat Completion model:
```python
from src.models import OpenAIModel

model = OpenAIModel(model_name="gpt-4")
response = model.generate_n_given_prompt(prompt, n=10, temperature=0.7)
```

### Llama Models (`src/models.py`)

Llama-3-70B-Instruct via Hugging Face:
```python
from src.models import Llama3Model

model = Llama3Model()
response = model.generate_n_given_prompt(prompt, n=10, temperature=0.7)
```

Features:
- Flash Attention 2 for efficient inference
- Auto device mapping for multi-GPU
- 8K token context window

## Extending the Framework

### Adding New Datasets

1. Add dataset files to `data/your_dataset/`
2. Create prompt substitution function in `src/utils.py`:
   ```python
   def substitute_prompt_your_dataset(row):
       return f"Your prompt template with {row['column_name']}"
   ```
3. Update `main.py` to handle your dataset

### Adding New Models

1. Subclass `BaseModel` in `src/models.py`:
   ```python
   class YourModel(BaseModel):
       def generate_given_prompt(self, prompt, temperature=0.0):
           # Your implementation
           return {'generation': text, 'prompt': prompt}
   ```
2. Update `main.py` to instantiate your model

### Adding New Uncertainty Metrics

1. Implement metric calculation in `src/edge_construction.py` or `src/utils.py`
2. Add to `pointwise_dict` in breakdown pipeline
3. Update UAD threshold calibration to include your metric

## File Descriptions

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 81 | Pipeline orchestrator and entry point |
| `src/models.py` | 124 | LLM model wrappers (OpenAI, Llama) |
| `src/model_generate.py` | 106 | Stage 1: Generation with caching |
| `src/break_and_merge.py` | 423 | Stage 2: Claim breakdown and merging |
| `src/edge_construction.py` | 131 | Stage 3: Graph construction and metrics |
| `src/uad.py` | 214 | Stage 4: Uncertainty-aware decoding |
| `src/utils.py` | 300 | Shared utilities and helper functions |
| `src/factscore_utils.py` | 459 | FactScore integration for fact-checking |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{jiang2024graph,
  title={Graph-based Uncertainty Metrics for Long-form Language Model Outputs},
  author={Jiang, Mingjian and Ruan, Yangjun and Sattigeri, Prasanna and Roukos, Salim and Hashimoto, Tatsunori},
  booktitle={NeurIPS},
  year={2024}
}
```

## License

[Add your license information here]

## Acknowledgments

This work builds upon:
- [FactScore](https://github.com/shmsw25/FActScore) for fact verification
- NetworkX for graph algorithms
- Hugging Face Transformers for model implementations

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
