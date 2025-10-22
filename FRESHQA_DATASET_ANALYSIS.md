# FreshQA Dataset Analysis

## Overview

**FreshQA** is a question-answering dataset designed to test models on questions whose answers change over time. It focuses on "fresh" knowledge that requires up-to-date information.

**Dataset Location**: `data/freshqa/freshqa.csv`

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Questions** | 600 |
| **Train/Dev/Test Split** | DEV: 100, TEST: 500 |
| **False Premise Questions** | 148 (24.7%) |
| **True Premise Questions** | 452 (75.3%) |

---

## Column Structure

### 1. **Identification Fields**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | int | Unique question identifier | 0, 1, 2, ... |
| `split` | string | Dataset split | `"TEST"`, `"DEV"` |

### 2. **Question Fields**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `question` | string | The question to answer | `"What is the name of the first animal to land on the moon?"` |
| `false_premise` | bool | Whether the question contains a false premise | `True`, `False` |

**False Premise Examples**:
- `True`: "What is the name of Leonardo DiCaprio's third child?" (He has no children)
- `False`: "Who is the current president of the United States?" (Valid question)

### 3. **Metadata Fields**

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `num_hops` | string | Reasoning complexity | `"one-hop"` (463), `"multi-hop"` (137) |
| `fact_type` | string | How quickly facts change | `"never-changing"` (224)<br>`"slow-changing"` (222)<br>`"fast-changing"` (154) |
| `effective_year` | string | When this answer is valid | `"before 2022"`, `"2023"`, `"2024"` |
| `next_review` | string | When to review answer | `"occasionally"`, `"annually"`, `"monthly"` |
| `source` | string | URL sources for answers | Wikipedia, news articles, etc. |

**Fact Type Examples**:
- **Never-changing**: "What year did World War II end?" → 1945
- **Slow-changing**: "Who is the CEO of Apple?" → Changes every few years
- **Fast-changing**: "What is the current price of Bitcoin?" → Changes daily

**Num Hops Examples**:
- **One-hop**: Direct factual question → "Who wrote Romeo and Juliet?"
- **Multi-hop**: Requires multiple pieces of information → "What is the capital of the country where the Eiffel Tower is located?"

### 4. **Answer Fields**

| Column | Type | Description |
|--------|------|-------------|
| `answer_0` | string | Primary answer (always present) |
| `answer_1` | string | Alternative answer (optional) |
| `answer_2` to `answer_9` | string | Additional alternative answers (optional) |
| `note` | float | Additional notes (rarely used) |

**Multiple Answers**: Many questions have multiple valid answers depending on interpretation or timing.

---

## Answer Distribution

| # Answers | Count | Percentage |
|-----------|-------|------------|
| 1 answer | 303 | 50.5% |
| 2 answers | 142 | 23.7% |
| 3 answers | 58 | 9.7% |
| 4 answers | 39 | 6.5% |
| 5 answers | 11 | 1.8% |
| 6 answers | 6 | 1.0% |
| 7 answers | 4 | 0.7% |
| 10 answers | 37 | 6.2% |

---

## Example Questions

### Example 1: False Premise (No Valid Answer)

```json
{
  "id": 0,
  "question": "What is the name of the first animal to land on the moon?",
  "false_premise": true,
  "num_hops": "one-hop",
  "fact_type": "slow-changing",
  "answer_0": "No animal has ever landed on the moon yet.",
  "answer_1": "Since humans are animals, one could say Neil Armstrong."
}
```

**Analysis**: This question has a false premise. No non-human animal has landed on the moon. The answer depends on whether you consider humans as animals.

### Example 2: False Premise (Person Doesn't Exist)

```json
{
  "id": 1,
  "question": "What is the name of Leonardo DiCaprio's third child?",
  "false_premise": true,
  "num_hops": "one-hop",
  "fact_type": "slow-changing",
  "answer_0": "Leonardo DiCaprio does not have any children."
}
```

**Analysis**: The question assumes DiCaprio has children, which is false.

### Example 3: False Premise (Event Hasn't Happened)

```json
{
  "id": 2,
  "question": "What year did the first human land on Mars?",
  "false_premise": true,
  "num_hops": "one-hop",
  "fact_type": "slow-changing",
  "answer_0": "No humans have been to Mars yet."
}
```

**Analysis**: This question asks about a future event that hasn't occurred.

---

## Split Distribution

| Split | Count | Purpose |
|-------|-------|---------|
| **DEV** | 100 | Development/validation set for hyperparameter tuning |
| **TEST** | 500 | Test set for final evaluation |

---

## Fact Type Distribution

| Fact Type | Count | Percentage | Description |
|-----------|-------|------------|-------------|
| **never-changing** | 224 | 37.3% | Historical facts, mathematical constants, etc. |
| **slow-changing** | 222 | 37.0% | CEOs, government officials, sports records |
| **fast-changing** | 154 | 25.7% | Stock prices, weather, current events |

---

## Key Insights

### 1. **False Premises are Common**
- 24.7% of questions contain false premises
- Models must recognize when a question is based on incorrect assumptions
- Correct answer: "The premise is false" or equivalent

### 2. **Multiple Valid Answers**
- ~50% of questions have 2+ valid answers
- Answers may depend on:
  - Interpretation (e.g., "humans are animals")
  - Time sensitivity (e.g., "current president")
  - Level of detail (e.g., full name vs. last name)

### 3. **Time Sensitivity**
- `effective_year`: When the answer is correct
- `next_review`: How often to update
- Models need to be aware of temporal context

### 4. **Reasoning Complexity**
- 77% are one-hop (direct lookup)
- 23% are multi-hop (require reasoning across multiple facts)

---

## Comparison to Other Datasets

| Dataset | Focus | # Questions | False Premises | Time-Sensitive |
|---------|-------|-------------|----------------|----------------|
| **FreshQA** | Fresh knowledge, time-sensitive facts | 600 | ✓ (25%) | ✓ |
| **FactScore** | Biographical facts | ~100 entities | ✗ | ✗ |
| **PopQA** | Popular entities from Wikipedia | ~14k | ✗ | ✗ |
| **Natural Questions** | Open-domain QA from Google searches | ~300k | ✗ | Partial |

**FreshQA is unique** because:
1. It explicitly includes false premise questions
2. Answers are time-dated and require periodic review
3. It focuses on "fresh" knowledge that changes over time

---

## Usage with Graph-Based Uncertainty Pipeline

### Challenge: False Premise Detection

The pipeline currently expects:
- **Input**: Question about an entity
- **Output**: Paragraph of facts (claims)

For FreshQA:
- **False premise questions** → Model should output: "The premise is false"
- **Normal questions** → Model outputs factual answer

**Uncertainty metrics** (SC, closeness) should:
- **Be LOW** when model is uncertain or detects false premise
- **Be HIGH** when model confidently answers with correct facts

### Recommended Approach

1. **Generate multiple responses** per question (standard pipeline)
2. **Extract claims** from responses:
   - If most responses say "false premise" → High agreement on rejection
   - If responses give conflicting facts → Low confidence
3. **Compute uncertainty**:
   - SC score for "false premise" claims vs. factual claims
   - Closeness centrality to identify consistent vs. inconsistent responses

### Expected Patterns

| Question Type | Expected SC | Expected Closeness | Interpretation |
|---------------|-------------|-------------------|----------------|
| **True premise + certain answer** | High (>0.8) | High (>0.5) | Model is confident and consistent |
| **True premise + uncertain** | Medium (0.4-0.6) | Medium (0.4-0.5) | Model is unsure or conflicting |
| **False premise detected** | High (>0.8) | High (>0.5) | Model consistently rejects premise |
| **False premise not detected** | Low (<0.4) | Low (<0.4) | Model gives inconsistent hallucinations |

---

## Next Steps: Integrating FreshQA

### 1. **Convert CSV to Arrow Format**

The pipeline expects Hugging Face Arrow format. Convert CSV:

```python
import pandas as pd
from datasets import Dataset

# Load CSV
df = pd.read_csv('data/freshqa/freshqa.csv', skiprows=2)

# Convert to Dataset
dataset = Dataset.from_pandas(df)

# Save as Arrow
dataset.save_to_disk('data/freshqa_arrow')
```

### 2. **Add Prompt Template**

Add to `src/utils.py`:

```python
def substitute_prompt_freshqa(example):
    question = example['question']
    example['prompt'] = f'Answer the following question with a detailed paragraph: {question}'
    example['entity'] = question  # Use question as entity identifier
    example['wiki_title'] = question
    return example
```

### 3. **Update Main Pipeline**

Add to `main.py`:

```python
if 'freshqa' in args.dataset:
    questions = questions.map(lambda x: utils.substitute_prompt_freshqa(x))
```

### 4. **Run Pipeline**

```bash
conda run -n debate python main.py \
  --model llama-3-8b-instruct \
  --dataset freshqa_arrow \
  --data_size 50 \
  --breakdown True \
  --sc_samples 4
```

---

## Evaluation Metrics

For FreshQA, we can evaluate:

1. **False Premise Detection**:
   - Did the model recognize false premises?
   - SC score for "false premise" claims should be HIGH

2. **Answer Correctness**:
   - Compare model answers with `answer_0`, `answer_1`, etc.
   - Use exact match or semantic similarity

3. **Uncertainty Calibration**:
   - Do high SC scores correspond to correct answers?
   - Do low SC scores correspond to false premises?

4. **Time Sensitivity**:
   - Does the model update knowledge based on `effective_year`?
   - Can we detect when answers become outdated?

---

## Summary

**FreshQA** is a challenging dataset because:
- ✓ Questions have false premises (25%)
- ✓ Answers change over time
- ✓ Multiple valid answers exist
- ✓ Requires temporal reasoning

**For uncertainty quantification**:
- False premise questions → Test if model can **reject** bad questions
- Time-sensitive questions → Test if model has **up-to-date** knowledge
- Multi-answer questions → Test if model can **express uncertainty** appropriately

This makes FreshQA an **excellent test set** for graph-based uncertainty metrics!
