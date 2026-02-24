# Experiment Analysis: experiment_gpt4o

## Configuration
| Parameter | Value |
|-----------|-------|
| Papers | 50 (test split) |
| Generator Model | gpt-4o-mini |
| Verifier Model | gpt-4.1 (LLM) |
| Claim Extractor | LLM-based |
| Iterations | 3 |
| Claims/Iteration | 10 |
| Responses/Question | 3 |

## Key Metrics Summary

### Table 1: Metric Progression Across Iterations
| Metric | Iteration 1 | Iteration 2 | Iteration 3 | Change |
|--------|-------------|-------------|-------------|--------|
| **Prior Coverage Ratio (PCR)** | 0.300 ± 0.153 | 0.635 ± 0.135 | 0.682 ± 0.121 | **+38.1%** |
| **Knowledge Frontier (KF)** | 56.9 ± 35.3 | 29.4 ± 19.9 | 25.4 ± 17.6 | **-55.3%** |
| **Verification Utility Ratio** | 0.602 | 0.434 | 0.256 | -34.6% |
| **Grounded Claims** | 18.2 | 45.8 | 49.7 | +31.5 |

### Table 2: Verification Statistics
| Iteration | Verified | Accepted | Rejected | Accept Rate |
|-----------|----------|----------|----------|-------------|
| 1 | 500 | 301 | 199 | **60.2%** |
| 2 | 471 | 207 | 264 | **43.9%** |
| 3 | 456 | 122 | 334 | **26.8%** |
| **Total** | 1427 | 630 | 797 | **44.1%** |

## Key Findings

### 1. Progressive Section Expansion Works
- PCR improved from **30.0% → 68.2%** across iterations (+38.1%)
- Knowledge Frontier reduced by **55.3%** (56.9 → 25.4 contested claims)
- Most improvement comes from Iteration 0→1 (adding Introduction section)

### 2. Verification Acceptance Rate Decreases Over Iterations
- Iteration 0: **60.2%** acceptance rate
- Iteration 1: **43.9%** acceptance rate  
- Iteration 2: **26.8%** acceptance rate
- This suggests "easy" claims are verified first, leaving harder claims for later

### 3. Paper Size Correlation
- Larger papers (≥100 claims) show **higher absolute improvement** (+47.8%)
- Smaller papers (<50 claims) achieve **higher final PCR** (77.2%)
- Moderate negative correlation between claim count and final PCR (-0.335)

### 4. Utility-Based Selection
- All 150 selection rounds used utility-based ranking
- Selection prioritizes claims with high centrality and novelty scores

## Visualizations

| Figure | Description |
|--------|-------------|
| metric_progression.png | Bar chart of PCR, KF, VUR across iterations |
| verification_results.png | Accepted vs Rejected claims by iteration |
| pcr_distribution.png | Box plot of PCR distribution |
| claims_grounding.png | Stacked bar showing grounding progress |
| per_paper_improvement.png | PCR start vs end for all papers |

