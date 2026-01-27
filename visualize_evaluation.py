"""
Visualize the truthfulness evaluation results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_results(filepath):
    """Load evaluation results."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_visualizations(results, output_dir):
    """Create comprehensive visualizations of the evaluation results."""

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Extract data
    base_total = results['base_method']['total_claims']
    base_truthful = results['base_method']['truthful_count']
    base_false = base_total - base_truthful
    base_rate = results['base_method']['truthful_rate']

    weighted_total = results['weighted_method']['total_claims']
    weighted_truthful = results['weighted_method']['truthful_count']
    weighted_false = weighted_total - weighted_truthful
    weighted_rate = results['weighted_method']['truthful_rate']
    weighted_prior = results['weighted_method']['prior_connected_truthful']
    weighted_novel_truthful = weighted_truthful - weighted_prior

    improvement = results['comparison']['improvement']

    # 1. Bar chart: Truthful Rate Comparison
    ax1 = plt.subplot(2, 3, 1)
    methods = ['Base Method\n(SC ≥ 40th percentile)', 'Weighted Method\n(Closeness ≥ 0.6)']
    rates = [base_rate * 100, weighted_rate * 100]
    colors = ['#3498db', '#2ecc71']
    bars = ax1.bar(methods, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Truthful Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Truthfulness Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.axhline(y=85, color='gray', linestyle='--', alpha=0.5, label='85% baseline')
    ax1.legend()

    # 2. Stacked Bar: Claim Breakdown
    ax2 = plt.subplot(2, 3, 2)

    # Base method: all claims are evaluated by LLM
    base_data = [base_truthful, base_false]

    # Weighted method: prior + novel
    weighted_data = [
        weighted_prior,  # Prior-connected (auto-truthful)
        weighted_novel_truthful,  # Novel truthful
        weighted_false  # Novel false
    ]

    x = np.arange(2)
    width = 0.6

    # Base method bars
    ax2.bar(0, base_truthful, width, label='Truthful (LLM)', color='#2ecc71', alpha=0.7)
    ax2.bar(0, base_false, width, bottom=base_truthful, label='False/Unverifiable', color='#e74c3c', alpha=0.7)

    # Weighted method bars
    ax2.bar(1, weighted_prior, width, label='Prior-Connected (Auto)', color='#27ae60', alpha=0.9)
    ax2.bar(1, weighted_novel_truthful, width, bottom=weighted_prior, label='Novel Truthful (LLM)', color='#2ecc71', alpha=0.7)
    ax2.bar(1, weighted_false, width, bottom=weighted_prior + weighted_novel_truthful, label='Novel False/Unverifiable', color='#e74c3c', alpha=0.7)

    ax2.set_ylabel('Number of Claims', fontsize=12, fontweight='bold')
    ax2.set_title('Claim Type Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Base Method', 'Weighted Method'])
    ax2.legend(loc='upper right', fontsize=9)

    # Add count labels
    ax2.text(0, base_truthful/2, f'{base_truthful}', ha='center', va='center', fontweight='bold', fontsize=11)
    ax2.text(0, base_truthful + base_false/2, f'{base_false}', ha='center', va='center', fontweight='bold', fontsize=11)

    ax2.text(1, weighted_prior/2, f'{weighted_prior}', ha='center', va='center', fontweight='bold', fontsize=11)
    ax2.text(1, weighted_prior + weighted_novel_truthful/2, f'{weighted_novel_truthful}', ha='center', va='center', fontweight='bold', fontsize=11)
    ax2.text(1, weighted_prior + weighted_novel_truthful + weighted_false/2, f'{weighted_false}', ha='center', va='center', fontweight='bold', fontsize=11)

    # 3. Pie chart: Base Method
    ax3 = plt.subplot(2, 3, 3)
    base_labels = [f'Truthful\n({base_truthful}, {base_rate*100:.1f}%)',
                   f'False/Unverifiable\n({base_false}, {(1-base_rate)*100:.1f}%)']
    base_sizes = [base_truthful, base_false]
    base_colors = ['#2ecc71', '#e74c3c']
    ax3.pie(base_sizes, labels=base_labels, colors=base_colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('Base Method Distribution', fontsize=14, fontweight='bold')

    # 4. Pie chart: Weighted Method
    ax4 = plt.subplot(2, 3, 4)
    weighted_labels = [
        f'Prior-Connected\n({weighted_prior}, {weighted_prior/weighted_total*100:.1f}%)',
        f'Novel Truthful\n({weighted_novel_truthful}, {weighted_novel_truthful/weighted_total*100:.1f}%)',
        f'Novel False/Unverifiable\n({weighted_false}, {weighted_false/weighted_total*100:.1f}%)'
    ]
    weighted_sizes = [weighted_prior, weighted_novel_truthful, weighted_false]
    weighted_colors = ['#27ae60', '#2ecc71', '#e74c3c']
    ax4.pie(weighted_sizes, labels=weighted_labels, colors=weighted_colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax4.set_title('Weighted Method Distribution', fontsize=14, fontweight='bold')

    # 5. Error Reduction
    ax5 = plt.subplot(2, 3, 5)
    error_data = [base_false, weighted_false]
    error_reduction = (base_false - weighted_false) / base_false * 100

    bars = ax5.bar(methods, error_data, color=['#e74c3c', '#c0392b'], alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, count in zip(bars, error_data):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax5.set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
    ax5.set_title(f'Error Count (Reduction: {error_reduction:.1f}%)', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, max(error_data) * 1.2])

    # 6. Summary Statistics Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')

    table_data = [
        ['Metric', 'Base Method', 'Weighted Method', 'Improvement'],
        ['Total Claims', f'{base_total}', f'{weighted_total}', '—'],
        ['Truthful Claims', f'{base_truthful}', f'{weighted_truthful}', f'+{weighted_truthful - base_truthful}'],
        ['Truthful Rate', f'{base_rate*100:.2f}%', f'{weighted_rate*100:.2f}%', f'+{improvement*100:.2f}%'],
        ['False/Unverifiable', f'{base_false}', f'{weighted_false}', f'-{base_false - weighted_false}'],
        ['Error Rate', f'{(1-base_rate)*100:.2f}%', f'{(1-weighted_rate)*100:.2f}%', f'-{(1-base_rate-1+weighted_rate)*100:.2f}%'],
        ['LLM Evaluations', f'{base_total}', f'{weighted_total - weighted_prior}', f'-{weighted_prior}'],
        ['Prior-Connected', '0', f'{weighted_prior}', f'+{weighted_prior}'],
    ]

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color metric names
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')

    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Top-K Claims Truthfulness Evaluation: Base vs Weighted Centrality Method',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = Path(output_dir) / 'truthfulness_evaluation_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    # Create a second figure for detailed claim-level analysis
    create_claim_level_analysis(results, output_dir)

    plt.show()

def create_claim_level_analysis(results, output_dir):
    """Create detailed claim-level analysis."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract claim-level data
    base_claims = results['base_method']['results']
    weighted_claims = results['weighted_method']['results']

    # 1. Verdict Distribution - Base Method
    ax1 = axes[0, 0]
    base_verdicts = {}
    for claim in base_claims:
        verdict = claim['verdict']
        base_verdicts[verdict] = base_verdicts.get(verdict, 0) + 1

    verdicts_base = list(base_verdicts.keys())
    counts_base = list(base_verdicts.values())
    colors_base = ['#2ecc71' if v == 'TRUE' else '#e74c3c' if v == 'FALSE' else '#f39c12'
                   for v in verdicts_base]

    ax1.bar(verdicts_base, counts_base, color=colors_base, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Verdict', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Base Method: Verdict Distribution', fontweight='bold')
    for i, (v, c) in enumerate(zip(verdicts_base, counts_base)):
        ax1.text(i, c, str(c), ha='center', va='bottom', fontweight='bold')

    # 2. Verdict Distribution - Weighted Method
    ax2 = axes[0, 1]
    weighted_verdicts = {}
    for claim in weighted_claims:
        verdict = claim['verdict']
        weighted_verdicts[verdict] = weighted_verdicts.get(verdict, 0) + 1

    verdicts_weighted = list(weighted_verdicts.keys())
    counts_weighted = list(weighted_verdicts.values())
    colors_weighted = ['#27ae60' if v == 'PRIOR_CONNECTED' else
                      '#2ecc71' if v == 'TRUE' else
                      '#e74c3c' if v == 'FALSE' else '#f39c12'
                      for v in verdicts_weighted]

    ax2.bar(verdicts_weighted, counts_weighted, color=colors_weighted, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Verdict', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Weighted Method: Verdict Distribution', fontweight='bold')
    for i, (v, c) in enumerate(zip(verdicts_weighted, counts_weighted)):
        ax2.text(i, c, str(c), ha='center', va='bottom', fontweight='bold')

    # 3. Confidence Distribution - Base Method (LLM-evaluated)
    ax3 = axes[1, 0]
    base_confidences = [claim['confidence'] for claim in base_claims if claim['verdict'] != 'PRIOR_CONNECTED']
    if base_confidences:
        ax3.hist(base_confidences, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(base_confidences), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(base_confidences):.3f}')
        ax3.set_xlabel('Confidence Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Base Method: LLM Confidence Distribution', fontweight='bold')
        ax3.legend()

    # 4. Confidence Distribution - Weighted Method (novel claims only)
    ax4 = axes[1, 1]
    weighted_confidences = [claim['confidence'] for claim in weighted_claims if claim['verdict'] != 'PRIOR_CONNECTED']
    if weighted_confidences:
        ax4.hist(weighted_confidences, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(weighted_confidences), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(weighted_confidences):.3f}')
        ax4.set_xlabel('Confidence Score', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Weighted Method: LLM Confidence Distribution (Novel Claims)', fontweight='bold')
        ax4.legend()

    fig.suptitle('Claim-Level Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path(output_dir) / 'claim_level_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Claim-level analysis saved to: {output_path}")

def main():
    results_file = 'results/topk_truthfulness_evaluation.json'
    output_dir = 'results'

    # Load results
    print("Loading evaluation results...")
    results = load_results(results_file)

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, output_dir)

    print("\nVisualization complete!")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
