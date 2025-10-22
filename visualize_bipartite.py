#!/usr/bin/env python3
"""
Visualize bipartite graph of generations and claims with uncertainty annotations.
Usage: python visualize_bipartite.py --input_file <path_to_json> --output_dir <output_directory>
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import argparse
from pathlib import Path
import networkx as nx
from networkx.algorithms import bipartite

def load_bipartite_data(json_file):
    """Load the bipartite graph data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_bipartite_graph(instance_data):
    """Create NetworkX bipartite graph from instance data."""
    G = nx.Graph()

    # Get generations
    generations = [instance_data['most_likely_generation']] + instance_data['more_generations'][:4]
    claims = instance_data['breakdown']

    # Get the adjacency matrix (sc_match)
    if 'sc_match_4samples' in instance_data:
        adjacency_matrix = np.array(instance_data['sc_match_4samples'])
    else:
        # Fallback: create identity matrix if no data
        adjacency_matrix = np.ones((len(generations), len(claims)))

    # Add generation nodes (bipartite set 0)
    for i, gen in enumerate(generations):
        gen_short = gen[:50] + "..." if len(gen) > 50 else gen
        G.add_node(f"G{i}", bipartite=0, label=gen_short, full_text=gen)

    # Add claim nodes (bipartite set 1) with uncertainty scores
    for j, claim_dict in enumerate(instance_data['pointwise_dict']):
        claim = claim_dict['claim']
        closeness = claim_dict.get('breakdown_closeness_centrality_4samples', 0)
        sc_score = claim_dict.get('sc_score_4samples', 0)

        G.add_node(
            f"C{j}",
            bipartite=1,
            label=claim[:40] + "..." if len(claim) > 40 else claim,
            full_text=claim,
            closeness=closeness,
            sc_score=sc_score
        )

    # Add edges based on adjacency matrix
    for i in range(len(generations)):
        for j in range(len(claims)):
            if adjacency_matrix[i][j] > 0:
                G.add_edge(f"G{i}", f"C{j}", weight=adjacency_matrix[i][j])

    return G

def visualize_bipartite_graph(G, output_path, entity_name="Entity", show_edges=True):
    """Visualize the bipartite graph with uncertainty annotations."""

    # Separate nodes into two sets
    generation_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    claim_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]

    # Create figure with adequate size
    fig, ax = plt.subplots(figsize=(20, 12))

    # Position nodes in two columns
    pos = {}

    # Position generation nodes on the left
    y_spacing_gen = 1.0 / (len(generation_nodes) + 1)
    for i, node in enumerate(generation_nodes):
        pos[node] = (0, 1 - (i + 1) * y_spacing_gen)

    # Position claim nodes on the right
    y_spacing_claim = 1.0 / (len(claim_nodes) + 1)
    for i, node in enumerate(claim_nodes):
        pos[node] = (2, 1 - (i + 1) * y_spacing_claim)

    # Draw edges first (if enabled)
    if show_edges and G.edges():
        nx.draw_networkx_edges(
            G, pos, alpha=0.2, width=1, edge_color='gray', ax=ax
        )

    # Draw generation nodes (left side)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=generation_nodes,
        node_color='lightblue',
        node_size=500,
        node_shape='s',  # square
        alpha=0.8,
        ax=ax
    )

    # Draw claim nodes with colors based on closeness centrality (right side)
    closeness_values = [G.nodes[n].get('closeness', 0) for n in claim_nodes]

    # Normalize closeness for colormap
    if closeness_values and max(closeness_values) > 0:
        norm = plt.Normalize(vmin=min(closeness_values), vmax=max(closeness_values))
        cmap = plt.cm.RdYlGn  # Red (low uncertainty) to Green (high certainty)
    else:
        norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.RdYlGn

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=claim_nodes,
        node_color=closeness_values,
        node_size=600,
        node_shape='o',  # circle
        cmap=cmap,
        vmin=norm.vmin,
        vmax=norm.vmax,
        alpha=0.9,
        ax=ax
    )

    # Add labels with uncertainty scores
    labels = {}
    for node in generation_nodes:
        labels[node] = f"{node}"

    for node in claim_nodes:
        closeness = G.nodes[node].get('closeness', 0)
        sc_score = G.nodes[node].get('sc_score', 0)
        labels[node] = f"{node}\nClose: {closeness:.3f}\nSC: {sc_score:.2f}"

    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    # Add detailed claim text annotations on the right margin
    claim_annotations = []
    for i, node in enumerate(claim_nodes):
        claim_text = G.nodes[node]['full_text']
        closeness = G.nodes[node].get('closeness', 0)
        y_pos = pos[node][1]

        # Add text annotation
        ann = ax.text(
            2.3, y_pos,
            f"{claim_text[:80]}",
            fontsize=7,
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow' if closeness < 0.4 else 'lightgreen', alpha=0.3)
        )
        claim_annotations.append(ann)

    # Add colorbar for closeness centrality
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Closeness Centrality (Uncertainty)', rotation=270, labelpad=20)

    # Add legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, fc='lightblue', label='Generations'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='High Certainty Claims'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Low Certainty Claims')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Set title and labels
    ax.set_title(f'Bipartite Graph: Generations ↔ Claims\nEntity: {entity_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('← Generations (Left) | Claims (Right) →', fontsize=12)
    ax.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()

def visualize_compact_bipartite(G, output_path, entity_name="Entity"):
    """Create a more compact visualization focusing on the graph structure."""

    generation_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    claim_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [3, 1]})

    # --- LEFT PANEL: Graph ---
    pos = {}
    y_spacing_gen = 1.0 / (len(generation_nodes) + 1)
    for i, node in enumerate(generation_nodes):
        pos[node] = (0, 1 - (i + 1) * y_spacing_gen)

    y_spacing_claim = 1.0 / (len(claim_nodes) + 1)
    for i, node in enumerate(claim_nodes):
        pos[node] = (1, 1 - (i + 1) * y_spacing_claim)

    # Draw edges
    if G.edges():
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, edge_color='gray', ax=ax1)

    # Draw generation nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=generation_nodes,
        node_color='#4A90E2', node_size=800, node_shape='s',
        alpha=0.9, ax=ax1, edgecolors='black', linewidths=2
    )

    # Draw claim nodes with closeness coloring
    closeness_values = [G.nodes[n].get('closeness', 0) for n in claim_nodes]
    norm = plt.Normalize(vmin=min(closeness_values) if closeness_values else 0,
                         vmax=max(closeness_values) if closeness_values else 1)
    cmap = plt.cm.RdYlGn

    nx.draw_networkx_nodes(
        G, pos, nodelist=claim_nodes,
        node_color=closeness_values, node_size=700, node_shape='o',
        cmap=cmap, vmin=norm.vmin, vmax=norm.vmax,
        alpha=0.9, ax=ax1, edgecolors='black', linewidths=2
    )

    # Labels
    gen_labels = {node: node for node in generation_nodes}
    claim_labels = {node: node for node in claim_nodes}

    nx.draw_networkx_labels(G, pos, gen_labels, font_size=10, font_weight='bold', ax=ax1)
    nx.draw_networkx_labels(G, pos, claim_labels, font_size=9, ax=ax1)

    ax1.set_title(f'Bipartite Graph Structure\n{entity_name}', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # --- RIGHT PANEL: Uncertainty Table ---
    ax2.axis('tight')
    ax2.axis('off')

    # Create table data
    table_data = [['Claim', 'Closeness', 'SC Score']]
    for node in claim_nodes:
        claim_short = G.nodes[node]['label']
        closeness = G.nodes[node].get('closeness', 0)
        sc_score = G.nodes[node].get('sc_score', 0)
        table_data.append([claim_short[:30], f'{closeness:.3f}', f'{sc_score:.2f}'])

    table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows based on closeness
    for i, node in enumerate(claim_nodes, start=1):
        closeness = G.nodes[node].get('closeness', 0)
        color = cmap(norm(closeness))
        table[(i, 1)].set_facecolor(color)

    ax2.set_title('Uncertainty Metrics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved compact visualization to: {output_path}")
    plt.close()

def create_uncertainty_histogram(data, output_path):
    """Create histogram of uncertainty metrics across all instances."""

    all_closeness = []
    all_sc_scores = []

    for instance in data:
        for claim_dict in instance.get('pointwise_dict', []):
            closeness = claim_dict.get('breakdown_closeness_centrality_4samples', 0)
            sc_score = claim_dict.get('sc_score_4samples', 0)
            all_closeness.append(closeness)
            all_sc_scores.append(sc_score)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Closeness histogram
    ax1.hist(all_closeness, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Closeness Centrality', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Closeness Centrality\n(Higher = More Certain)', fontsize=12, fontweight='bold')
    ax1.axvline(np.mean(all_closeness), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_closeness):.3f}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # SC score histogram
    ax2.hist(all_sc_scores, bins=20, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Self-Consistency Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Self-Consistency Scores\n(Higher = More Consistent)', fontsize=12, fontweight='bold')
    ax2.axvline(np.mean(all_sc_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_sc_scores):.3f}')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved histogram to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize bipartite graph with uncertainty metrics')
    parser.add_argument('--input_file', type=str, required=True, help='Path to bipartite JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--max_instances', type=int, default=None, help='Maximum number of instances to visualize')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_bipartite_data(args.input_file)

    if not isinstance(data, list):
        data = [data]

    # Limit instances if specified
    if args.max_instances:
        data = data[:args.max_instances]

    print(f"Visualizing {len(data)} instance(s)...")

    # Create visualizations for each instance
    for idx, instance_data in enumerate(data):
        entity_name = instance_data.get('entity', f'Instance {idx}')
        print(f"\nProcessing {idx + 1}/{len(data)}: {entity_name}")

        # Create graph
        G = create_bipartite_graph(instance_data)

        # Full visualization
        output_full = output_dir / f"bipartite_{idx:03d}_{entity_name.replace(' ', '_')}_full.png"
        visualize_bipartite_graph(G, output_full, entity_name)

        # Compact visualization
        output_compact = output_dir / f"bipartite_{idx:03d}_{entity_name.replace(' ', '_')}_compact.png"
        visualize_compact_bipartite(G, output_compact, entity_name)

    # Create summary histogram
    if len(data) > 1:
        histogram_path = output_dir / "uncertainty_distribution.png"
        create_uncertainty_histogram(data, histogram_path)

    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"  - Created {len(data) * 2} bipartite graphs (full + compact)")
    if len(data) > 1:
        print(f"  - Created 1 uncertainty distribution histogram")

if __name__ == "__main__":
    main()
