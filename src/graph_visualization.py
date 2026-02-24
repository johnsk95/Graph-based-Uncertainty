"""
Knowledge Graph Visualization Module

Visualizes the knowledge graph at each iteration of the QASPER experiment.
Supports color coding for different node types and claim states.

Node Types:
- P_i: Prior nodes (from title/abstract) - Green
- R_i: Response nodes - Blue
- E_i: Expert-verified prior nodes - Orange
- C_i: Claim nodes - Colored by state

Claim States:
- Grounded (prior-entailed): Light green
- Contested: Gray
- Verified & Accepted: Dark green
- Verified & Rejected: Red
"""

import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


# ==============================================================================
# COLOR SCHEMES
# ==============================================================================

NODE_COLORS = {
    # Source nodes
    "prior": "#2ECC71",        # Green - original priors (P_i)
    "response": "#3498DB",     # Blue - responses (R_i)
    "expert_verified": "#E67E22",  # Orange - expert verified (E_i)
    "section_prior": "#27AE60",    # Darker green - section priors

    # Claim nodes by state
    "claim_grounded": "#A9DFBF",    # Light green - grounded claims
    "claim_contested": "#BDC3C7",   # Gray - contested claims
    "claim_accepted": "#1ABC9C",    # Teal - verified & accepted
    "claim_rejected": "#E74C3C",    # Red - verified & rejected
}

NODE_SHAPES = {
    "prior": "s",      # Square
    "response": "o",   # Circle
    "expert_verified": "^",  # Triangle up
    "section_prior": "D",    # Diamond
    "claim": "o",      # Circle
}


# ==============================================================================
# GRAPH RECONSTRUCTION
# ==============================================================================

def reconstruct_graph_from_snapshot(graph_snapshot: Dict) -> nx.Graph:
    """
    Reconstruct a NetworkX graph from a serialized graph snapshot.

    Args:
        graph_snapshot: Dict with 'nodes' and 'edges' lists

    Returns:
        NetworkX Graph
    """
    G = nx.Graph()

    # Add nodes
    for node_data in graph_snapshot.get("nodes", []):
        node_id = node_data["id"]
        G.add_node(node_id, **node_data)

    # Add edges
    for edge_data in graph_snapshot.get("edges", []):
        source = edge_data["source"]
        target = edge_data["target"]
        G.add_edge(source, target, **edge_data)

    return G


def get_node_display_id(node_id: str, node_type: str, iteration: int = 0) -> str:
    """
    Convert internal node ID to display format.

    Args:
        node_id: Internal node ID (e.g., "S_prior", "S_R0", "C5")
        node_type: Node type
        iteration: Current iteration for context

    Returns:
        Display ID (e.g., "P_0", "R_1", "E_0", "C_5")
    """
    if node_id == "S_prior":
        return "P_0"
    elif node_id.startswith("S_R"):
        # Response node: S_R0 -> R_0
        idx = node_id[3:]
        return f"R_{idx}"
    elif node_id.startswith("S_verified_"):
        # Expert verified: S_verified_0 -> E_0
        idx = node_id.split("_")[-1]
        return f"E_{idx}"
    elif node_id.startswith("S_section_"):
        # Section prior: S_section_1 -> P_1 (section)
        idx = node_id.split("_")[-1]
        return f"P_{idx}"
    elif node_id.startswith("C"):
        # Claim node
        return node_id
    else:
        return node_id


def get_node_color(
    node_id: str,
    node_attrs: Dict,
    prior_entailed: List[str],
    verification_results: List[Dict]
) -> str:
    """
    Determine node color based on type and state.

    Args:
        node_id: Node ID
        node_attrs: Node attributes
        prior_entailed: List of grounded claim IDs
        verification_results: List of verification result dicts

    Returns:
        Color string
    """
    node_type = node_attrs.get("type", "unknown")

    # Source nodes
    if node_type == "prior":
        if "verification_round" in node_attrs:
            return NODE_COLORS["expert_verified"]
        elif "section_name" in node_attrs:
            return NODE_COLORS["section_prior"]
        else:
            return NODE_COLORS["prior"]
    elif node_type == "response":
        return NODE_COLORS["response"]

    # Claim nodes
    if node_type == "claim" or node_id.startswith("C"):
        # Check verification status
        verified_results = {v["claim_id"]: v for v in verification_results}

        if node_id in verified_results:
            if verified_results[node_id]["verdict"]:
                return NODE_COLORS["claim_accepted"]
            else:
                return NODE_COLORS["claim_rejected"]
        elif node_id in prior_entailed:
            return NODE_COLORS["claim_grounded"]
        else:
            return NODE_COLORS["claim_contested"]

    return "#95A5A6"  # Default gray


def get_node_size(node_type: str) -> int:
    """Get node size based on type."""
    if node_type in ["prior", "response"]:
        return 800
    elif node_type == "claim":
        return 400
    else:
        return 500


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def visualize_iteration_graph(
    graph_snapshot: Dict,
    iteration: int,
    paper_id: str,
    prior_entailed: List[str],
    verification_results: List[Dict],
    output_path: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 16),
    show_only_grounded: bool = False,
    rejected_claim_ids: Optional[List[str]] = None
):
    """
    Visualize the knowledge graph for a single iteration.

    Args:
        graph_snapshot: Graph snapshot dict
        iteration: Iteration number
        paper_id: Paper ID for labeling
        prior_entailed: List of grounded claim IDs
        verification_results: Verification results for this iteration
        output_path: Path to save the figure
        title: Optional custom title
        figsize: Figure size
        show_only_grounded: If True, only show grounded/verified claims (for final graph)
        rejected_claim_ids: List of claim IDs that were rejected (to exclude)
    """
    # Reconstruct graph
    G = reconstruct_graph_from_snapshot(graph_snapshot)

    if G.number_of_nodes() == 0:
        print(f"Warning: Empty graph for iteration {iteration}")
        return

    # Build set of rejected claims to exclude
    rejected_ids = set(rejected_claim_ids or [])

    # For current iteration, also add newly rejected claims
    for v in verification_results:
        if not v.get("verdict", True):
            rejected_ids.add(v["claim_id"])

    # Filter nodes if needed
    if show_only_grounded or rejected_ids:
        nodes_to_remove = []
        for node_id in G.nodes():
            node_type = G.nodes[node_id].get("type", "unknown")
            # Keep all source nodes (priors, responses)
            if node_type in ["prior", "response"]:
                continue
            # Remove rejected claims
            if node_id in rejected_ids:
                nodes_to_remove.append(node_id)
                continue
            # For final graph, only keep grounded claims
            if show_only_grounded and node_id not in prior_entailed:
                nodes_to_remove.append(node_id)

        for node_id in nodes_to_remove:
            G.remove_node(node_id)

    if G.number_of_nodes() == 0:
        print(f"Warning: No nodes to display for iteration {iteration}")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Separate nodes by type for layout
    source_nodes = []
    claim_nodes = []

    for node_id in G.nodes():
        node_type = G.nodes[node_id].get("type", "unknown")
        if node_type in ["prior", "response"]:
            source_nodes.append(node_id)
        else:
            claim_nodes.append(node_id)

    # Use bipartite layout if we have both types
    if source_nodes and claim_nodes:
        pos = {}

        # Calculate vertical spacing for source nodes (more space between them)
        n_sources = len(source_nodes)
        source_spacing = 1.5  # Increased spacing between source nodes
        source_height = n_sources * source_spacing

        for i, node_id in enumerate(sorted(source_nodes)):
            pos[node_id] = (0, source_height - i * source_spacing)

        # Calculate vertical spacing for claim nodes (more space between them)
        n_claims = len(claim_nodes)
        claim_spacing = 0.8  # Spacing between claim nodes
        claim_height = n_claims * claim_spacing

        # Center claims vertically relative to sources
        claim_offset = (source_height - claim_height) / 2

        for i, node_id in enumerate(sorted(claim_nodes)):
            pos[node_id] = (3, claim_height - i * claim_spacing + claim_offset)
    else:
        # Fallback to spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw edges with weights
    edges_to_draw = list(G.edges())
    if edges_to_draw:
        edge_weights = [G[u][v].get("weight", 1.0) for u, v in edges_to_draw]
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_widths = [0.5 + 2.0 * w / max_weight for w in edge_weights]

        # Color edges by link type
        edge_colors = []
        for u, v in edges_to_draw:
            link_type = G[u][v].get("link_type", "neutral")
            if link_type == "anchor":
                edge_colors.append("#27AE60")  # Green for anchor
            elif link_type == "agree":
                edge_colors.append("#2980B9")  # Blue for agree
            elif link_type == "contra":
                edge_colors.append("#C0392B")  # Red for contra
            else:
                edge_colors.append("#7F8C8D")  # Gray for neutral

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.6
        )

    # Draw nodes by type
    for node_id in G.nodes():
        node_attrs = G.nodes[node_id]
        color = get_node_color(node_id, node_attrs, prior_entailed, verification_results)
        node_type = node_attrs.get("type", "claim")
        size = get_node_size(node_type)

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=[node_id],
            node_color=[color],
            node_size=size,
            alpha=0.9
        )

    # Draw labels with display IDs
    labels = {}
    for node_id in G.nodes():
        node_attrs = G.nodes[node_id]
        node_type = node_attrs.get("type", "unknown")
        labels[node_id] = get_node_display_id(node_id, node_type, iteration)

    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=8,
        font_weight='bold'
    )

    # Create legend
    legend_patches = [
        mpatches.Patch(color=NODE_COLORS["prior"], label="Prior (P_i)"),
        mpatches.Patch(color=NODE_COLORS["response"], label="Response (R_i)"),
        mpatches.Patch(color=NODE_COLORS["expert_verified"], label="Expert Verified (E_i)"),
        mpatches.Patch(color=NODE_COLORS["claim_grounded"], label="Grounded Claim"),
        mpatches.Patch(color=NODE_COLORS["claim_contested"], label="Contested Claim"),
        mpatches.Patch(color=NODE_COLORS["claim_accepted"], label="Verified & Accepted"),
        mpatches.Patch(color=NODE_COLORS["claim_rejected"], label="Verified & Rejected"),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9)

    # Set title - use paper_id only
    if title is None:
        title = f"{paper_id} - Iteration {iteration}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Remove axes
    ax.axis('off')

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved graph visualization: {output_path}")


def visualize_paper_graphs(
    paper_result: Dict,
    output_dir: str
):
    """
    Visualize all iteration graphs for a single paper.

    Args:
        paper_result: Paper result dict from results.json
        output_dir: Directory to save visualizations
    """
    paper_id = paper_result["paper_id"]

    # Create vis directory
    vis_dir = Path(output_dir) / "vis" / paper_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    iterations = paper_result.get("iterations", [])

    # Track rejected claims across iterations
    cumulative_rejected_ids = []

    for it_result in iterations:
        iteration = it_result["iteration"]
        graph_snapshot = it_result.get("graph_snapshot")

        if graph_snapshot is None:
            print(f"No graph snapshot for {paper_id} iteration {iteration}")
            continue

        # Get prior entailed claims
        prior_entailed = graph_snapshot.get("prior_entailed_claims", [])

        # Get verification results for this iteration
        verification_results = it_result.get("verification_results", [])

        # Generate output path
        output_path = vis_dir / f"iteration_{iteration}.png"

        # Visualize (excluding previously rejected claims)
        visualize_iteration_graph(
            graph_snapshot=graph_snapshot,
            iteration=iteration,
            paper_id=paper_id,
            prior_entailed=prior_entailed,
            verification_results=verification_results,
            output_path=str(output_path),
            title=f"{paper_id} - Iteration {iteration}",
            rejected_claim_ids=cumulative_rejected_ids.copy()
        )

        # Add newly rejected claims to cumulative list
        for v in verification_results:
            if not v.get("verdict", True):
                cumulative_rejected_ids.append(v["claim_id"])

    # Also visualize final graph if available
    final_graph = paper_result.get("final_graph")
    if final_graph:
        prior_entailed = final_graph.get("prior_entailed_claims", [])
        output_path = vis_dir / "final_graph.png"

        # Get all verification results from all iterations
        all_verification_results = []
        for it_result in iterations:
            all_verification_results.extend(it_result.get("verification_results", []))

        # For final graph, only show grounded/verified claims
        visualize_iteration_graph(
            graph_snapshot=final_graph,
            iteration=len(iterations),
            paper_id=paper_id,
            prior_entailed=prior_entailed,
            verification_results=all_verification_results,
            output_path=str(output_path),
            title=f"{paper_id} - Final Graph",
            show_only_grounded=True,
            rejected_claim_ids=cumulative_rejected_ids
        )


def visualize_results_file(results_path: str, output_dir: Optional[str] = None):
    """
    Visualize all graphs from a results.json file.

    Args:
        results_path: Path to results.json
        output_dir: Output directory (defaults to same as results file)
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = str(Path(results_path).parent)

    individual_results = results.get("individual_results", [])

    print(f"Visualizing graphs for {len(individual_results)} papers...")

    for paper_result in individual_results:
        try:
            visualize_paper_graphs(paper_result, output_dir)
        except Exception as e:
            print(f"Error visualizing {paper_result.get('paper_id', 'unknown')}: {e}")

    print(f"Visualizations saved to: {output_dir}/vis/")


# ==============================================================================
# ALTERNATIVE LAYOUTS
# ==============================================================================

def visualize_hierarchical(
    graph_snapshot: Dict,
    iteration: int,
    paper_id: str,
    prior_entailed: List[str],
    verification_results: List[Dict],
    output_path: str,
    figsize: Tuple[int, int] = (20, 14)
):
    """
    Visualize with hierarchical layout (source nodes on top, claims below).
    """
    G = reconstruct_graph_from_snapshot(graph_snapshot)

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Categorize nodes
    prior_nodes = []
    response_nodes = []
    expert_nodes = []
    claim_nodes = []

    for node_id in G.nodes():
        node_attrs = G.nodes[node_id]
        node_type = node_attrs.get("type", "unknown")

        if node_type == "prior":
            if "verification_round" in node_attrs:
                expert_nodes.append(node_id)
            else:
                prior_nodes.append(node_id)
        elif node_type == "response":
            response_nodes.append(node_id)
        else:
            claim_nodes.append(node_id)

    # Calculate positions
    pos = {}

    # Top row: Priors
    y_prior = 3
    for i, node_id in enumerate(sorted(prior_nodes)):
        x = i - len(prior_nodes) / 2
        pos[node_id] = (x, y_prior)

    # Second row: Responses
    y_response = 2
    for i, node_id in enumerate(sorted(response_nodes)):
        x = (i - len(response_nodes) / 2) * 0.8
        pos[node_id] = (x, y_response)

    # Third row: Expert verified
    y_expert = 1.5
    for i, node_id in enumerate(sorted(expert_nodes)):
        x = (i - len(expert_nodes) / 2) * 0.6
        pos[node_id] = (x, y_expert)

    # Bottom: Claims (spread out)
    y_claim = 0
    n_claims = len(claim_nodes)
    for i, node_id in enumerate(sorted(claim_nodes)):
        x = (i - n_claims / 2) * 0.3
        pos[node_id] = (x, y_claim)

    # Draw edges
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        link_type = G[u][v].get("link_type", "neutral")
        weight = G[u][v].get("weight", 1.0)

        if link_type == "anchor":
            edge_colors.append("#27AE60")
        elif link_type == "agree":
            edge_colors.append("#2980B9")
        elif link_type == "contra":
            edge_colors.append("#C0392B")
        else:
            edge_colors.append("#BDC3C7")

        edge_widths.append(0.3 + weight * 0.5)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.5,
        connectionstyle="arc3,rad=0.1"
    )

    # Draw nodes
    for node_id in G.nodes():
        node_attrs = G.nodes[node_id]
        color = get_node_color(node_id, node_attrs, prior_entailed, verification_results)
        node_type = node_attrs.get("type", "claim")
        size = get_node_size(node_type)

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=[node_id],
            node_color=[color],
            node_size=size,
            alpha=0.9
        )

    # Labels
    labels = {node_id: get_node_display_id(node_id, G.nodes[node_id].get("type", ""), iteration)
              for node_id in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight='bold')

    # Legend
    legend_patches = [
        mpatches.Patch(color=NODE_COLORS["prior"], label="Prior (P_i)"),
        mpatches.Patch(color=NODE_COLORS["response"], label="Response (R_i)"),
        mpatches.Patch(color=NODE_COLORS["expert_verified"], label="Expert Verified (E_i)"),
        mpatches.Patch(color=NODE_COLORS["claim_grounded"], label="Grounded"),
        mpatches.Patch(color=NODE_COLORS["claim_contested"], label="Contested"),
        mpatches.Patch(color=NODE_COLORS["claim_accepted"], label="Accepted"),
        mpatches.Patch(color=NODE_COLORS["claim_rejected"], label="Rejected"),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

    ax.set_title(f"Knowledge Graph - {paper_id} - Iteration {iteration}", fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python graph_visualization.py <results.json>")
        sys.exit(1)

    results_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    visualize_results_file(results_path, output_dir)
