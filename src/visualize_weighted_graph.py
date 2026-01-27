"""
Visualization module for weighted bipartite graphs
Creates color-coded graph visualizations with edge weight distinctions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import os


class WeightedGraphVisualizer:
    """Visualizes weighted bipartite graphs with color-coded edge types."""

    # Color scheme for different edge types
    EDGE_COLORS = {
        'anchor': '#FF0000',       # Red - Highest weight
        'agreement': '#FFA500',    # Orange - High weight
        'neutral': '#808080',      # Gray - Standard weight
        'contradictory': '#0000FF',  # Blue - Low weight
        'none': '#FFFFFF'          # White - No edge (shouldn't be visible)
    }

    EDGE_WIDTHS = {
        'anchor': 4.0,
        'agreement': 3.0,
        'neutral': 2.0,
        'contradictory': 1.5,
        'none': 0.5
    }

    EDGE_STYLES = {
        'anchor': 'solid',
        'agreement': 'solid',
        'neutral': 'dashed',
        'contradictory': 'dotted',
        'none': 'dotted'
    }

    def __init__(self):
        """Initialize the visualizer."""
        pass

    def visualize_instance(self,
                          instance_data: Dict,
                          output_path: str,
                          max_claims: int = 20,
                          figsize: Tuple[int, int] = (20, 12),
                          show_edge_labels: bool = False):
        """
        Visualize a single instance's weighted bipartite graph.

        Args:
            instance_data: Dictionary containing weighted graph data
            output_path: Path to save the visualization
            max_claims: Maximum number of claims to display (for readability)
            figsize: Figure size (width, height)
            show_edge_labels: Whether to show edge weights on edges
        """
        # Extract data
        weighted_graph = instance_data.get('weighted_graph', {})
        edge_weights = np.array(weighted_graph.get('edge_weights', []))
        edge_types = weighted_graph.get('edge_types', [])
        claims = instance_data.get('breakdown', [])
        prior_claims = instance_data.get('prior_claims', [])
        all_generations = instance_data.get('all_generations', [])

        num_priors = weighted_graph.get('num_priors', 0)
        num_responses = weighted_graph.get('num_responses', 0)
        num_claims = min(weighted_graph.get('num_claims', 0), max_claims)

        # Truncate if too many claims
        if len(claims) > max_claims:
            claims = claims[:max_claims]
            edge_weights = edge_weights[:, :max_claims]
            edge_types = [row[:max_claims] for row in edge_types]

        # Create graph
        G = nx.Graph()

        # Add nodes
        # Prior nodes
        prior_nodes = [f"P{i}" for i in range(num_priors)]
        for node in prior_nodes:
            G.add_node(node, bipartite=0, node_type='prior')

        # Response nodes
        response_nodes = [f"R{i}" for i in range(num_responses)]
        for node in response_nodes:
            G.add_node(node, bipartite=0, node_type='response')

        # Claim nodes
        claim_nodes = [f"C{i}" for i in range(num_claims)]
        for node in claim_nodes:
            G.add_node(node, bipartite=1, node_type='claim')

        # Add edges with attributes
        edges_by_type = {etype: [] for etype in self.EDGE_COLORS.keys()}

        for i in range(num_priors + num_responses):
            for j in range(num_claims):
                weight = edge_weights[i, j]
                if weight > 0:  # Only add edges with positive weight
                    edge_type = edge_types[i][j]

                    if i < num_priors:
                        source = f"P{i}"
                    else:
                        source = f"R{i - num_priors}"

                    target = f"C{j}"

                    G.add_edge(source, target, weight=weight, edge_type=edge_type)
                    edges_by_type[edge_type].append((source, target, weight))

        # Create layout
        pos = self._create_bipartite_layout(
            prior_nodes, response_nodes, claim_nodes, num_claims
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw nodes
        # Prior nodes (red squares)
        if prior_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=prior_nodes,
                node_color='#FFB3B3',
                node_shape='s',
                node_size=3000,
                edgecolors='red',
                linewidths=3,
                ax=ax
            )

        # Response nodes (blue squares)
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=response_nodes,
            node_color='#B3D9FF',
            node_shape='s',
            node_size=2000,
            edgecolors='blue',
            linewidths=2,
            ax=ax
        )

        # Claim nodes (circles, colored by category)
        pointwise_dict = instance_data.get('pointwise_dict', [])
        claim_colors = []
        for i in range(num_claims):
            if i < len(pointwise_dict):
                category = pointwise_dict[i].get('category', 'unknown')
                if category == 'grounded':
                    claim_colors.append('#90EE90')  # Light green
                elif category == 'boundary':
                    claim_colors.append('#FFFFE0')  # Light yellow
                elif category == 'contradictory':
                    claim_colors.append('#FFB3B3')  # Light red
                else:
                    claim_colors.append('#FFFFFF')  # White
            else:
                claim_colors.append('#FFFFFF')

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=claim_nodes,
            node_color=claim_colors,
            node_shape='o',
            node_size=1500,
            edgecolors='black',
            linewidths=1.5,
            ax=ax
        )

        # Draw edges by type
        for edge_type, edges in edges_by_type.items():
            if edges and edge_type != 'none':
                edge_list = [(e[0], e[1]) for e in edges]
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edge_list,
                    edge_color=self.EDGE_COLORS[edge_type],
                    width=self.EDGE_WIDTHS[edge_type],
                    style=self.EDGE_STYLES[edge_type],
                    alpha=0.6,
                    ax=ax
                )

        # Draw node labels
        # Source node labels (Prior, R0, R1, etc.)
        source_labels = {}
        for node in prior_nodes:
            source_labels[node] = "Prior"
        for i, node in enumerate(response_nodes):
            source_labels[node] = f"R{i}"

        nx.draw_networkx_labels(
            G, pos,
            labels=source_labels,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        # Claim node labels (C0, C1, etc.)
        claim_labels = {f"C{i}": f"C{i}" for i in range(num_claims)}
        nx.draw_networkx_labels(
            G, pos,
            labels=claim_labels,
            font_size=8,
            ax=ax
        )

        # Add edge weight labels if requested
        if show_edge_labels:
            edge_labels = {}
            for edge_type, edges in edges_by_type.items():
                for source, target, weight in edges:
                    edge_labels[(source, target)] = f"{weight:.1f}"

            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=6,
                ax=ax
            )

        # Create legend
        legend_elements = [
            # Edge types
            mpatches.Patch(color=self.EDGE_COLORS['anchor'], label='Anchor (w=3.0): Prior → Prior claim'),
            mpatches.Patch(color=self.EDGE_COLORS['agreement'], label='Agreement (w=2.0): Response → Prior claim'),
            mpatches.Patch(color=self.EDGE_COLORS['neutral'], label='Neutral (w=1.0): Response → Novel claim'),
            mpatches.Patch(color=self.EDGE_COLORS['contradictory'], label='Contradictory (w=0.5): Response → Conflicting claim'),
            mpatches.Patch(color='white', label=''),  # Spacer
            # Node types
            mpatches.Patch(facecolor='#FFB3B3', edgecolor='red', label='Prior nodes (WikiData)'),
            mpatches.Patch(facecolor='#B3D9FF', edgecolor='blue', label='Response nodes (LLM)'),
            mpatches.Patch(color='white', label=''),  # Spacer
            # Claim categories
            mpatches.Patch(color='#90EE90', label='Grounded claims (high centrality)'),
            mpatches.Patch(color='#FFFFE0', label='Boundary claims (intermediate)'),
            mpatches.Patch(color='#FFB3B3', label='Contradictory claims (low centrality)'),
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True
        )

        # Add title with statistics
        entity = instance_data.get('entity', 'Unknown')
        title = f"Weighted Bipartite Graph: {entity}\n"
        title += f"Priors: {num_priors} | Responses: {num_responses} | Claims: {num_claims}\n"
        title += f"Edge counts: Anchor={len(edges_by_type['anchor'])}, "
        title += f"Agreement={len(edges_by_type['agreement'])}, "
        title += f"Neutral={len(edges_by_type['neutral'])}, "
        title += f"Contradictory={len(edges_by_type['contradictory'])}"

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

        # Add claim text on the right side
        claim_text_x = 1.25
        claim_text_y_start = 0.95
        claim_text_y_step = 0.8 / max(num_claims, 1)

        for i in range(num_claims):
            claim_text = claims[i]
            if len(claim_text) > 80:
                claim_text = claim_text[:77] + "..."

            # Get category for coloring
            category = 'unknown'
            if i < len(pointwise_dict):
                category = pointwise_dict[i].get('category', 'unknown')
                closeness = pointwise_dict[i].get('weighted_closeness', 0)
                claim_text = f"C{i}: {claim_text}\n     (closeness={closeness:.3f}, {category})"

            color = 'black'
            if category == 'grounded':
                color = 'green'
            elif category == 'boundary':
                color = 'orange'
            elif category == 'contradictory':
                color = 'red'

            fig.text(
                claim_text_x, claim_text_y_start - i * claim_text_y_step,
                claim_text,
                fontsize=8,
                verticalalignment='top',
                color=color,
                wrap=True
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization to: {output_path}")

    def _create_bipartite_layout(self,
                                  prior_nodes: List[str],
                                  response_nodes: List[str],
                                  claim_nodes: List[str],
                                  num_claims: int) -> Dict:
        """
        Create a bipartite layout with priors and responses on the left,
        claims on the right.

        Args:
            prior_nodes: List of prior node IDs
            response_nodes: List of response node IDs
            claim_nodes: List of claim node IDs
            num_claims: Number of claims

        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        pos = {}

        # Source nodes (priors + responses) on the left
        num_sources = len(prior_nodes) + len(response_nodes)
        source_y_positions = np.linspace(1, 0, num_sources)

        # Prior nodes at the top
        for i, node in enumerate(prior_nodes):
            pos[node] = (0, source_y_positions[i])

        # Response nodes below
        for i, node in enumerate(response_nodes):
            pos[node] = (0, source_y_positions[len(prior_nodes) + i])

        # Claim nodes on the right
        claim_y_positions = np.linspace(1, 0, num_claims)
        for i, node in enumerate(claim_nodes):
            pos[node] = (1, claim_y_positions[i])

        return pos

    def visualize_batch(self,
                       sequences: List[Dict],
                       output_dir: str,
                       max_instances: int = 10,
                       max_claims: int = 20):
        """
        Visualize multiple instances in batch.

        Args:
            sequences: List of instance dictionaries
            output_dir: Directory to save visualizations
            max_instances: Maximum number of instances to visualize
            max_claims: Maximum claims per graph
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating visualizations for {min(len(sequences), max_instances)} instances...")

        for i, instance in enumerate(sequences[:max_instances]):
            entity = instance.get('entity', f'instance_{i}')
            # Clean entity name for filename
            safe_entity = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in entity)
            safe_entity = safe_entity.replace(' ', '_')[:50]

            output_path = os.path.join(output_dir, f"weighted_graph_{i}_{safe_entity}.png")

            print(f"  [{i+1}/{min(len(sequences), max_instances)}] Visualizing: {entity}")

            try:
                self.visualize_instance(
                    instance,
                    output_path,
                    max_claims=max_claims,
                    show_edge_labels=False
                )
            except Exception as e:
                print(f"    Error visualizing instance {i}: {e}")
                continue

        print(f"\n✓ Visualizations saved to: {output_dir}")


def visualize_weighted_graphs(sequences: List[Dict],
                              output_dir: str,
                              max_instances: int = 10,
                              max_claims: int = 20):
    """
    Main function to visualize weighted bipartite graphs.

    Args:
        sequences: List of sequence dictionaries with weighted graph data
        output_dir: Directory to save visualizations
        max_instances: Maximum number of instances to visualize
        max_claims: Maximum claims to show per graph
    """
    visualizer = WeightedGraphVisualizer()
    visualizer.visualize_batch(sequences, output_dir, max_instances, max_claims)


if __name__ == "__main__":
    # Test visualization with mock data
    import json

    print("Testing weighted graph visualization...")

    test_instance = {
        'entity': 'Test Example',
        'all_generations': ['Response 1', 'Response 2'],
        'prior_claims': ['Prior claim 1', 'Prior claim 2'],
        'breakdown': [
            'Claim 1',
            'Claim 2',
            'Claim 3',
            'Claim 4'
        ],
        'weighted_graph': {
            'edge_weights': [
                [3.0, 3.0, 0.0, 0.0],  # Prior
                [2.0, 0.0, 1.0, 0.5],  # Response 0
                [2.0, 2.0, 1.0, 0.0],  # Response 1
            ],
            'edge_types': [
                ['anchor', 'anchor', 'none', 'none'],
                ['agreement', 'none', 'neutral', 'contradictory'],
                ['agreement', 'agreement', 'neutral', 'none']
            ],
            'num_priors': 1,
            'num_responses': 2,
            'num_claims': 4
        },
        'pointwise_dict': [
            {'claim': 'Claim 1', 'category': 'grounded', 'weighted_closeness': 0.85},
            {'claim': 'Claim 2', 'category': 'grounded', 'weighted_closeness': 0.75},
            {'claim': 'Claim 3', 'category': 'boundary', 'weighted_closeness': 0.45},
            {'claim': 'Claim 4', 'category': 'boundary', 'weighted_closeness': 0.30},
        ]
    }

    visualizer = WeightedGraphVisualizer()
    visualizer.visualize_instance(test_instance, 'test_weighted_graph.png', max_claims=20)
    print("\n✓ Test visualization saved to: test_weighted_graph.png")
