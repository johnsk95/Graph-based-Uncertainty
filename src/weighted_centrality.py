"""
Weighted Centrality Calculations for Phase 1
Implements weighted closeness centrality using edge costs (1/weight)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple


def calculate_weighted_closeness_centrality(edge_weights: np.ndarray,
                                            num_priors: int,
                                            num_responses: int,
                                            num_claims: int) -> Dict[str, np.ndarray]:
    """
    Calculate weighted closeness centrality for claims in bipartite graph.

    Uses Dijkstra's algorithm to compute shortest paths with edge costs = 1/weight.
    Implements the formula from Appendix A:

        CC(v) = (N-1) / Σ dW(v,u) · |Vv| / N

    where dW(v,u) is the weighted shortest path distance.

    Args:
        edge_weights: Matrix of shape (num_priors + num_responses, num_claims)
                     with edge weights
        num_priors: Number of prior nodes
        num_responses: Number of response nodes
        num_claims: Number of claim nodes

    Returns:
        Dictionary with:
            - 'weighted_closeness': Array of closeness scores for each claim
            - 'avg_distance_to_sources': Average distance from each claim to all sources
            - 'connected_component_size': Size of connected component for each claim
    """
    # Total number of source nodes (priors + responses)
    num_sources = num_priors + num_responses

    # Total number of nodes in graph
    N = num_sources + num_claims

    # Build NetworkX graph for efficient shortest path computation
    G = nx.Graph()

    # Add nodes
    source_nodes = [f"S{i}" for i in range(num_sources)]  # S0 = prior, S1...SN = responses
    claim_nodes = [f"C{i}" for i in range(num_claims)]

    G.add_nodes_from(source_nodes, bipartite=0)
    G.add_nodes_from(claim_nodes, bipartite=1)

    # Add weighted edges
    # Edge cost = 1/weight (lower weight = higher cost)
    for source_idx in range(num_sources):
        for claim_idx in range(num_claims):
            weight = edge_weights[source_idx, claim_idx]
            if weight > 0:  # Only add edge if weight > 0
                cost = 1.0 / weight
                G.add_edge(f"S{source_idx}", f"C{claim_idx}", weight=cost)

    # Calculate weighted closeness centrality for each claim
    closeness_scores = np.zeros(num_claims)
    avg_distances = np.zeros(num_claims)
    component_sizes = np.zeros(num_claims)

    for claim_idx in range(num_claims):
        claim_node = f"C{claim_idx}"

        # Get connected component containing this claim
        if claim_node in G:
            connected_component = nx.node_connected_component(G, claim_node)
            component_size = len(connected_component)
            component_sizes[claim_idx] = component_size

            # Compute shortest paths from this claim to all other nodes in component
            try:
                # Use Dijkstra's algorithm with edge costs
                lengths = nx.single_source_dijkstra_path_length(G, claim_node, weight='weight')

                # Sum distances to all reachable nodes
                total_distance = sum(lengths.values())

                # Average distance to source nodes only
                source_distances = [lengths.get(f"S{i}", float('inf'))
                                  for i in range(num_sources)
                                  if f"S{i}" in lengths]
                avg_distances[claim_idx] = np.mean(source_distances) if source_distances else float('inf')

                # Closeness centrality formula (modified for disconnected graphs)
                if total_distance > 0:
                    closeness = ((N - 1) / total_distance) * (component_size / N)
                else:
                    closeness = 0.0

                closeness_scores[claim_idx] = closeness

            except nx.NetworkXError:
                # Node is isolated
                closeness_scores[claim_idx] = 0.0
                avg_distances[claim_idx] = float('inf')
        else:
            # Claim has no edges
            closeness_scores[claim_idx] = 0.0
            avg_distances[claim_idx] = float('inf')
            component_sizes[claim_idx] = 1

    return {
        'weighted_closeness': closeness_scores,
        'avg_distance_to_sources': avg_distances,
        'connected_component_size': component_sizes
    }


def calculate_additional_centralities(edge_weights: np.ndarray,
                                      num_priors: int,
                                      num_responses: int,
                                      num_claims: int) -> Dict[str, np.ndarray]:
    """
    Calculate additional centrality metrics for the weighted graph.

    Args:
        edge_weights: Matrix of edge weights
        num_priors: Number of prior nodes
        num_responses: Number of response nodes
        num_claims: Number of claim nodes

    Returns:
        Dictionary with various centrality metrics for claims
    """
    num_sources = num_priors + num_responses

    # Build graph
    G = nx.Graph()
    source_nodes = [f"S{i}" for i in range(num_sources)]
    claim_nodes = [f"C{i}" for i in range(num_claims)]

    G.add_nodes_from(source_nodes, bipartite=0)
    G.add_nodes_from(claim_nodes, bipartite=1)

    # Add edges with costs
    for source_idx in range(num_sources):
        for claim_idx in range(num_claims):
            weight = edge_weights[source_idx, claim_idx]
            if weight > 0:
                cost = 1.0 / weight
                G.add_edge(f"S{source_idx}", f"C{claim_idx}", weight=cost)

    # Calculate various centrality metrics
    try:
        # Betweenness centrality (weighted)
        betweenness = nx.betweenness_centrality(G, weight='weight')
        betweenness_scores = np.array([betweenness.get(f"C{i}", 0.0) for i in range(num_claims)])
    except:
        betweenness_scores = np.zeros(num_claims)

    try:
        # Eigenvector centrality (use weight as strength, not cost)
        G_strength = G.copy()
        for u, v, data in G_strength.edges(data=True):
            G_strength[u][v]['weight'] = 1.0 / data['weight']  # Convert cost back to weight

        eigenvector = nx.eigenvector_centrality(G_strength, weight='weight', max_iter=1000)
        eigenvector_scores = np.array([eigenvector.get(f"C{i}", 0.0) for i in range(num_claims)])
    except:
        eigenvector_scores = np.zeros(num_claims)

    try:
        # PageRank (weighted)
        pagerank = nx.pagerank(G, weight='weight')
        pagerank_scores = np.array([pagerank.get(f"C{i}", 0.0) for i in range(num_claims)])
    except:
        pagerank_scores = np.zeros(num_claims)

    # Degree centrality (count of non-zero edges)
    degree_scores = np.sum(edge_weights > 0, axis=0)

    # Weighted degree (sum of weights)
    weighted_degree_scores = np.sum(edge_weights, axis=0)

    return {
        'betweenness_centrality': betweenness_scores,
        'eigenvector_centrality': eigenvector_scores,
        'pagerank': pagerank_scores,
        'degree_centrality': degree_scores,
        'weighted_degree': weighted_degree_scores
    }


def identify_claim_categories(closeness_scores: np.ndarray,
                              delta_true: float = 0.6,
                              delta_false: float = 0.2) -> Dict[str, List[int]]:
    """
    Categorize claims based on weighted closeness centrality thresholds.

    Args:
        closeness_scores: Array of closeness centrality scores
        delta_true: High threshold for grounded/true claims
        delta_false: Low threshold for contradictory/false claims

    Returns:
        Dictionary with:
            - 'grounded': Indices of grounded claims (high centrality)
            - 'boundary': Indices of boundary/contested claims (intermediate)
            - 'contradictory': Indices of contradictory claims (low centrality)
    """
    grounded_indices = [i for i, score in enumerate(closeness_scores) if score >= delta_true]
    contradictory_indices = [i for i, score in enumerate(closeness_scores) if score <= delta_false]
    boundary_indices = [i for i, score in enumerate(closeness_scores)
                       if delta_false < score < delta_true]

    return {
        'grounded': grounded_indices,
        'boundary': boundary_indices,
        'contradictory': contradictory_indices
    }


def compute_all_weighted_metrics(edge_weights: np.ndarray,
                                 num_priors: int,
                                 num_responses: int,
                                 num_claims: int,
                                 delta_true: float = 0.6,
                                 delta_false: float = 0.2) -> Dict:
    """
    Compute all weighted centrality metrics and categorize claims.

    Args:
        edge_weights: Edge weight matrix
        num_priors: Number of prior nodes
        num_responses: Number of response nodes
        num_claims: Number of claim nodes
        delta_true: Threshold for grounded claims
        delta_false: Threshold for contradictory claims

    Returns:
        Dictionary with all metrics and categorizations
    """
    # Weighted closeness centrality (primary metric)
    closeness_results = calculate_weighted_closeness_centrality(
        edge_weights, num_priors, num_responses, num_claims
    )

    # Additional centrality metrics
    additional_metrics = calculate_additional_centralities(
        edge_weights, num_priors, num_responses, num_claims
    )

    # Categorize claims
    categories = identify_claim_categories(
        closeness_results['weighted_closeness'],
        delta_true,
        delta_false
    )

    # Combine all results
    return {
        **closeness_results,
        **additional_metrics,
        'claim_categories': categories
    }


if __name__ == "__main__":
    # Test with a simple example
    print("Testing weighted centrality calculations...")

    # Example: 1 prior, 3 responses, 5 claims
    num_priors = 1
    num_responses = 3
    num_claims = 5

    # Create sample edge weights
    edge_weights = np.array([
        # Prior → Claims: [C0, C1, C2, C3, C4]
        [3.0, 3.0, 0.0, 0.0, 0.0],  # Prior supports C0, C1 (anchor)
        # Response 0
        [2.0, 2.0, 1.0, 0.5, 0.0],  # Agrees with C0,C1, neutral C2, contradicts C3
        # Response 1
        [2.0, 0.0, 1.0, 1.0, 0.0],  # Agrees with C0, neutral C2,C3
        # Response 2
        [2.0, 2.0, 1.0, 0.0, 1.0],  # Agrees with C0,C1, neutral C2,C4
    ])

    # Compute metrics
    results = compute_all_weighted_metrics(
        edge_weights, num_priors, num_responses, num_claims
    )

    print("\nWeighted Closeness Centrality:")
    for i, score in enumerate(results['weighted_closeness']):
        print(f"  Claim {i}: {score:.4f}")

    print("\nClaim Categories:")
    print(f"  Grounded: {results['claim_categories']['grounded']}")
    print(f"  Boundary: {results['claim_categories']['boundary']}")
    print(f"  Contradictory: {results['claim_categories']['contradictory']}")

    print("\nWeighted Degree:")
    for i, degree in enumerate(results['weighted_degree']):
        print(f"  Claim {i}: {degree:.2f}")
