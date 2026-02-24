"""
Iterative Knowledge Expansion Module (Section 4.4)

Implements the informative claim selection algorithm for human-in-the-loop
knowledge expansion. This module selects the most valuable contested claims
for human/expert verification based on a utility-based ranking approach.

Key components:
1. Priority Score = λ1 * S_centrality + λ2 * S_novelty
2. Cost = Embedding distance from claim to priors
3. Utility = PriorityScore × (1 - σ(CC)) - Cost
4. Selection = Top-k claims by utility
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
from sklearn.metrics.pairwise import cosine_distances


# ==============================================================================
# MODULE 1: PRIORITY SCORE COMPUTATION
# ==============================================================================

def compute_centrality_scores(
    graph: nx.Graph,
    contested_claim_ids: List[str],
    response_node_ids: List[str]
) -> Dict[str, float]:
    """
    Compute betweenness centrality for contested claims on the contested subgraph.

    The contested subgraph contains only response nodes and contested claim nodes,
    excluding priors and prior-entailed claims. This identifies "source" claims
    that are central to clusters of unverified claims.

    Args:
        graph: NetworkX graph (full bipartite graph)
        contested_claim_ids: List of contested claim node IDs
        response_node_ids: List of response node IDs

    Returns:
        centrality_scores: Dict mapping claim_id -> raw betweenness centrality
    """
    # Build contested subgraph (responses + contested claims only)
    subgraph_nodes = set(contested_claim_ids) | set(response_node_ids)

    # Filter to only nodes that exist in the graph
    existing_nodes = [n for n in subgraph_nodes if n in graph]

    if len(existing_nodes) == 0:
        return {cid: 0.0 for cid in contested_claim_ids}

    G_contested = graph.subgraph(existing_nodes).copy()

    # Compute betweenness centrality on contested subgraph
    # Use weight='cost' for weighted version (lower cost = higher weight path)
    try:
        if G_contested.number_of_edges() > 0:
            betweenness = nx.betweenness_centrality(G_contested, weight='cost', normalized=True)
        else:
            betweenness = {n: 0.0 for n in G_contested.nodes()}
    except Exception:
        betweenness = {n: 0.0 for n in G_contested.nodes()}

    # Extract only claim nodes
    centrality_scores = {cid: betweenness.get(cid, 0.0) for cid in contested_claim_ids}

    return centrality_scores


def compute_novelty_scores(
    graph: nx.Graph,
    contested_claim_ids: List[str],
    prior_entailed_claim_ids: List[str]
) -> Dict[str, Optional[float]]:
    """
    Compute graph-based novelty as shortest path distance to nearest prior-entailed claim.

    Novel claims are those that are far from any known/grounded knowledge. This
    identifies claims at the "frontier" of the knowledge boundary.

    Uses weighted distance where edge cost = 1 / edge_weight.

    Args:
        graph: NetworkX graph with edge weights (must have 'cost' attribute)
        contested_claim_ids: List of contested claim node IDs
        prior_entailed_claim_ids: List of prior-entailed claim node IDs (C_P)

    Returns:
        novelty_scores: Dict mapping claim_id -> raw distance score (None if disconnected)
    """
    novelty_scores = {}

    for cid in contested_claim_ids:
        if cid not in graph:
            novelty_scores[cid] = None
            continue

        min_distance = float('inf')

        for pid in prior_entailed_claim_ids:
            if pid not in graph:
                continue
            try:
                # Use 'cost' attribute (1/weight) for weighted shortest path
                distance = nx.shortest_path_length(
                    graph, source=cid, target=pid, weight='cost'
                )
                min_distance = min(min_distance, distance)
            except nx.NetworkXNoPath:
                continue

        # Handle disconnected claims
        if min_distance == float('inf'):
            novelty_scores[cid] = None  # Will handle in normalization
        else:
            novelty_scores[cid] = min_distance

    return novelty_scores


def z_normalize_positive(scores_dict: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Z-score normalize and shift to ensure all values are positive.

    Handles None values (disconnected claims) by assigning them
    the maximum normalized value + 1 (most novel).

    Args:
        scores_dict: Dict mapping claim_id -> raw score (can contain None)

    Returns:
        normalized_dict: Dict mapping claim_id -> normalized positive score
    """
    # Separate valid and invalid scores
    valid_scores = {k: v for k, v in scores_dict.items() if v is not None}
    invalid_ids = [k for k, v in scores_dict.items() if v is None]

    if len(valid_scores) == 0:
        return {k: 0.0 for k in scores_dict}

    values = np.array(list(valid_scores.values()))
    mean_val = np.mean(values)
    std_val = np.std(values) + 1e-8  # Avoid division by zero

    # Z-score normalization
    z_scores = {k: (v - mean_val) / std_val for k, v in valid_scores.items()}

    # Shift to positive (min becomes 0)
    min_z = min(z_scores.values()) if z_scores else 0.0
    normalized = {k: v - min_z for k, v in z_scores.items()}

    # Assign max value + 1 to disconnected claims (most novel)
    if invalid_ids:
        max_normalized = max(normalized.values()) if normalized else 0.0
        for k in invalid_ids:
            normalized[k] = max_normalized + 1.0

    return normalized


def compute_priority_scores(
    centrality_scores: Dict[str, float],
    novelty_scores: Dict[str, Optional[float]],
    lambda1: float = 1.0,
    lambda2: float = 1.0
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Compute combined priority score for each contested claim.

    PriorityScore(c) = λ1 * S_centrality(c) + λ2 * S_novelty(c)

    Args:
        centrality_scores: Dict of raw centrality scores
        novelty_scores: Dict of raw novelty scores (can contain None)
        lambda1: Weight for centrality component
        lambda2: Weight for novelty component

    Returns:
        Tuple of:
            - priority_scores: Dict mapping claim_id -> combined priority score
            - S_centrality: Dict of normalized centrality scores
            - S_novelty: Dict of normalized novelty scores
    """
    # Normalize both scores
    S_centrality = z_normalize_positive(centrality_scores)
    S_novelty = z_normalize_positive(novelty_scores)

    # Combine with weights
    priority_scores = {}
    for cid in centrality_scores.keys():
        priority_scores[cid] = lambda1 * S_centrality.get(cid, 0.0) + lambda2 * S_novelty.get(cid, 0.0)

    return priority_scores, S_centrality, S_novelty


# ==============================================================================
# MODULE 2: COST COMPUTATION
# ==============================================================================

def compute_embedding_costs(
    claim_embeddings: Dict[int, np.ndarray],
    prior_embeddings: np.ndarray,
    contested_claim_indices: List[int],
    k: int = 5
) -> Dict[int, float]:
    """
    Compute embedding-based verification cost for contested claims.

    Cost is based on cosine distance from claim to priors:
    - Claims similar to priors are easier to verify (lower cost)
    - Claims far from priors require more effort (higher cost)

    Args:
        claim_embeddings: Dict mapping claim_index -> embedding vector (np.array)
        prior_embeddings: np.array of shape (num_priors, embedding_dim)
        contested_claim_indices: List of contested claim indices
        k: Number of nearest priors for averaging

    Returns:
        raw_costs: Dict mapping claim_index -> raw embedding distance
    """
    raw_costs = {}

    if len(prior_embeddings) == 0:
        # No priors - all claims have uniform cost
        return {cid: 1.0 for cid in contested_claim_indices}

    k = min(k, len(prior_embeddings))

    for cid in contested_claim_indices:
        if cid not in claim_embeddings:
            raw_costs[cid] = 1.0  # Default cost
            continue

        claim_emb = claim_embeddings[cid].reshape(1, -1)

        # Compute cosine distances to all priors
        distances = cosine_distances(claim_emb, prior_embeddings)[0]

        # Minimum distance (nearest prior)
        min_dist = np.min(distances)

        # Average of k-nearest
        k_nearest_dists = np.partition(distances, k - 1)[:k]
        avg_k_dist = np.mean(k_nearest_dists)

        # Weighted combination (emphasize k-nearest average)
        raw_costs[cid] = 0.4 * min_dist + 0.6 * avg_k_dist

    return raw_costs


def compute_normalized_costs(
    claim_embeddings: Dict[int, np.ndarray],
    prior_embeddings: np.ndarray,
    contested_claim_indices: List[int],
    k: int = 5
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute z-normalized positive costs.

    Args:
        claim_embeddings: Dict mapping claim_index -> embedding vector
        prior_embeddings: np.array of prior embeddings
        contested_claim_indices: List of contested claim indices
        k: Number of nearest priors for cost computation

    Returns:
        Tuple of (normalized_costs, raw_costs)
    """
    raw_costs = compute_embedding_costs(
        claim_embeddings, prior_embeddings, contested_claim_indices, k
    )
    normalized_costs = z_normalize_positive(raw_costs)

    return normalized_costs, raw_costs


# ==============================================================================
# MODULE 3: CLAIM SELECTION AND ROUTING
# ==============================================================================

def sigmoid(x: float, scale: float = 1.0, shift: float = 0.0) -> float:
    """Sigmoid function to map values to [0, 1] probability."""
    return 1.0 / (1.0 + np.exp(-scale * (x - shift)))


def normalize_closeness_centrality(
    cc_scores: Dict[str, float],
    method: str = 'sigmoid'
) -> Dict[str, float]:
    """
    Normalize closeness centrality to [0, 1] confidence scores.

    Args:
        cc_scores: Dict mapping claim_id -> raw closeness centrality
        method: 'sigmoid' or 'minmax'

    Returns:
        sigma_cc: Dict mapping claim_id -> normalized confidence [0, 1]
    """
    if len(cc_scores) == 0:
        return {}

    values = np.array(list(cc_scores.values()))

    if method == 'minmax':
        min_v, max_v = np.min(values), np.max(values)
        range_v = max_v - min_v + 1e-8
        sigma_cc = {k: (v - min_v) / range_v for k, v in cc_scores.items()}

    elif method == 'sigmoid':
        # Center sigmoid at median CC
        median_cc = np.median(values)
        std_cc = np.std(values) + 1e-8
        scale = 2.0 / std_cc  # Adjust steepness based on spread
        sigma_cc = {k: sigmoid(v, scale=scale, shift=median_cc) for k, v in cc_scores.items()}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return sigma_cc


def compute_utilities(
    priority_scores: Dict[str, float],
    costs: Dict[str, float],
    sigma_cc: Dict[str, float]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute net utility for each contested claim.

    Utility(c) = PriorityScore(c) × (1 - σ(CC(c))) - Cost(c)

    The term (1 - σ(CC)) represents the marginal value of human verification:
    - Claims with high closeness centrality are already well-supported
    - Verifying them provides less new information
    - Claims with low CC benefit more from verification

    Args:
        priority_scores: Dict mapping claim_id -> priority score (normalized)
        costs: Dict mapping claim_id -> normalized cost
        sigma_cc: Dict mapping claim_id -> confidence from CC [0, 1]

    Returns:
        Tuple of:
            - utilities: Dict mapping claim_id -> net utility
            - info_gains: Dict mapping claim_id -> information gain (before cost)
    """
    utilities = {}
    info_gains = {}

    for cid in priority_scores.keys():
        # Information gain: high priority + low confidence = high gain
        confidence = sigma_cc.get(cid, 0.5)
        info_gain = priority_scores[cid] * (1 - confidence)

        # Net utility: information gain minus cost
        cost = costs.get(cid, 0.0)
        net_utility = info_gain - cost

        info_gains[cid] = info_gain
        utilities[cid] = net_utility

    return utilities, info_gains


def select_claims_for_verification(
    contested_claim_ids: List[str],
    closeness_centrality: Dict[str, float],
    priority_scores: Dict[str, float],
    costs: Dict[str, float],
    k: int
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """
    Select top-k contested claims for verification based on utility.

    Args:
        contested_claim_ids: List of contested claim IDs
        closeness_centrality: Dict mapping claim_id -> raw CC score
        priority_scores: Dict mapping claim_id -> priority score (normalized)
        costs: Dict mapping claim_id -> cost (normalized)
        k: Number of claims to select for verification

    Returns:
        Tuple of:
            - selection: Dict with keys 'verify' and 'discard', each containing list of claim_ids
            - diagnostics: Dict with intermediate computations for analysis
    """
    if len(contested_claim_ids) == 0:
        return {'verify': [], 'discard': []}, {}

    # Normalize closeness centrality to confidence scores
    cc_subset = {cid: closeness_centrality.get(cid, 0.0) for cid in contested_claim_ids}
    sigma_cc = normalize_closeness_centrality(cc_subset)

    # Compute utilities
    utilities, info_gains = compute_utilities(priority_scores, costs, sigma_cc)

    # Sort by utility (descending)
    sorted_claims = sorted(contested_claim_ids, key=lambda x: utilities.get(x, 0.0), reverse=True)

    # Select top-k for verification, discard the rest
    k = min(k, len(sorted_claims))
    verify_ids = sorted_claims[:k]
    discard_ids = sorted_claims[k:]

    selection = {
        'verify': verify_ids,
        'discard': discard_ids
    }

    diagnostics = {
        'sigma_cc': sigma_cc,
        'utilities': utilities,
        'info_gains': info_gains,
        'priority_scores': {cid: priority_scores.get(cid, 0.0) for cid in contested_claim_ids},
        'costs': {cid: costs.get(cid, 0.0) for cid in contested_claim_ids}
    }

    return selection, diagnostics


# ==============================================================================
# MODULE 4: MAIN ORCHESTRATION CLASS
# ==============================================================================

class IterativeKnowledgeExpansion:
    """
    Main class for iterative knowledge expansion via informative claim selection.

    This module operates on contested (boundary) claims only. Grounded claims
    have already been collapsed in the previous stage (Claim Collapsing, Section 4.3).

    The algorithm selects claims that are most informative, defined as claims that:
    1. Resolve the most ambiguity (high betweenness centrality in contested subgraph)
    2. Expand the knowledge boundary into novel areas (far from known priors)

    Usage:
        expander = IterativeKnowledgeExpansion(
            graph=G,
            claim_embeddings=claim_emb_dict,
            prior_embeddings=prior_emb_array,
            prior_entailed_claim_ids=grounded_ids,
            response_node_ids=response_ids,
            closeness_centrality=cc_dict
        )
        selection, diagnostics = expander.select_claims(k=20)
    """

    def __init__(
        self,
        graph: nx.Graph,
        claim_embeddings: Dict[int, np.ndarray],
        prior_embeddings: np.ndarray,
        prior_entailed_claim_ids: List[str],
        response_node_ids: List[str],
        closeness_centrality: Dict[str, float],
        contested_claim_ids: Optional[List[str]] = None,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        k_neighbors: int = 5
    ):
        """
        Initialize the iterative knowledge expansion module.

        Args:
            graph: NetworkX bipartite graph with weighted edges
            claim_embeddings: Dict mapping claim_id -> embedding vector
            prior_embeddings: np.array of prior embeddings (num_priors, embedding_dim)
            prior_entailed_claim_ids: List/Set of claim IDs entailed by priors (C_P, grounded)
            response_node_ids: List of response node IDs in the graph
            closeness_centrality: Dict mapping claim_id -> CC score (for utility weighting)
            contested_claim_ids: Optional list of contested claim IDs. If None, will be
                                computed as all claims not in prior_entailed_claim_ids
            lambda1: Weight for centrality in priority score (default: 1.0)
            lambda2: Weight for novelty in priority score (default: 1.0)
            k_neighbors: Number of nearest priors for cost computation (default: 5)
        """
        self.graph = graph.copy()
        self.claim_embeddings = claim_embeddings
        self.prior_embeddings = prior_embeddings
        self.prior_entailed_claim_ids = set(prior_entailed_claim_ids)
        self.response_node_ids = list(response_node_ids)
        self.closeness_centrality = closeness_centrality
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.k_neighbors = k_neighbors

        # If contested claims not provided, compute them
        if contested_claim_ids is not None:
            self._contested_claim_ids = list(contested_claim_ids)
        else:
            self._contested_claim_ids = None

        # Ensure graph has cost attribute on edges
        self._add_edge_costs()

    def _add_edge_costs(self):
        """Add cost attribute to edges (cost = 1/weight)."""
        for u, v, data in self.graph.edges(data=True):
            if 'weight' in data and data['weight'] > 0:
                data['cost'] = 1.0 / data['weight']
            else:
                data['cost'] = 1.0  # Default cost for unweighted edges

    def get_contested_claims(self) -> List[str]:
        """
        Identify contested claims (not entailed by priors).

        Returns:
            List of contested claim node IDs
        """
        if self._contested_claim_ids is not None:
            return self._contested_claim_ids

        # Get all claim nodes from graph
        all_claim_ids = [
            n for n in self.graph.nodes()
            if self.graph.nodes[n].get('bipartite') == 1 or
               str(n).startswith('C')  # Claim node naming convention
        ]

        # Filter out prior-entailed claims
        contested = [cid for cid in all_claim_ids if cid not in self.prior_entailed_claim_ids]

        return contested

    def compute_scores(self, contested_claim_ids: List[str]) -> Dict[str, Any]:
        """
        Compute all scores for contested claims.

        Args:
            contested_claim_ids: List of contested claim IDs

        Returns:
            Dictionary containing all computed scores
        """
        # Centrality scores (betweenness on contested subgraph)
        centrality_raw = compute_centrality_scores(
            self.graph, contested_claim_ids, self.response_node_ids
        )

        # Novelty scores (graph distance to priors)
        novelty_raw = compute_novelty_scores(
            self.graph, contested_claim_ids, list(self.prior_entailed_claim_ids)
        )

        # Priority scores (combined, normalized)
        priority_scores, S_centrality, S_novelty = compute_priority_scores(
            centrality_raw, novelty_raw, self.lambda1, self.lambda2
        )

        # Convert claim IDs to indices for embedding lookup
        # Assumes claim IDs are like "C0", "C1", etc. or integers
        contested_indices = []
        for cid in contested_claim_ids:
            if isinstance(cid, int):
                contested_indices.append(cid)
            elif isinstance(cid, str) and cid.startswith('C'):
                try:
                    contested_indices.append(int(cid[1:]))
                except ValueError:
                    contested_indices.append(hash(cid) % 10000)
            else:
                contested_indices.append(hash(cid) % 10000)

        # Costs (embedding distance, normalized)
        costs, costs_raw = compute_normalized_costs(
            self.claim_embeddings, self.prior_embeddings, contested_indices, self.k_neighbors
        )

        # Map indices back to claim IDs
        costs_by_id = {}
        costs_raw_by_id = {}
        for cid, idx in zip(contested_claim_ids, contested_indices):
            costs_by_id[cid] = costs.get(idx, 0.0)
            costs_raw_by_id[cid] = costs_raw.get(idx, 0.0)

        return {
            'priority_scores': priority_scores,
            'costs': costs_by_id,
            'S_centrality': S_centrality,
            'S_novelty': S_novelty,
            'centrality_raw': centrality_raw,
            'novelty_raw': novelty_raw,
            'costs_raw': costs_raw_by_id
        }

    def select_claims(self, k: int) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """
        Main method: select top-k claims for verification.

        Args:
            k: Number of claims to select for verification (budget)

        Returns:
            Tuple of:
                - selection: Dict with 'verify' and 'discard' claim lists
                - diagnostics: Dict with all intermediate scores for analysis
        """
        # Get contested claims
        contested_claim_ids = self.get_contested_claims()

        if len(contested_claim_ids) == 0:
            print("No contested claims found.")
            return {'verify': [], 'discard': []}, {}

        print(f"Found {len(contested_claim_ids)} contested claims")

        # Compute all scores
        scores = self.compute_scores(contested_claim_ids)

        # Select claims for verification
        selection, select_diagnostics = select_claims_for_verification(
            contested_claim_ids=contested_claim_ids,
            closeness_centrality=self.closeness_centrality,
            priority_scores=scores['priority_scores'],
            costs=scores['costs'],
            k=k
        )

        # Merge diagnostics
        diagnostics = {**scores, **select_diagnostics}
        diagnostics['num_contested'] = len(contested_claim_ids)
        diagnostics['num_selected'] = len(selection['verify'])

        return selection, diagnostics

    def update_after_verification(
        self,
        verified_claims: List[Tuple[str, bool]],
        new_prior_ids: List[str]
    ):
        """
        Update internal state after a verification round.

        Args:
            verified_claims: List of (claim_id, verdict) tuples
            new_prior_ids: List of claim_ids that are now confirmed as priors
        """
        # Add new priors to the set
        self.prior_entailed_claim_ids.update(new_prior_ids)

        # Clear cached contested claims to force recomputation
        self._contested_claim_ids = None

        print(f"Added {len(new_prior_ids)} new priors")
        print(f"Total prior-entailed claims: {len(self.prior_entailed_claim_ids)}")

    def get_claim_details(self, claim_id: str, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed information about a specific claim.

        Args:
            claim_id: The claim ID to get details for
            diagnostics: The diagnostics dict from select_claims()

        Returns:
            Dictionary with all scores and metrics for the claim
        """
        return {
            'claim_id': claim_id,
            'priority_score': diagnostics['priority_scores'].get(claim_id, 0.0),
            'cost': diagnostics['costs'].get(claim_id, 0.0),
            'centrality_normalized': diagnostics['S_centrality'].get(claim_id, 0.0),
            'novelty_normalized': diagnostics['S_novelty'].get(claim_id, 0.0),
            'centrality_raw': diagnostics['centrality_raw'].get(claim_id, 0.0),
            'novelty_raw': diagnostics['novelty_raw'].get(claim_id, None),
            'sigma_cc': diagnostics.get('sigma_cc', {}).get(claim_id, 0.5),
            'info_gain': diagnostics.get('info_gains', {}).get(claim_id, 0.0),
            'utility': diagnostics.get('utilities', {}).get(claim_id, 0.0),
            'closeness_centrality': self.closeness_centrality.get(claim_id, 0.0)
        }


# ==============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# ==============================================================================

def build_graph_from_weighted_data(
    weighted_graph_data: Dict,
    num_priors: int,
    num_responses: int,
    num_claims: int
) -> nx.Graph:
    """
    Build a NetworkX graph from weighted graph data structure.

    Args:
        weighted_graph_data: Dict containing 'edge_weights' matrix
        num_priors: Number of prior nodes
        num_responses: Number of response nodes
        num_claims: Number of claim nodes

    Returns:
        NetworkX Graph with proper node labels and edge weights
    """
    G = nx.Graph()

    # Add source nodes (priors + responses)
    num_sources = num_priors + num_responses
    source_nodes = [f"S{i}" for i in range(num_sources)]
    claim_nodes = [f"C{i}" for i in range(num_claims)]

    G.add_nodes_from(source_nodes, bipartite=0, type='source')
    G.add_nodes_from(claim_nodes, bipartite=1, type='claim')

    # Add weighted edges
    edge_weights = np.array(weighted_graph_data['edge_weights'])

    for source_idx in range(num_sources):
        for claim_idx in range(num_claims):
            weight = edge_weights[source_idx, claim_idx]
            if weight > 0:
                cost = 1.0 / weight
                G.add_edge(f"S{source_idx}", f"C{claim_idx}", weight=weight, cost=cost)

    return G


def get_node_ids_from_graph_data(
    weighted_graph_data: Dict,
    num_priors: int,
    num_responses: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract node ID lists from graph data.

    Returns:
        Tuple of (prior_node_ids, response_node_ids, claim_node_ids)
    """
    prior_node_ids = [f"S{i}" for i in range(num_priors)]
    response_node_ids = [f"S{i}" for i in range(num_priors, num_priors + num_responses)]

    num_claims = len(weighted_graph_data['edge_weights'][0]) if weighted_graph_data['edge_weights'] else 0
    claim_node_ids = [f"C{i}" for i in range(num_claims)]

    return prior_node_ids, response_node_ids, claim_node_ids


if __name__ == "__main__":
    # Test the module with a simple example
    print("Testing Iterative Knowledge Expansion Module...")

    # Create a simple test graph
    # 1 prior, 3 responses, 5 claims
    num_priors = 1
    num_responses = 3
    num_claims = 5

    # Sample edge weights
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

    weighted_graph_data = {'edge_weights': edge_weights.tolist()}

    # Build graph
    G = build_graph_from_weighted_data(weighted_graph_data, num_priors, num_responses, num_claims)

    # Get node IDs
    prior_ids, response_ids, claim_ids = get_node_ids_from_graph_data(
        weighted_graph_data, num_priors, num_responses
    )

    # Mock embeddings (random for testing)
    np.random.seed(42)
    claim_embeddings = {i: np.random.randn(384) for i in range(num_claims)}
    prior_embeddings = np.random.randn(2, 384)  # 2 prior claims

    # Mock closeness centrality (from previous stage)
    closeness_centrality = {
        'C0': 0.85,  # Grounded
        'C1': 0.75,  # Grounded
        'C2': 0.45,  # Boundary
        'C3': 0.35,  # Boundary
        'C4': 0.25,  # Boundary/low
    }

    # Prior-entailed claims (grounded)
    prior_entailed = ['C0', 'C1']

    # Contested claims (boundary)
    contested = ['C2', 'C3', 'C4']

    # Initialize expander
    expander = IterativeKnowledgeExpansion(
        graph=G,
        claim_embeddings=claim_embeddings,
        prior_embeddings=prior_embeddings,
        prior_entailed_claim_ids=prior_entailed,
        response_node_ids=response_ids,
        closeness_centrality=closeness_centrality,
        contested_claim_ids=contested,
        lambda1=1.0,
        lambda2=1.0,
        k_neighbors=2
    )

    # Select claims
    k = 2  # Select top 2 for verification
    selection, diagnostics = expander.select_claims(k=k)

    print(f"\nClaims to verify: {selection['verify']}")
    print(f"Claims discarded: {selection['discard']}")

    # Print details for selected claims
    print("\nSelected claim details:")
    for cid in selection['verify']:
        details = expander.get_claim_details(cid, diagnostics)
        print(f"\n  {cid}:")
        print(f"    Priority Score: {details['priority_score']:.3f}")
        print(f"    Cost: {details['cost']:.3f}")
        print(f"    sigma(CC): {details['sigma_cc']:.3f}")
        print(f"    Info Gain: {details['info_gain']:.3f}")
        print(f"    Utility: {details['utility']:.3f}")

    print("\n✓ Iterative Knowledge Expansion module working!")
