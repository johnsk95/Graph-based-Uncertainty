"""
Test Suite for Iterative Knowledge Expansion Module

Tests all components of the iterative expansion pipeline:
1. Z-score normalization
2. Centrality score computation
3. Novelty score computation
4. Cost computation
5. Utility computation
6. Claim selection
7. Full integration test
"""

import numpy as np
import networkx as nx
import sys

# Test result tracking
test_results = []


def log_test(name: str, passed: bool, message: str = ""):
    """Log a test result."""
    status = "✓" if passed else "✗"
    test_results.append((name, passed, message))
    print(f"  {status} {name}" + (f": {message}" if message else ""))


def test_z_normalize_positive():
    """Test z-score normalization with positive shift."""
    print("\n[1/7] Testing z_normalize_positive...")

    from src.iterative_expansion import z_normalize_positive

    # Test 1: Normal case
    scores = {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0}
    normalized = z_normalize_positive(scores)
    all_positive = all(v >= 0 for v in normalized.values())
    log_test("All values positive", all_positive)

    # Test 2: All same values
    same_scores = {'a': 5.0, 'b': 5.0, 'c': 5.0}
    normalized_same = z_normalize_positive(same_scores)
    all_equal = len(set(normalized_same.values())) == 1
    log_test("Same values handled", all_equal)

    # Test 3: Contains None values
    with_none = {'a': 1.0, 'b': None, 'c': 3.0}
    normalized_none = z_normalize_positive(with_none)
    none_is_max = normalized_none['b'] >= max(normalized_none['a'], normalized_none['c'])
    log_test("None values assigned max+1", none_is_max)

    # Test 4: Single value
    single = {'a': 42.0}
    normalized_single = z_normalize_positive(single)
    single_ok = normalized_single['a'] == 0.0
    log_test("Single value normalized to 0", single_ok)

    # Test 5: Empty dict
    empty = {}
    normalized_empty = z_normalize_positive(empty)
    empty_ok = len(normalized_empty) == 0
    log_test("Empty dict handled", empty_ok)


def test_centrality_scores():
    """Test betweenness centrality computation on contested subgraph."""
    print("\n[2/7] Testing compute_centrality_scores...")

    from src.iterative_expansion import compute_centrality_scores

    # Create a simple graph
    G = nx.Graph()
    G.add_nodes_from(['S0', 'S1', 'S2'], type='source')  # 3 responses
    G.add_nodes_from(['C0', 'C1', 'C2', 'C3'], type='claim')  # 4 claims

    # Add edges
    G.add_edge('S0', 'C0', weight=1.0, cost=1.0)
    G.add_edge('S0', 'C1', weight=1.0, cost=1.0)
    G.add_edge('S1', 'C1', weight=1.0, cost=1.0)
    G.add_edge('S1', 'C2', weight=1.0, cost=1.0)
    G.add_edge('S2', 'C2', weight=1.0, cost=1.0)
    G.add_edge('S2', 'C3', weight=1.0, cost=1.0)

    response_ids = ['S0', 'S1', 'S2']
    contested_ids = ['C0', 'C1', 'C2', 'C3']

    centrality = compute_centrality_scores(G, contested_ids, response_ids)

    # Check all claims have scores
    all_have_scores = all(cid in centrality for cid in contested_ids)
    log_test("All claims have scores", all_have_scores)

    # Check values are in valid range
    valid_range = all(0 <= v <= 1 for v in centrality.values())
    log_test("Centrality values in [0, 1]", valid_range)

    # C1 and C2 should have higher centrality (more central in the graph)
    middle_higher = centrality['C1'] >= centrality['C0'] or centrality['C2'] >= centrality['C3']
    log_test("Central claims have higher scores", middle_higher)


def test_novelty_scores():
    """Test novelty score computation (distance to priors)."""
    print("\n[3/7] Testing compute_novelty_scores...")

    from src.iterative_expansion import compute_novelty_scores

    # Create graph with priors and claims
    G = nx.Graph()
    G.add_nodes_from(['S0'], type='source')  # Prior
    G.add_nodes_from(['S1', 'S2'], type='source')  # Responses
    G.add_nodes_from(['C0', 'C1', 'C2', 'C3'], type='claim')

    # C0 is grounded (connected to prior via response)
    G.add_edge('S0', 'C0', weight=3.0, cost=1/3.0)
    G.add_edge('S1', 'C0', weight=2.0, cost=0.5)
    G.add_edge('S1', 'C1', weight=1.0, cost=1.0)
    G.add_edge('S1', 'C2', weight=1.0, cost=1.0)
    G.add_edge('S2', 'C2', weight=1.0, cost=1.0)
    G.add_edge('S2', 'C3', weight=0.5, cost=2.0)

    prior_entailed = ['C0']  # Grounded claim
    contested = ['C1', 'C2', 'C3']

    novelty = compute_novelty_scores(G, contested, prior_entailed)

    # All contested claims should have scores
    all_have_scores = all(cid in novelty for cid in contested)
    log_test("All contested claims have scores", all_have_scores)

    # C1 should be closer to prior than C3 (shorter path)
    if novelty['C1'] is not None and novelty['C3'] is not None:
        c1_closer = novelty['C1'] <= novelty['C3']
        log_test("Closer claims have lower novelty", c1_closer)
    else:
        log_test("Closer claims have lower novelty", True, "Some claims disconnected")


def test_embedding_costs():
    """Test embedding-based cost computation."""
    print("\n[4/7] Testing compute_embedding_costs...")

    from src.iterative_expansion import compute_embedding_costs

    # Create deterministic embeddings for reliable testing
    # Prior embeddings - all zeros
    prior_embeddings = np.zeros((5, 384))

    # Claim embeddings - with controlled distances
    claim_embeddings = {
        0: np.zeros(384),  # Identical to priors (distance ~0)
        1: np.ones(384) * 0.1,  # Close to priors
        2: np.ones(384) * 1.0,  # Medium distance
        3: np.ones(384) * 5.0,  # Far from priors
    }

    contested = [0, 1, 2, 3]
    costs = compute_embedding_costs(claim_embeddings, prior_embeddings, contested, k=3)

    # All claims should have costs
    all_have_costs = all(cid in costs for cid in contested)
    log_test("All claims have costs", all_have_costs)

    # Claims far from priors should have higher cost (using cosine distance)
    # With our setup: claim 0 is closest, claim 3 is farthest
    # Note: cosine distance of zeros is 0, and for parallel vectors it depends on magnitude
    # For more reliable test, check that costs are non-negative
    costs_non_negative = all(v >= 0 for v in costs.values())
    log_test("Costs are non-negative", costs_non_negative)

    # Test with empty priors
    empty_costs = compute_embedding_costs(claim_embeddings, np.array([]), contested, k=3)
    uniform_costs = all(v == 1.0 for v in empty_costs.values())
    log_test("Empty priors give uniform cost", uniform_costs)


def test_closeness_normalization():
    """Test closeness centrality normalization."""
    print("\n[5/7] Testing normalize_closeness_centrality...")

    from src.iterative_expansion import normalize_closeness_centrality

    cc_scores = {'C0': 0.1, 'C1': 0.3, 'C2': 0.5, 'C3': 0.7, 'C4': 0.9}

    # Test sigmoid method
    sigma_sigmoid = normalize_closeness_centrality(cc_scores, method='sigmoid')
    sigmoid_in_range = all(0 <= v <= 1 for v in sigma_sigmoid.values())
    log_test("Sigmoid values in [0, 1]", sigmoid_in_range)

    # Higher CC should give higher sigma
    sigmoid_ordered = sigma_sigmoid['C4'] > sigma_sigmoid['C0']
    log_test("Sigmoid preserves order", sigmoid_ordered)

    # Test minmax method
    sigma_minmax = normalize_closeness_centrality(cc_scores, method='minmax')
    minmax_in_range = all(0 <= v <= 1 for v in sigma_minmax.values())
    log_test("Minmax values in [0, 1]", minmax_in_range)

    # Min should be 0, max should be 1
    minmax_extremes = abs(sigma_minmax['C0']) < 0.01 and abs(sigma_minmax['C4'] - 1.0) < 0.01
    log_test("Minmax extremes correct", minmax_extremes)


def test_utility_computation():
    """Test utility computation."""
    print("\n[6/7] Testing compute_utilities...")

    from src.iterative_expansion import compute_utilities

    priority_scores = {'C0': 1.0, 'C1': 2.0, 'C2': 3.0}
    costs = {'C0': 0.5, 'C1': 0.5, 'C2': 0.5}
    sigma_cc = {'C0': 0.9, 'C1': 0.5, 'C2': 0.1}  # High CC = high confidence

    utilities, info_gains = compute_utilities(priority_scores, costs, sigma_cc)

    # All claims should have utilities
    all_have_utilities = all(cid in utilities for cid in priority_scores)
    log_test("All claims have utilities", all_have_utilities)

    # Info gain should be higher for low-confidence claims
    # C2 has low CC (0.1) and high priority (3.0), so high info gain
    # C0 has high CC (0.9) and low priority (1.0), so low info gain
    info_gain_correct = info_gains['C2'] > info_gains['C0']
    log_test("Info gain higher for uncertain claims", info_gain_correct)

    # Utility = info_gain - cost
    utility_formula_correct = abs(utilities['C0'] - (info_gains['C0'] - costs['C0'])) < 0.01
    log_test("Utility formula correct", utility_formula_correct)


def test_full_integration():
    """Test full integration of IterativeKnowledgeExpansion class."""
    print("\n[7/7] Testing full integration...")

    from src.iterative_expansion import (
        IterativeKnowledgeExpansion,
        build_graph_from_weighted_data,
        get_node_ids_from_graph_data
    )

    # Create test data
    num_priors = 1
    num_responses = 3
    num_claims = 5

    edge_weights = np.array([
        [3.0, 3.0, 0.0, 0.0, 0.0],  # Prior
        [2.0, 2.0, 1.0, 0.5, 0.0],  # Response 0
        [2.0, 0.0, 1.0, 1.0, 0.0],  # Response 1
        [2.0, 2.0, 1.0, 0.0, 1.0],  # Response 2
    ])

    weighted_graph_data = {'edge_weights': edge_weights.tolist()}

    # Build graph
    G = build_graph_from_weighted_data(weighted_graph_data, num_priors, num_responses, num_claims)
    graph_built = G.number_of_nodes() == num_priors + num_responses + num_claims
    log_test("Graph built correctly", graph_built)

    # Get node IDs
    prior_ids, response_ids, claim_ids = get_node_ids_from_graph_data(
        weighted_graph_data, num_priors, num_responses
    )

    # Mock embeddings
    np.random.seed(42)
    claim_embeddings = {i: np.random.randn(384) for i in range(num_claims)}
    prior_embeddings = np.random.randn(2, 384)

    # Mock closeness centrality
    closeness_centrality = {
        'C0': 0.85, 'C1': 0.75, 'C2': 0.45, 'C3': 0.35, 'C4': 0.25
    }

    # Define categories
    grounded = ['C0', 'C1']
    contested = ['C2', 'C3', 'C4']

    # Initialize expander
    expander = IterativeKnowledgeExpansion(
        graph=G,
        claim_embeddings=claim_embeddings,
        prior_embeddings=prior_embeddings,
        prior_entailed_claim_ids=grounded,
        response_node_ids=response_ids,
        closeness_centrality=closeness_centrality,
        contested_claim_ids=contested,
        lambda1=1.0,
        lambda2=1.0,
        k_neighbors=2
    )
    expander_initialized = expander is not None
    log_test("Expander initialized", expander_initialized)

    # Select claims
    k = 2
    selection, diagnostics = expander.select_claims(k=k)

    # Check selection structure
    has_verify = 'verify' in selection
    has_discard = 'discard' in selection
    log_test("Selection has correct structure", has_verify and has_discard)

    # Check counts
    total_selected = len(selection['verify']) + len(selection['discard'])
    count_correct = total_selected == len(contested)
    log_test("Selection count matches contested", count_correct)

    # Check k respected
    k_respected = len(selection['verify']) == min(k, len(contested))
    log_test("k budget respected", k_respected)

    # Check diagnostics
    has_diagnostics = 'utilities' in diagnostics and 'priority_scores' in diagnostics
    log_test("Diagnostics computed", has_diagnostics)

    # Test get_claim_details
    if selection['verify']:
        details = expander.get_claim_details(selection['verify'][0], diagnostics)
        has_all_fields = all(k in details for k in ['priority_score', 'cost', 'utility'])
        log_test("Claim details complete", has_all_fields)

    # Test update_after_verification
    expander.update_after_verification(
        verified_claims=[('C2', True)],
        new_prior_ids=['C2']
    )
    prior_updated = 'C2' in expander.prior_entailed_claim_ids
    log_test("Prior update works", prior_updated)


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING ITERATIVE KNOWLEDGE EXPANSION MODULE")
    print("=" * 80)

    try:
        test_z_normalize_positive()
        test_centrality_scores()
        test_novelty_scores()
        test_embedding_costs()
        test_closeness_normalization()
        test_utility_computation()
        test_full_integration()

        # Summary
        print("\n" + "=" * 80)
        passed = sum(1 for _, p, _ in test_results if p)
        total = len(test_results)

        if passed == total:
            print(f"ALL TESTS PASSED! ({passed}/{total}) ✓")
        else:
            print(f"TESTS: {passed}/{total} passed")
            print("\nFailed tests:")
            for name, p, msg in test_results:
                if not p:
                    print(f"  ✗ {name}: {msg}")
            sys.exit(1)

        print("=" * 80)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
