"""
Test Suite for Expert Verification and Graph Update Modules

Tests all components of the verification pipeline:
1. VerificationResult data class
2. MockExpertVerifier
3. Context preparation utilities
4. GraphUpdater and update functions
5. VerificationPipeline
6. Full integration test
"""

import networkx as nx
import sys

# Test result tracking
test_results = []


def log_test(name: str, passed: bool, message: str = ""):
    """Log a test result."""
    status = "✓" if passed else "✗"
    test_results.append((name, passed, message))
    print(f"  {status} {name}" + (f": {message}" if message else ""))


def test_verification_result():
    """Test VerificationResult data class."""
    print("\n[1/8] Testing VerificationResult...")

    from src.expert_verification import VerificationResult

    # Test 1: Create result
    result = VerificationResult(
        claim_id='C0',
        claim_text='The sky is blue.',
        verdict=True,
        reasoning='Supported by context',
        confidence=0.95,
        verifier_type='test'
    )
    created_ok = result.claim_id == 'C0' and result.verdict is True
    log_test("Result creation", created_ok)

    # Test 2: Serialization
    result_dict = result.to_dict()
    has_all_keys = all(k in result_dict for k in [
        'claim_id', 'claim_text', 'verdict', 'reasoning', 'confidence', 'timestamp'
    ])
    log_test("Serialization to dict", has_all_keys)

    # Test 3: Deserialization
    restored = VerificationResult.from_dict(result_dict)
    restored_ok = (
        restored.claim_id == result.claim_id and
        restored.verdict == result.verdict and
        restored.reasoning == result.reasoning
    )
    log_test("Deserialization from dict", restored_ok)

    # Test 4: Default values
    minimal = VerificationResult(
        claim_id='C1',
        claim_text='Test claim',
        verdict=False,
        reasoning='Rejected'
    )
    has_defaults = (
        minimal.confidence is None and
        minimal.verifier_type == "unknown" and
        minimal.metadata == {}
    )
    log_test("Default values", has_defaults)


def test_mock_verifier():
    """Test MockExpertVerifier."""
    print("\n[2/8] Testing MockExpertVerifier...")

    from src.expert_verification import MockExpertVerifier

    # Test 1: Default verdict
    verifier = MockExpertVerifier(default_verdict=True)
    result = verifier.verify_claim('C0', 'Test claim', 'Context')
    default_ok = result.verdict is True
    log_test("Default verdict applied", default_ok)

    # Test 2: Verdict map
    verifier_map = MockExpertVerifier(
        default_verdict=True,
        verdict_map={'C1': False, 'C2': False}
    )
    result_c0 = verifier_map.verify_claim('C0', 'Claim 0', 'Context')
    result_c1 = verifier_map.verify_claim('C1', 'Claim 1', 'Context')
    map_ok = result_c0.verdict is True and result_c1.verdict is False
    log_test("Verdict map applied", map_ok)

    # Test 3: Batch verification
    claims = [
        {'claim_id': 'C0', 'claim_text': 'Claim 0'},
        {'claim_id': 'C1', 'claim_text': 'Claim 1'},
        {'claim_id': 'C2', 'claim_text': 'Claim 2'},
    ]
    results = verifier_map.verify_batch(claims, 'Context')
    batch_ok = (
        len(results) == 3 and
        results[0].verdict is True and
        results[1].verdict is False and
        results[2].verdict is False
    )
    log_test("Batch verification", batch_ok)

    # Test 4: Verification count tracking
    count_ok = verifier_map.verification_count == 5  # 2 single + 3 batch
    log_test("Verification count tracked", count_ok)


def test_context_preparation():
    """Test context preparation utilities."""
    print("\n[3/8] Testing context preparation...")

    from src.expert_verification import (
        prepare_context_for_claim,
        prepare_batch_with_individual_context
    )

    # Test 1: Basic context preparation
    context_docs = {
        'doc1': 'This is document one with information.',
        'doc2': 'This is document two with more data.'
    }
    context = prepare_context_for_claim('C0', context_docs)
    has_both = 'document one' in context and 'document two' in context
    log_test("Basic context preparation", has_both)

    # Test 2: With mapping
    mapping = {'C0': ['doc1'], 'C1': ['doc2']}
    context_mapped = prepare_context_for_claim('C0', context_docs, mapping)
    only_doc1 = 'document one' in context_mapped and 'document two' not in context_mapped
    log_test("Context with mapping", only_doc1)

    # Test 3: Batch preparation
    claims = [
        {'claim_id': 'C0', 'claim_text': 'Claim 0'},
        {'claim_id': 'C1', 'claim_text': 'Claim 1'},
    ]
    prepared = prepare_batch_with_individual_context(claims, context_docs, mapping)
    batch_prep_ok = (
        len(prepared) == 2 and
        'context' in prepared[0] and
        'document one' in prepared[0]['context']
    )
    log_test("Batch preparation", batch_prep_ok)

    # Test 4: Max context length
    long_doc = {'doc1': 'A' * 10000}
    truncated = prepare_context_for_claim('C0', long_doc, max_context_length=100)
    truncated_ok = len(truncated) <= 200  # Some overhead for headers
    log_test("Context truncation", truncated_ok)


def test_verifier_factory():
    """Test create_verifier factory function."""
    print("\n[4/8] Testing verifier factory...")

    from src.expert_verification import create_verifier, MockExpertVerifier

    # Test 1: Create mock verifier
    mock = create_verifier('mock', default_verdict=False)
    is_mock = isinstance(mock, MockExpertVerifier)
    log_test("Create mock verifier", is_mock)

    # Test 2: Mock with parameters
    mock_param = create_verifier('mock', verdict_map={'C0': True})
    result = mock_param.verify_claim('C0', 'Test', 'Context')
    params_ok = result.verdict is True
    log_test("Mock with parameters", params_ok)

    # Test 3: Invalid type
    try:
        create_verifier('invalid_type')
        invalid_ok = False
    except ValueError:
        invalid_ok = True
    log_test("Invalid type raises error", invalid_ok)


def test_graph_update_functions():
    """Test individual graph update functions."""
    print("\n[5/8] Testing graph update functions...")

    from src.graph_update import (
        create_verification_prior,
        connect_accepted_claims,
        archive_and_remove_rejected_claims,
        update_prior_entailed_claims,
        recompute_closeness_centrality
    )

    # Create test graph
    G = nx.Graph()
    G.add_node('S0', type='prior', bipartite=0)
    G.add_node('S1', type='response', bipartite=0)
    for i in range(4):
        G.add_node(f'C{i}', type='claim', bipartite=1, text=f'Claim {i}')
    G.add_edge('S0', 'C0', weight=3.0, cost=1/3)
    G.add_edge('S1', 'C1', weight=1.0, cost=1.0)
    G.add_edge('S1', 'C2', weight=1.0, cost=1.0)
    G.add_edge('S1', 'C3', weight=1.0, cost=1.0)

    initial_nodes = G.number_of_nodes()

    # Test 1: Create verification prior
    prior_id = create_verification_prior(G, verification_round=1, verifier_type='test')
    prior_created = prior_id in G.nodes and G.nodes[prior_id]['type'] == 'prior'
    log_test("Create verification prior", prior_created)

    # Test 2: Connect accepted claims
    new_edges = connect_accepted_claims(G, prior_id, ['C1', 'C2'])
    edges_ok = len(new_edges) == 2 and G.has_edge(prior_id, 'C1')
    log_test("Connect accepted claims", edges_ok)

    # Test 3: Anchor weight applied
    edge_data = G.edges[prior_id, 'C1']
    weight_ok = edge_data['weight'] == 3.0 and edge_data['link_type'] == 'anchor'
    log_test("Anchor edge weight correct", weight_ok)

    # Test 4: Archive and remove rejected claims
    archived = archive_and_remove_rejected_claims(
        G, ['C3'], rejection_reasons={'C3': 'Test rejection'}, verification_round=1
    )
    archived_ok = (
        len(archived) == 1 and
        archived[0].claim_id == 'C3' and
        'C3' not in G.nodes
    )
    log_test("Archive rejected claims", archived_ok)

    # Test 5: Update prior-entailed claims
    prior_entailed = {'C0'}
    updated = update_prior_entailed_claims(prior_entailed, ['C1', 'C2'], ['C3'])
    updated_ok = 'C1' in updated and 'C2' in updated and 'C3' not in updated
    log_test("Update prior-entailed set", updated_ok)

    # Test 6: Recompute closeness centrality
    cc = recompute_closeness_centrality(G)
    cc_ok = len(cc) > 0 and all(0 <= v <= 1 for v in cc.values())
    log_test("Recompute closeness centrality", cc_ok)


def test_graph_updater_class():
    """Test GraphUpdater class."""
    print("\n[6/8] Testing GraphUpdater class...")

    from src.graph_update import GraphUpdater
    from src.expert_verification import VerificationResult

    # Create test graph
    G = nx.Graph()
    G.add_node('S0', type='prior', bipartite=0)
    G.add_node('S1', type='response', bipartite=0)
    for i in range(5):
        G.add_node(f'C{i}', type='claim', bipartite=1, text=f'Claim {i}')
    G.add_edge('S0', 'C0', weight=3.0, cost=1/3, link_type='anchor')
    G.add_edge('S1', 'C1', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S1', 'C2', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S1', 'C3', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S1', 'C4', weight=0.5, cost=2.0, link_type='contra')

    # Initialize updater
    updater = GraphUpdater(G, prior_entailed_claim_ids={'C0'})
    init_ok = updater.verification_round == 0
    log_test("GraphUpdater initialized", init_ok)

    # Process verification results
    results = [
        VerificationResult('C1', 'Claim 1', True, 'Accepted'),
        VerificationResult('C2', 'Claim 2', True, 'Accepted'),
        VerificationResult('C3', 'Claim 3', False, 'Rejected'),
    ]

    summary, new_cc = updater.process_verification_results(results, verifier_type='test')

    # Test 1: Summary correct
    summary_ok = (
        summary.round_number == 1 and
        summary.accepted_count == 2 and
        summary.rejected_count == 1
    )
    log_test("Verification summary correct", summary_ok)

    # Test 2: Prior-entailed updated
    prior_ok = 'C1' in updater.prior_entailed_claim_ids and 'C2' in updater.prior_entailed_claim_ids
    log_test("Prior-entailed updated", prior_ok)

    # Test 3: Rejected claim archived
    archived_ok = len(updater.false_claims_archive) == 1 and updater.false_claims_archive[0].claim_id == 'C3'
    log_test("Rejected claim archived", archived_ok)

    # Test 4: Rejected claim removed from graph
    removed_ok = 'C3' not in G.nodes
    log_test("Rejected claim removed from graph", removed_ok)

    # Test 5: Verification history tracked
    history_ok = len(updater.verification_history) == 1
    log_test("Verification history tracked", history_ok)

    # Test 6: Get remaining contested
    remaining = updater.get_remaining_contested_claims(['C0', 'C1', 'C2', 'C3', 'C4'])
    remaining_ok = 'C4' in remaining and 'C1' not in remaining
    log_test("Get remaining contested claims", remaining_ok)

    # Test 7: Restore archived claim
    restored = updater.restore_archived_claim('C3')
    restore_ok = restored and 'C3' in G.nodes
    log_test("Restore archived claim", restore_ok)


def test_verification_pipeline():
    """Test VerificationPipeline class."""
    print("\n[7/8] Testing VerificationPipeline...")

    from src.verification_pipeline import VerificationPipeline
    from src.expert_verification import MockExpertVerifier

    # Create test graph
    G = nx.Graph()
    G.add_node('S0', type='prior', bipartite=0)
    G.add_node('S1', type='response', bipartite=0)
    for i in range(4):
        G.add_node(f'C{i}', type='claim', bipartite=1, text=f'Claim {i}')
    G.add_edge('S0', 'C0', weight=3.0, cost=1/3, link_type='anchor')
    G.add_edge('S1', 'C1', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S1', 'C2', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S1', 'C3', weight=1.0, cost=1.0, link_type='neutral')

    # Initialize with mock verifier
    mock_verifier = MockExpertVerifier(
        default_verdict=True,
        verdict_map={'C3': False}
    )

    pipeline = VerificationPipeline(
        graph=G,
        prior_entailed_claim_ids={'C0'},
        verifier=mock_verifier
    )
    init_ok = pipeline.graph is not None
    log_test("Pipeline initialized", init_ok)

    # Run verification round
    claims = [
        {'claim_id': 'C1', 'claim_text': 'Claim 1'},
        {'claim_id': 'C2', 'claim_text': 'Claim 2'},
        {'claim_id': 'C3', 'claim_text': 'Claim 3'},
    ]

    summary, new_cc = pipeline.run_verification_round(claims, context="Test context")

    # Test 1: Round completed
    round_ok = summary.round_number == 1
    log_test("Verification round completed", round_ok)

    # Test 2: Correct verdicts applied
    verdicts_ok = summary.accepted_count == 2 and summary.rejected_count == 1
    log_test("Correct verdicts applied", verdicts_ok)

    # Test 3: Prior-entailed accessible
    prior_ok = 'C1' in pipeline.prior_entailed_claim_ids
    log_test("Prior-entailed accessible via property", prior_ok)

    # Test 4: Verification history accessible
    history_ok = len(pipeline.verification_history) == 1
    log_test("Verification history accessible", history_ok)

    # Test 5: False claims archive accessible
    archive_ok = len(pipeline.false_claims_archive) == 1
    log_test("False claims archive accessible", archive_ok)

    # Test 6: Empty claims handled
    empty_summary, _ = pipeline.run_verification_round([], context="Test")
    empty_ok = empty_summary is None
    log_test("Empty claims handled", empty_ok)


def test_full_integration():
    """Test full integration of all modules."""
    print("\n[8/8] Testing full integration...")

    import numpy as np
    from src.verification_pipeline import IterativeExpansionRunner
    from src.expert_verification import MockExpertVerifier

    # Create test graph
    G = nx.Graph()
    G.add_node('S0', type='prior', bipartite=0)
    G.add_node('S1', type='response', bipartite=0)
    G.add_node('S2', type='response', bipartite=0)

    num_claims = 6
    for i in range(num_claims):
        G.add_node(f'C{i}', type='claim', bipartite=1, text=f'Test claim {i}')

    G.add_edge('S0', 'C0', weight=3.0, cost=1/3, link_type='anchor')
    G.add_edge('S1', 'C1', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S1', 'C2', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S1', 'C3', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S2', 'C4', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S2', 'C5', weight=0.5, cost=2.0, link_type='contra')

    # Mock embeddings
    np.random.seed(42)
    claim_embeddings = {i: np.random.randn(384) for i in range(num_claims)}
    prior_embeddings = np.random.randn(2, 384)

    # Initial closeness centrality
    closeness_centrality = {f'C{i}': 0.5 - i * 0.05 for i in range(num_claims)}

    # Mock verifier: accept all except C5
    mock_verifier = MockExpertVerifier(
        default_verdict=True,
        verdict_map={'C5': False}
    )

    # Initialize runner
    runner = IterativeExpansionRunner(
        graph=G,
        claim_embeddings=claim_embeddings,
        prior_embeddings=prior_embeddings,
        prior_entailed_claim_ids={'C0'},
        response_node_ids=['S1', 'S2'],
        closeness_centrality=closeness_centrality,
        verifier=mock_verifier,
        lambda1=1.0,
        lambda2=1.0,
        k_neighbors=2
    )
    runner_init_ok = runner.iteration_count == 0
    log_test("IterativeExpansionRunner initialized", runner_init_ok)

    # Run single iteration
    summary, new_cc, diagnostics = runner.run_single_iteration(
        k=3,
        context="Test context for verification"
    )

    # Test 1: Iteration ran
    iteration_ok = summary is not None and runner.iteration_count == 1
    log_test("Single iteration completed", iteration_ok)

    # Test 2: Prior-entailed updated
    prior_updated_ok = len(runner.prior_entailed_claim_ids) > 1
    log_test("Prior-entailed claims updated", prior_updated_ok)

    # Test 3: Diagnostics available
    diagnostics_ok = 'utilities' in diagnostics and 'priority_scores' in diagnostics
    log_test("Diagnostics available", diagnostics_ok)

    # Test 4: Iteration summaries tracked
    summaries_ok = len(runner.iteration_summaries) == 1
    log_test("Iteration summaries tracked", summaries_ok)

    # Test 5: Run another iteration
    summary2, _, _ = runner.run_single_iteration(
        k=2,
        context="More test context"
    )
    multi_iteration_ok = runner.iteration_count == 2
    log_test("Multiple iterations supported", multi_iteration_ok)


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING EXPERT VERIFICATION AND GRAPH UPDATE MODULES")
    print("=" * 80)

    try:
        test_verification_result()
        test_mock_verifier()
        test_context_preparation()
        test_verifier_factory()
        test_graph_update_functions()
        test_graph_updater_class()
        test_verification_pipeline()
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
