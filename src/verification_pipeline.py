"""
Complete Verification Pipeline

Combines expert verification and graph updates into a unified pipeline
that can be used for iterative knowledge expansion.

This module integrates:
1. IterativeKnowledgeExpansion - Selects claims for verification
2. ExpertVerifier - Verifies claims (LLM, human, or mock)
3. GraphUpdater - Updates graph based on verification results

The pipeline supports:
- Single verification rounds
- Multi-round iterative expansion
- Different verifier backends
- Progress tracking and logging
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple, Callable, Any
import networkx as nx
import numpy as np

from src.expert_verification import (
    BaseExpertVerifier,
    LLMExpertVerifier,
    MockExpertVerifier,
    VerificationResult,
    create_verifier
)
from src.graph_update import (
    GraphUpdater,
    VerificationRoundSummary,
    ArchivedClaim,
    recompute_closeness_centrality,
    EDGE_WEIGHTS
)
from src.iterative_expansion import IterativeKnowledgeExpansion


# ==============================================================================
# VERIFICATION PIPELINE
# ==============================================================================

class VerificationPipeline:
    """
    Complete pipeline combining LLM/expert verification and graph updates.

    Provides a unified interface for running verification rounds and
    updating the knowledge graph accordingly.
    """

    def __init__(
        self,
        graph: nx.Graph,
        prior_entailed_claim_ids: Set[str],
        verifier: Optional[BaseExpertVerifier] = None,
        verifier_type: str = "mock",
        verifier_kwargs: Optional[dict] = None,
        w_anchor: float = EDGE_WEIGHTS['anchor'],
        output_dir: Optional[str] = None
    ):
        """
        Initialize the verification pipeline.

        Args:
            graph: NetworkX bipartite graph
            prior_entailed_claim_ids: Current set of prior-entailed claim IDs
            verifier: Pre-configured verifier instance (optional)
            verifier_type: Type of verifier if not provided ('llm', 'mock', 'cli_human')
            verifier_kwargs: Arguments for verifier constructor
            w_anchor: Edge weight for anchor links
            output_dir: Directory for saving outputs (optional)
        """
        self.graph_updater = GraphUpdater(
            graph=graph,
            prior_entailed_claim_ids=prior_entailed_claim_ids,
            w_anchor=w_anchor
        )

        # Initialize verifier
        if verifier is not None:
            self.verifier = verifier
        else:
            verifier_kwargs = verifier_kwargs or {}
            self.verifier = create_verifier(verifier_type, **verifier_kwargs)

        self.verifier_type = verifier_type if verifier is None else type(verifier).__name__
        self.output_dir = output_dir

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    def run_verification_round(
        self,
        claims_to_verify: List[Dict[str, str]],
        context: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[VerificationRoundSummary, Dict[str, float]]:
        """
        Run a complete verification round with shared context.

        Args:
            claims_to_verify: List of dicts with 'claim_id' and 'claim_text'
            context: Context document for verification (shared across claims)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Tuple of (VerificationRoundSummary, new_closeness_centrality)
        """
        if len(claims_to_verify) == 0:
            print("No claims to verify.")
            return None, {}

        print(f"Verifying {len(claims_to_verify)} claims...")

        # Step 1: Run verification
        verification_results = self.verifier.verify_batch(
            claims=claims_to_verify,
            context=context,
            progress_callback=progress_callback
        )

        # Step 2: Update graph
        print("Updating graph...")
        summary, new_cc = self.graph_updater.process_verification_results(
            verification_results=verification_results,
            verifier_type=self.verifier_type
        )

        # Print summary
        self._print_round_summary(summary)

        # Save outputs if output_dir specified
        if self.output_dir:
            self._save_round_outputs(summary, verification_results)

        return summary, new_cc

    def run_verification_round_individual_context(
        self,
        claims_with_context: List[Dict[str, str]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[VerificationRoundSummary, Dict[str, float]]:
        """
        Run verification where each claim has individual context.

        Args:
            claims_with_context: List of dicts with 'claim_id', 'claim_text', 'context'
            progress_callback: Optional callback(current, total)

        Returns:
            Tuple of (VerificationRoundSummary, new_closeness_centrality)
        """
        if len(claims_with_context) == 0:
            print("No claims to verify.")
            return None, {}

        print(f"Verifying {len(claims_with_context)} claims with individual context...")

        # Step 1: Run verification
        verification_results = self.verifier.verify_batch_individual_context(
            claims_with_context=claims_with_context,
            progress_callback=progress_callback
        )

        # Step 2: Update graph
        print("Updating graph...")
        summary, new_cc = self.graph_updater.process_verification_results(
            verification_results=verification_results,
            verifier_type=self.verifier_type
        )

        self._print_round_summary(summary)

        if self.output_dir:
            self._save_round_outputs(summary, verification_results)

        return summary, new_cc

    def _print_round_summary(self, summary: VerificationRoundSummary):
        """Print verification round summary."""
        print(f"\nVerification Round {summary.round_number} Complete:")
        print(f"  Accepted: {summary.accepted_count}")
        print(f"  Rejected: {summary.rejected_count}")
        print(f"  New edges added: {summary.new_edges_added}")
        print(f"  Total prior-entailed claims: {summary.total_prior_entailed}")

    def _save_round_outputs(
        self,
        summary: VerificationRoundSummary,
        results: List[VerificationResult]
    ):
        """Save round outputs to files."""
        round_dir = Path(self.output_dir) / f"round_{summary.round_number}"
        round_dir.mkdir(exist_ok=True)

        # Save summary
        with open(round_dir / "summary.json", 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)

        # Save results
        results_data = [r.to_dict() for r in results]
        with open(round_dir / "verification_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)

    @property
    def graph(self) -> nx.Graph:
        """Get the current graph."""
        return self.graph_updater.graph

    @property
    def prior_entailed_claim_ids(self) -> Set[str]:
        """Get current prior-entailed claim IDs."""
        return self.graph_updater.prior_entailed_claim_ids

    @property
    def closeness_centrality(self) -> Dict[str, float]:
        """Compute and return current closeness centrality."""
        return recompute_closeness_centrality(self.graph_updater.graph)

    @property
    def verification_history(self) -> List[VerificationRoundSummary]:
        """Get verification history."""
        return self.graph_updater.verification_history

    @property
    def false_claims_archive(self) -> List[ArchivedClaim]:
        """Get archived false claims."""
        return self.graph_updater.false_claims_archive

    def export_all(self, output_dir: Optional[str] = None):
        """Export all data (archive, history) to files."""
        out_dir = output_dir or self.output_dir
        if not out_dir:
            raise ValueError("No output directory specified")

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        self.graph_updater.export_archive(f"{out_dir}/false_claims_archive.json")
        self.graph_updater.export_verification_history(f"{out_dir}/verification_history.json")

        # Save final graph state
        state = self.graph_updater.get_current_state()
        with open(f"{out_dir}/final_state.json", 'w') as f:
            json.dump(state, f, indent=2)


# ==============================================================================
# ITERATIVE EXPANSION RUNNER
# ==============================================================================

class IterativeExpansionRunner:
    """
    Runs multiple rounds of iterative knowledge expansion.

    Combines claim selection (IterativeKnowledgeExpansion) with
    verification (VerificationPipeline) in an iterative loop.
    """

    def __init__(
        self,
        graph: nx.Graph,
        claim_embeddings: Dict[int, np.ndarray],
        prior_embeddings: np.ndarray,
        prior_entailed_claim_ids: Set[str],
        response_node_ids: List[str],
        closeness_centrality: Dict[str, float],
        verifier: Optional[BaseExpertVerifier] = None,
        verifier_type: str = "mock",
        verifier_kwargs: Optional[dict] = None,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        k_neighbors: int = 5,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the iterative expansion runner.

        Args:
            graph: NetworkX bipartite graph
            claim_embeddings: Dict mapping claim_id -> embedding vector
            prior_embeddings: Array of prior embeddings
            prior_entailed_claim_ids: Initial set of prior-entailed claim IDs
            response_node_ids: List of response node IDs
            closeness_centrality: Initial closeness centrality scores
            verifier: Pre-configured verifier (optional)
            verifier_type: Type of verifier if not provided
            verifier_kwargs: Verifier constructor arguments
            lambda1: Weight for centrality in priority score
            lambda2: Weight for novelty in priority score
            k_neighbors: Nearest priors for cost computation
            output_dir: Directory for saving outputs
        """
        self.graph = graph
        self.claim_embeddings = claim_embeddings
        self.prior_embeddings = prior_embeddings
        self.prior_entailed_claim_ids = set(prior_entailed_claim_ids)
        self.response_node_ids = response_node_ids
        self.closeness_centrality = closeness_centrality
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.k_neighbors = k_neighbors
        self.output_dir = output_dir

        # Initialize pipeline
        self.pipeline = VerificationPipeline(
            graph=graph,
            prior_entailed_claim_ids=prior_entailed_claim_ids,
            verifier=verifier,
            verifier_type=verifier_type,
            verifier_kwargs=verifier_kwargs,
            output_dir=output_dir
        )

        # Track iterations
        self.iteration_count = 0
        self.iteration_summaries: List[Dict] = []

    def run_single_iteration(
        self,
        k: int,
        context: str,
        contested_claim_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[Optional[VerificationRoundSummary], Dict[str, float], Dict]:
        """
        Run a single iteration of expansion.

        Args:
            k: Number of claims to select for verification
            context: Context document for verification
            contested_claim_ids: Optional explicit list of contested claims
            progress_callback: Optional progress callback

        Returns:
            Tuple of (summary, new_closeness_centrality, selection_diagnostics)
        """
        self.iteration_count += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {self.iteration_count}")
        print(f"{'='*60}")

        # Step 1: Select claims for verification
        expander = IterativeKnowledgeExpansion(
            graph=self.graph,
            claim_embeddings=self.claim_embeddings,
            prior_embeddings=self.prior_embeddings,
            prior_entailed_claim_ids=self.prior_entailed_claim_ids,
            response_node_ids=self.response_node_ids,
            closeness_centrality=self.closeness_centrality,
            contested_claim_ids=contested_claim_ids,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            k_neighbors=self.k_neighbors
        )

        selection, diagnostics = expander.select_claims(k=k)

        if len(selection['verify']) == 0:
            print("No claims to verify. Expansion complete.")
            return None, self.closeness_centrality, diagnostics

        print(f"Selected {len(selection['verify'])} claims for verification")

        # Step 2: Prepare claims for verification
        claims_to_verify = []
        for cid in selection['verify']:
            claim_text = ""
            if cid in self.graph.nodes:
                node_data = self.graph.nodes[cid]
                claim_text = node_data.get('text', node_data.get('claim_text', ''))
            claims_to_verify.append({
                'claim_id': cid,
                'claim_text': claim_text
            })

        # Step 3: Run verification
        summary, new_cc = self.pipeline.run_verification_round(
            claims_to_verify=claims_to_verify,
            context=context,
            progress_callback=progress_callback
        )

        # Step 4: Update state
        self.prior_entailed_claim_ids = self.pipeline.prior_entailed_claim_ids
        self.closeness_centrality = new_cc

        # Track iteration
        self.iteration_summaries.append({
            'iteration': self.iteration_count,
            'selected': len(selection['verify']),
            'accepted': summary.accepted_count if summary else 0,
            'rejected': summary.rejected_count if summary else 0,
            'total_prior_entailed': len(self.prior_entailed_claim_ids)
        })

        return summary, new_cc, diagnostics

    def run_expansion_loop(
        self,
        context: str,
        k: int = 20,
        max_iterations: int = 10,
        convergence_threshold: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Run multiple expansion iterations until convergence or max iterations.

        Args:
            context: Context document for verification
            k: Number of claims per iteration
            max_iterations: Maximum iterations to run
            convergence_threshold: Stop if fewer than this many claims change
            progress_callback: Optional progress callback

        Returns:
            Summary dict with final state and statistics
        """
        print("\n" + "=" * 60)
        print("STARTING ITERATIVE EXPANSION LOOP")
        print("=" * 60)
        print(f"Max iterations: {max_iterations}")
        print(f"Claims per iteration: {k}")
        print(f"Initial prior-entailed: {len(self.prior_entailed_claim_ids)}")

        for iteration in range(max_iterations):
            prev_count = len(self.prior_entailed_claim_ids)

            summary, _, _ = self.run_single_iteration(
                k=k,
                context=context,
                progress_callback=progress_callback
            )

            if summary is None:
                print("\nNo more claims to verify. Convergence reached.")
                break

            # Check convergence
            change = abs(len(self.prior_entailed_claim_ids) - prev_count)
            if change <= convergence_threshold:
                print(f"\nConverged (change={change} <= threshold={convergence_threshold})")
                break

        # Final summary
        final_summary = {
            'total_iterations': self.iteration_count,
            'final_prior_entailed': len(self.prior_entailed_claim_ids),
            'total_archived': len(self.pipeline.false_claims_archive),
            'iteration_summaries': self.iteration_summaries,
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges()
        }

        print("\n" + "=" * 60)
        print("EXPANSION LOOP COMPLETE")
        print("=" * 60)
        print(f"Total iterations: {final_summary['total_iterations']}")
        print(f"Final prior-entailed claims: {final_summary['final_prior_entailed']}")
        print(f"Archived false claims: {final_summary['total_archived']}")

        # Export if output_dir specified
        if self.output_dir:
            self.pipeline.export_all(self.output_dir)
            with open(f"{self.output_dir}/expansion_summary.json", 'w') as f:
                json.dump(final_summary, f, indent=2)

        return final_summary


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def run_single_verification_round(
    graph: nx.Graph,
    claims_to_verify: List[Dict[str, str]],
    context: str,
    prior_entailed_claim_ids: Set[str],
    verifier_type: str = "mock",
    verifier_kwargs: Optional[dict] = None
) -> Tuple[nx.Graph, Set[str], VerificationRoundSummary]:
    """
    Convenience function to run a single verification round.

    Args:
        graph: NetworkX graph
        claims_to_verify: List of {'claim_id': ..., 'claim_text': ...}
        context: Verification context
        prior_entailed_claim_ids: Current grounded claims
        verifier_type: Type of verifier
        verifier_kwargs: Verifier arguments

    Returns:
        Tuple of (updated_graph, updated_prior_entailed, summary)
    """
    pipeline = VerificationPipeline(
        graph=graph,
        prior_entailed_claim_ids=prior_entailed_claim_ids,
        verifier_type=verifier_type,
        verifier_kwargs=verifier_kwargs
    )

    summary, _ = pipeline.run_verification_round(claims_to_verify, context)

    return pipeline.graph, pipeline.prior_entailed_claim_ids, summary


if __name__ == "__main__":
    # Test the verification pipeline
    print("Testing Verification Pipeline...")

    # Create test graph
    G = nx.Graph()
    G.add_node('S0', type='prior', bipartite=0)
    G.add_node('S1', type='response', bipartite=0)
    G.add_node('S2', type='response', bipartite=0)

    for i in range(5):
        G.add_node(f'C{i}', type='claim', bipartite=1, text=f'Test claim {i}')

    G.add_edge('S0', 'C0', weight=3.0, cost=1/3, link_type='anchor')
    G.add_edge('S1', 'C1', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S1', 'C2', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S2', 'C3', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S2', 'C4', weight=0.5, cost=2.0, link_type='contra')

    # Initialize pipeline with mock verifier
    mock_verifier = MockExpertVerifier(
        default_verdict=True,
        verdict_map={'C4': False}  # Reject C4
    )

    pipeline = VerificationPipeline(
        graph=G,
        prior_entailed_claim_ids={'C0'},
        verifier=mock_verifier
    )

    # Run verification
    claims = [
        {'claim_id': 'C1', 'claim_text': 'Test claim 1'},
        {'claim_id': 'C2', 'claim_text': 'Test claim 2'},
        {'claim_id': 'C4', 'claim_text': 'Test claim 4'},
    ]

    context = "This is test context for verification."

    summary, new_cc = pipeline.run_verification_round(claims, context)

    print(f"\nFinal state:")
    print(f"  Prior-entailed: {pipeline.prior_entailed_claim_ids}")
    print(f"  Archived: {len(pipeline.false_claims_archive)}")
    print(f"  Graph nodes: {pipeline.graph.number_of_nodes()}")

    print("\n✓ Verification Pipeline working!")
