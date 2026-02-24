"""
Graph Update Module for Expert Verification

Handles updates to the knowledge graph based on verification results:
1. Creating verification prior nodes
2. Connecting accepted claims with anchor links
3. Archiving and removing rejected claims
4. Updating prior-entailed claims set
5. Recomputing closeness centrality

This module works with NetworkX graphs and integrates with the
weighted bipartite graph structure from Phase 1.
"""

import json
import networkx as nx
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple, Any

from src.expert_verification import VerificationResult


# ==============================================================================
# EDGE WEIGHT CONSTANTS
# ==============================================================================

EDGE_WEIGHTS = {
    'anchor': 3.0,      # Prior -> Claim (ground truth / verified)
    'agree': 2.0,       # Response -> Claim (agrees with prior)
    'neutral': 1.0,     # Response -> Claim (no prior reference)
    'contra': 0.5       # Response -> Claim (contradicts prior)
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ArchivedClaim:
    """
    Archived claim that was rejected and removed from the graph.

    Stores all information needed to potentially restore the claim
    or analyze rejection patterns.
    """
    claim_id: str
    claim_text: str
    node_data: dict
    edges: List[Tuple[str, str, dict]]  # List of (u, v, edge_data)
    rejection_reason: str
    archived_at: str = field(default_factory=lambda: datetime.now().isoformat())
    verification_round: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'claim_id': self.claim_id,
            'claim_text': self.claim_text,
            'node_data': self.node_data,
            'edges': [(e[0], e[1], e[2]) for e in self.edges],
            'rejection_reason': self.rejection_reason,
            'archived_at': self.archived_at,
            'verification_round': self.verification_round
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ArchivedClaim':
        """Create from dictionary."""
        return cls(
            claim_id=data['claim_id'],
            claim_text=data['claim_text'],
            node_data=data['node_data'],
            edges=[(e[0], e[1], e[2]) for e in data['edges']],
            rejection_reason=data['rejection_reason'],
            archived_at=data.get('archived_at', datetime.now().isoformat()),
            verification_round=data.get('verification_round')
        )


@dataclass
class VerificationRoundSummary:
    """
    Summary of a verification round.

    Contains statistics and identifiers for tracking verification progress.
    """
    round_number: int
    verification_prior_id: str
    total_verified: int
    accepted_count: int
    rejected_count: int
    accepted_claim_ids: List[str]
    rejected_claim_ids: List[str]
    new_edges_added: int
    total_prior_entailed: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    verifier_type: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'round_number': self.round_number,
            'verification_prior_id': self.verification_prior_id,
            'total_verified': self.total_verified,
            'accepted_count': self.accepted_count,
            'rejected_count': self.rejected_count,
            'accepted_claim_ids': self.accepted_claim_ids,
            'rejected_claim_ids': self.rejected_claim_ids,
            'new_edges_added': self.new_edges_added,
            'total_prior_entailed': self.total_prior_entailed,
            'timestamp': self.timestamp,
            'verifier_type': self.verifier_type,
            'metadata': self.metadata
        }


# ==============================================================================
# GRAPH UPDATE FUNCTIONS
# ==============================================================================

def create_verification_prior(
    graph: nx.Graph,
    verification_round: int,
    verifier_type: str = "expert",
    verifier_model: Optional[str] = None,
    metadata: Optional[dict] = None
) -> str:
    """
    Create a new prior node representing an expert verification event.

    This node serves as an anchor point for claims that were verified
    and accepted by the expert.

    Args:
        graph: NetworkX graph
        verification_round: Integer identifier for this verification round
        verifier_type: Type of verifier ('llm', 'human', etc.)
        verifier_model: Model identifier if LLM verifier
        metadata: Optional additional metadata

    Returns:
        prior_node_id: ID of the new prior node
    """
    prior_node_id = f"P_verification_round_{verification_round}"

    node_attrs = {
        'type': 'prior',
        'subtype': 'expert_verification',
        'bipartite': 0,  # Source node in bipartite graph
        'verification_round': verification_round,
        'verifier_type': verifier_type,
        'created_at': datetime.now().isoformat(),
    }

    if verifier_model:
        node_attrs['verifier_model'] = verifier_model

    if metadata:
        node_attrs.update(metadata)

    graph.add_node(prior_node_id, **node_attrs)

    return prior_node_id


def connect_accepted_claims(
    graph: nx.Graph,
    verification_prior_id: str,
    accepted_claim_ids: List[str],
    w_anchor: float = EDGE_WEIGHTS['anchor']
) -> List[Tuple[str, str]]:
    """
    Connect accepted claims to the verification prior with anchor links.

    Creates high-weight edges from the verification prior to each accepted
    claim, establishing them as grounded/verified knowledge.

    Args:
        graph: NetworkX graph
        verification_prior_id: ID of the verification prior node
        accepted_claim_ids: List of claim IDs that were accepted
        w_anchor: Edge weight for anchor links (default: 3.0)

    Returns:
        new_edges: List of (prior, claim) edge tuples added
    """
    new_edges = []

    for claim_id in accepted_claim_ids:
        if claim_id not in graph.nodes:
            print(f"Warning: Claim {claim_id} not found in graph, skipping")
            continue

        # Add anchor edge from verification prior to claim
        graph.add_edge(
            verification_prior_id,
            claim_id,
            weight=w_anchor,
            cost=1.0 / w_anchor,
            link_type='anchor',
            source='expert_verification'
        )
        new_edges.append((verification_prior_id, claim_id))

        # Update claim node metadata
        if 'verified' not in graph.nodes[claim_id]:
            graph.nodes[claim_id]['verified'] = True
        graph.nodes[claim_id]['verified_as'] = 'accepted'
        graph.nodes[claim_id]['verification_prior'] = verification_prior_id
        graph.nodes[claim_id]['verified_at'] = datetime.now().isoformat()

    return new_edges


def archive_and_remove_rejected_claims(
    graph: nx.Graph,
    rejected_claim_ids: List[str],
    rejection_reasons: Optional[Dict[str, str]] = None,
    verification_round: Optional[int] = None
) -> List[ArchivedClaim]:
    """
    Archive rejected claims and remove them from the graph.

    Stores complete information about each rejected claim before removal,
    enabling later analysis or potential restoration.

    Args:
        graph: NetworkX graph
        rejected_claim_ids: List of claim IDs to reject and remove
        rejection_reasons: Optional dict mapping claim_id -> rejection reasoning
        verification_round: Optional round number for archive metadata

    Returns:
        archived_claims: List of ArchivedClaim objects
    """
    archived_claims = []
    rejection_reasons = rejection_reasons or {}

    for claim_id in rejected_claim_ids:
        if claim_id not in graph.nodes:
            print(f"Warning: Claim {claim_id} not found in graph, skipping")
            continue

        # Store claim info before removal
        node_data = dict(graph.nodes[claim_id])
        claim_text = node_data.get('text', node_data.get('claim_text', ''))

        # Store all edges
        edges = []
        for neighbor in list(graph.neighbors(claim_id)):
            edge_data = dict(graph.edges[claim_id, neighbor])
            edges.append((claim_id, neighbor, edge_data))

        # For directed graphs, also check predecessors
        if graph.is_directed():
            for predecessor in list(graph.predecessors(claim_id)):
                if predecessor != claim_id:  # Avoid self-loops counted twice
                    edge_data = dict(graph.edges[predecessor, claim_id])
                    edges.append((predecessor, claim_id, edge_data))

        archived_claim = ArchivedClaim(
            claim_id=claim_id,
            claim_text=claim_text,
            node_data=node_data,
            edges=edges,
            rejection_reason=rejection_reasons.get(claim_id, "Expert rejected"),
            verification_round=verification_round
        )
        archived_claims.append(archived_claim)

        # Remove claim from graph
        graph.remove_node(claim_id)

    return archived_claims


def update_prior_entailed_claims(
    prior_entailed_claim_ids: Set[str],
    accepted_claim_ids: List[str],
    rejected_claim_ids: List[str]
) -> Set[str]:
    """
    Update the set of prior-entailed claims after verification.

    Args:
        prior_entailed_claim_ids: Current set of prior-entailed claim IDs
        accepted_claim_ids: Newly accepted claim IDs (add to set)
        rejected_claim_ids: Rejected claim IDs (ensure removed)

    Returns:
        Updated set of prior-entailed claim IDs
    """
    # Create a copy to avoid modifying the original
    updated = set(prior_entailed_claim_ids)

    # Add accepted claims
    updated.update(accepted_claim_ids)

    # Ensure rejected claims are not in the set
    updated.difference_update(rejected_claim_ids)

    return updated


def recompute_closeness_centrality(
    graph: nx.Graph,
    claim_node_ids: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Recompute grounded closeness centrality after graph update.

    Uses weighted shortest paths where edge cost = 1/weight, so
    higher-weight edges (anchors) contribute to shorter paths.

    Args:
        graph: Updated NetworkX graph
        claim_node_ids: Optional list of claim IDs. If None, auto-detect.

    Returns:
        closeness_centrality: Dict mapping claim_id -> CC score
    """
    # Ensure cost attribute is set on all edges
    for u, v, data in graph.edges(data=True):
        if 'cost' not in data:
            weight = data.get('weight', 1.0)
            if weight > 0:
                data['cost'] = 1.0 / weight
            else:
                data['cost'] = float('inf')

    # Auto-detect claim nodes if not provided
    if claim_node_ids is None:
        claim_node_ids = [
            n for n in graph.nodes()
            if graph.nodes[n].get('type') == 'claim' or
               graph.nodes[n].get('bipartite') == 1 or
               str(n).startswith('C')
        ]

    if len(graph.nodes()) == 0:
        return {}

    # Compute closeness centrality using cost as distance
    try:
        cc = nx.closeness_centrality(graph, distance='cost')
    except Exception:
        # Fallback to unweighted if weighted fails
        cc = nx.closeness_centrality(graph)

    # Filter to only claim nodes that still exist
    closeness_centrality = {
        cid: cc.get(cid, 0.0)
        for cid in claim_node_ids
        if cid in graph.nodes
    }

    return closeness_centrality


def update_response_edges_for_accepted_claims(
    graph: nx.Graph,
    accepted_claim_ids: List[str],
    w_agree: float = EDGE_WEIGHTS['agree']
) -> int:
    """
    Update edge weights from responses to accepted claims.

    After a claim is verified, edges from responses that support it
    can be upgraded to "agreement" links (higher weight).

    Args:
        graph: NetworkX graph
        accepted_claim_ids: List of accepted claim IDs
        w_agree: New weight for agreement links

    Returns:
        Number of edges updated
    """
    edges_updated = 0

    for claim_id in accepted_claim_ids:
        if claim_id not in graph.nodes:
            continue

        for neighbor in graph.neighbors(claim_id):
            # Check if neighbor is a response node
            node_data = graph.nodes[neighbor]
            if node_data.get('type') == 'response' or str(neighbor).startswith('S'):
                edge_data = graph.edges[neighbor, claim_id]

                # Upgrade neutral links to agreement links
                if edge_data.get('link_type') == 'neutral':
                    edge_data['weight'] = w_agree
                    edge_data['cost'] = 1.0 / w_agree
                    edge_data['link_type'] = 'agree'
                    edges_updated += 1

    return edges_updated


# ==============================================================================
# GRAPH UPDATER CLASS
# ==============================================================================

class GraphUpdater:
    """
    Handles graph updates after expert verification.

    Manages the complete update workflow:
    1. Creates verification prior nodes
    2. Connects accepted claims
    3. Archives and removes rejected claims
    4. Updates tracking sets
    5. Recomputes centrality metrics
    """

    def __init__(
        self,
        graph: nx.Graph,
        prior_entailed_claim_ids: Set[str],
        w_anchor: float = EDGE_WEIGHTS['anchor'],
        w_agree: float = EDGE_WEIGHTS['agree'],
        upgrade_response_edges: bool = True
    ):
        """
        Initialize the graph updater.

        Args:
            graph: NetworkX bipartite graph
            prior_entailed_claim_ids: Current set of prior-entailed claim IDs
            w_anchor: Edge weight for anchor links
            w_agree: Edge weight for agreement links
            upgrade_response_edges: Whether to upgrade response edges for accepted claims
        """
        self.graph = graph
        self.prior_entailed_claim_ids = set(prior_entailed_claim_ids)
        self.w_anchor = w_anchor
        self.w_agree = w_agree
        self.upgrade_response_edges = upgrade_response_edges

        # Archives and history
        self.false_claims_archive: List[ArchivedClaim] = []
        self.verification_history: List[VerificationRoundSummary] = []
        self.verification_round = 0

    def process_verification_results(
        self,
        verification_results: List[VerificationResult],
        verifier_type: str = "expert",
        verifier_model: Optional[str] = None
    ) -> Tuple[VerificationRoundSummary, Dict[str, float]]:
        """
        Process verification results and update the graph.

        Args:
            verification_results: List of VerificationResult from verifier
            verifier_type: Type of verifier used
            verifier_model: Model identifier if LLM verifier

        Returns:
            Tuple of (VerificationRoundSummary, new_closeness_centrality)
        """
        self.verification_round += 1

        # Separate accepted and rejected claims
        accepted_results = [r for r in verification_results if r.verdict is True]
        rejected_results = [r for r in verification_results if r.verdict is False]

        accepted_claim_ids = [r.claim_id for r in accepted_results]
        rejected_claim_ids = [r.claim_id for r in rejected_results]

        # Build rejection reasons dict
        rejection_reasons = {
            r.claim_id: r.reasoning for r in rejected_results
        }

        # Step 1: Create verification prior node
        verification_prior_id = create_verification_prior(
            self.graph,
            self.verification_round,
            verifier_type=verifier_type,
            verifier_model=verifier_model,
            metadata={
                'total_verified': len(verification_results),
                'accepted_count': len(accepted_claim_ids),
                'rejected_count': len(rejected_claim_ids)
            }
        )

        # Step 2: Connect accepted claims with anchor links
        new_edges = connect_accepted_claims(
            self.graph,
            verification_prior_id,
            accepted_claim_ids,
            self.w_anchor
        )

        # Step 3: Optionally upgrade response edges
        if self.upgrade_response_edges:
            update_response_edges_for_accepted_claims(
                self.graph,
                accepted_claim_ids,
                self.w_agree
            )

        # Step 4: Archive and remove rejected claims
        archived = archive_and_remove_rejected_claims(
            self.graph,
            rejected_claim_ids,
            rejection_reasons,
            self.verification_round
        )
        self.false_claims_archive.extend(archived)

        # Step 5: Update prior-entailed claims set
        self.prior_entailed_claim_ids = update_prior_entailed_claims(
            self.prior_entailed_claim_ids,
            accepted_claim_ids,
            rejected_claim_ids
        )

        # Step 6: Recompute closeness centrality
        new_closeness_centrality = recompute_closeness_centrality(self.graph)

        # Create summary
        summary = VerificationRoundSummary(
            round_number=self.verification_round,
            verification_prior_id=verification_prior_id,
            total_verified=len(verification_results),
            accepted_count=len(accepted_claim_ids),
            rejected_count=len(rejected_claim_ids),
            accepted_claim_ids=accepted_claim_ids,
            rejected_claim_ids=rejected_claim_ids,
            new_edges_added=len(new_edges),
            total_prior_entailed=len(self.prior_entailed_claim_ids),
            verifier_type=verifier_type,
            metadata={'verifier_model': verifier_model} if verifier_model else {}
        )

        self.verification_history.append(summary)

        return summary, new_closeness_centrality

    def get_current_state(self) -> dict:
        """Get current state of the graph updater."""
        return {
            'verification_round': self.verification_round,
            'total_prior_entailed': len(self.prior_entailed_claim_ids),
            'total_archived_false_claims': len(self.false_claims_archive),
            'graph_num_nodes': self.graph.number_of_nodes(),
            'graph_num_edges': self.graph.number_of_edges()
        }

    def get_remaining_contested_claims(self, all_claim_ids: List[str]) -> List[str]:
        """
        Get claim IDs that are still contested (not yet prior-entailed).

        Args:
            all_claim_ids: List of all claim IDs in the original set

        Returns:
            List of claim IDs not in prior_entailed_claim_ids
        """
        return [
            cid for cid in all_claim_ids
            if cid in self.graph.nodes and cid not in self.prior_entailed_claim_ids
        ]

    def export_archive(self, filepath: str):
        """Export false claims archive to JSON file."""
        archive_data = [ac.to_dict() for ac in self.false_claims_archive]

        with open(filepath, 'w') as f:
            json.dump(archive_data, f, indent=2, default=str)

    def export_verification_history(self, filepath: str):
        """Export verification history to JSON file."""
        history_data = [s.to_dict() for s in self.verification_history]

        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)

    def restore_archived_claim(self, claim_id: str) -> bool:
        """
        Restore a previously archived claim back to the graph.

        Args:
            claim_id: ID of the claim to restore

        Returns:
            True if restored, False if not found in archive
        """
        for i, archived in enumerate(self.false_claims_archive):
            if archived.claim_id == claim_id:
                # Restore node
                self.graph.add_node(claim_id, **archived.node_data)

                # Restore edges (only if other node still exists)
                for u, v, edge_data in archived.edges:
                    if u in self.graph.nodes and v in self.graph.nodes:
                        self.graph.add_edge(u, v, **edge_data)

                # Remove from archive
                self.false_claims_archive.pop(i)
                return True

        return False


if __name__ == "__main__":
    # Test the graph update module
    print("Testing Graph Update Module...")

    # Create a simple test graph
    G = nx.Graph()

    # Add source nodes
    G.add_node('S0', type='prior', bipartite=0)
    G.add_node('S1', type='response', bipartite=0)
    G.add_node('S2', type='response', bipartite=0)

    # Add claim nodes
    for i in range(5):
        G.add_node(f'C{i}', type='claim', bipartite=1, text=f'Claim {i}')

    # Add edges
    G.add_edge('S0', 'C0', weight=3.0, cost=1/3, link_type='anchor')
    G.add_edge('S1', 'C0', weight=2.0, cost=0.5, link_type='agree')
    G.add_edge('S1', 'C1', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S1', 'C2', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S2', 'C2', weight=1.0, cost=1.0, link_type='neutral')
    G.add_edge('S2', 'C3', weight=0.5, cost=2.0, link_type='contra')
    G.add_edge('S2', 'C4', weight=1.0, cost=1.0, link_type='neutral')

    print(f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Initialize updater
    updater = GraphUpdater(
        graph=G,
        prior_entailed_claim_ids={'C0'}
    )

    # Create mock verification results
    results = [
        VerificationResult(
            claim_id='C1',
            claim_text='Claim 1',
            verdict=True,
            reasoning='Verified correct'
        ),
        VerificationResult(
            claim_id='C2',
            claim_text='Claim 2',
            verdict=True,
            reasoning='Verified correct'
        ),
        VerificationResult(
            claim_id='C3',
            claim_text='Claim 3',
            verdict=False,
            reasoning='Contradicts known facts'
        ),
    ]

    # Process results
    summary, new_cc = updater.process_verification_results(results, verifier_type='mock')

    print(f"\nVerification Round {summary.round_number}:")
    print(f"  Accepted: {summary.accepted_count} ({summary.accepted_claim_ids})")
    print(f"  Rejected: {summary.rejected_count} ({summary.rejected_claim_ids})")
    print(f"  Total prior-entailed: {summary.total_prior_entailed}")

    print(f"\nUpdated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Archived claims: {len(updater.false_claims_archive)}")

    state = updater.get_current_state()
    print(f"\nCurrent state: {state}")

    print("\n✓ Graph Update module working!")
