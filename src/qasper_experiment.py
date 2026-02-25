"""
QASPER Knowledge Expansion Experiment

Implements the progressive section expansion experiment on the QASPER dataset.
Key design principles:
1. Response generation happens ONLY in Iteration 0
2. Claims are FIXED across iterations
3. Subsequent iterations add sections as verified priors
4. Isolates the effect of prior knowledge on claim grounding
"""

import json
import os
import time
import networkx as nx
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field

from src.qasper_data import (
    PriorInfo,
    QuestionClassification,
    construct_prior,
    construct_expert_context,
    get_new_section_for_iteration,
    classify_question_answerability,
    validate_paper_structure,
    identify_section_index,
    normalize_qasper_questions,
    INTRODUCTION_VARIANTS,
    METHODS_VARIANTS
)
from src.expert_verification import (
    BaseExpertVerifier,
    MockExpertVerifier,
    VerificationResult,
    create_verifier
)
from src.graph_update import (
    GraphUpdater,
    VerificationRoundSummary,
    recompute_closeness_centrality,
    EDGE_WEIGHTS
)
from src.iterative_expansion import (
    IterativeKnowledgeExpansion,
    compute_centrality_scores,
    compute_novelty_scores,
    compute_priority_scores,
    select_claims_for_verification,
    z_normalize_positive
)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ResponseSample:
    """A single response sample for a question."""
    sample_id: str
    question_id: str
    text: str
    claims: List[Dict] = field(default_factory=list)


@dataclass
class QuestionResponses:
    """All response samples for a single question."""
    question_id: str
    question_text: str
    samples: List[ResponseSample] = field(default_factory=list)


@dataclass
class IterationMetrics:
    """Metrics computed for a single iteration."""
    prior_coverage_ratio: float  # Fraction of claims grounded in priors
    knowledge_frontier_size: int  # Number of contested claims
    question_answerability_rate: float  # Fraction answerable from prior
    verification_utility_ratio: float  # Accepted / verified
    total_claims: int
    grounded_claims: int
    contested_claims: int

    def to_dict(self) -> dict:
        return {
            "prior_coverage_ratio": self.prior_coverage_ratio,
            "knowledge_frontier_size": self.knowledge_frontier_size,
            "question_answerability_rate": self.question_answerability_rate,
            "verification_utility_ratio": self.verification_utility_ratio,
            "total_claims": self.total_claims,
            "grounded_claims": self.grounded_claims,
            "contested_claims": self.contested_claims
        }


@dataclass
class GraphSnapshot:
    """Snapshot of graph state at a particular iteration."""
    nodes: List[Dict]  # All nodes with attributes
    edges: List[Dict]  # All edges with weights
    prior_entailed_claims: List[str]  # Claim IDs grounded in priors
    contested_claims: List[str]  # Claim IDs still contested

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "prior_entailed_claims": self.prior_entailed_claims,
            "contested_claims": self.contested_claims
        }


@dataclass
class IterationResult:
    """Result of a single iteration."""
    iteration: int
    prior_info: PriorInfo
    num_total_claims: int
    num_contested_claims: int
    num_verified: int
    num_accepted: int
    num_rejected: int
    metrics: IterationMetrics
    question_classifications: List[QuestionClassification]
    verification_results: List[VerificationResult] = field(default_factory=list)
    graph_snapshot: Optional[GraphSnapshot] = None
    claims_status: List[Dict] = field(default_factory=list)  # Status of each claim this iteration
    selection_diagnostics: Dict = field(default_factory=dict)  # Claim selection diagnostics

    def to_dict(self) -> dict:
        result = {
            "iteration": self.iteration,
            "prior_sections": self.prior_info.sections_included,
            "prior_token_count": self.prior_info.token_count,
            "num_total_claims": self.num_total_claims,
            "num_contested_claims": self.num_contested_claims,
            "num_verified": self.num_verified,
            "num_accepted": self.num_accepted,
            "num_rejected": self.num_rejected,
            "metrics": self.metrics.to_dict(),
            "question_classifications": [q.to_dict() for q in self.question_classifications],
            "verification_results": [
                {
                    "claim_id": v.claim_id,
                    "verdict": v.verdict,
                    "reasoning": v.reasoning,
                    "confidence": getattr(v, 'confidence', None),
                    "evidence_used": getattr(v, 'evidence_used', None)
                }
                for v in self.verification_results
            ],
            "claims_status": self.claims_status,
            "selection_diagnostics": self._serialize_selection_diagnostics()
        }
        if self.graph_snapshot:
            result["graph_snapshot"] = self.graph_snapshot.to_dict()
        return result

    def _serialize_selection_diagnostics(self) -> Dict:
        """Serialize selection diagnostics, handling non-JSON-serializable types."""
        if not self.selection_diagnostics:
            return {}

        serialized = {}
        for key, value in self.selection_diagnostics.items():
            if isinstance(value, dict):
                # Convert any numpy types in nested dicts
                serialized[key] = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in value.items()
                }
            elif isinstance(value, (list, tuple)):
                serialized[key] = list(value)
            elif hasattr(value, 'item'):  # numpy scalar
                serialized[key] = float(value)
            else:
                serialized[key] = value
        return serialized


@dataclass
class PaperExperimentResult:
    """Complete result for a single paper experiment."""
    paper_id: str
    paper_title: str
    iterations: List[IterationResult]
    responses: List[QuestionResponses]
    all_claims: List[Dict]
    final_metrics: Dict
    final_graph_snapshot: Optional[GraphSnapshot] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        # Serialize responses with full text
        responses_data = []
        for qr in self.responses:
            qr_data = {
                "question_id": qr.question_id,
                "question_text": qr.question_text,
                "samples": [
                    {
                        "sample_id": s.sample_id,
                        "text": s.text,
                        "claims": [
                            {"text": c.get("text", c) if isinstance(c, dict) else str(c)}
                            for c in s.claims
                        ] if s.claims else []
                    }
                    for s in qr.samples
                ]
            }
            responses_data.append(qr_data)

        # Serialize all claims with full details
        claims_data = [
            {
                "id": c.get("id"),
                "text": c.get("text"),
                "source": c.get("source"),
                "source_type": c.get("source_type"),
                "source_section": c.get("source_section", ""),
                "source_paragraph": c.get("source_paragraph", ""),
                "question_id": c.get("question_id"),
                "sample_id": c.get("sample_id"),
                "confidence": c.get("confidence")
            }
            for c in self.all_claims
        ]

        result = {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "iterations": [it.to_dict() for it in self.iterations],
            "responses": responses_data,
            "claims": claims_data,
            "num_responses": sum(len(qr.samples) for qr in self.responses),
            "num_claims": len(self.all_claims),
            "final_metrics": self.final_metrics,
            "error": self.error
        }

        if self.final_graph_snapshot:
            result["final_graph"] = self.final_graph_snapshot.to_dict()

        return result


# ==============================================================================
# QASPER EXPERIMENT CLASS
# ==============================================================================

class QASPERExperiment:
    """
    Run knowledge expansion experiment on QASPER dataset.

    Key Design:
    - Response generation happens ONLY in Iteration 0
    - Subsequent iterations add sections as priors (no new generation)
    - Claims remain fixed; only their grounding status changes
    """

    def __init__(
        self,
        generator_model: Any,  # BaseModel from src/models.py
        claim_extractor: Any,  # BreakdownProcessor from src/break_and_merge.py
        expert_verifier: Optional[BaseExpertVerifier] = None,
        verifier_type: str = "mock",
        verifier_kwargs: Optional[dict] = None,
        embedding_model: Any = None,  # EmbeddingModel from src/embeddings.py
        w_anchor: float = EDGE_WEIGHTS['anchor'],
        w_agree: float = EDGE_WEIGHTS['agree'],
        w_neutral: float = EDGE_WEIGHTS['neutral'],
        w_contra: float = EDGE_WEIGHTS['contra'],
        delta_true: float = 0.6,  # Threshold for grounded
        delta_false: float = 0.2,  # Threshold for contradictory
        output_dir: Optional[str] = None
    ):
        """
        Initialize the QASPER experiment.

        Args:
            generator_model: LLM model for generating responses
            claim_extractor: Processor for extracting claims from text
            expert_verifier: Pre-configured verifier (optional)
            verifier_type: Type of verifier ('llm', 'mock', etc.)
            verifier_kwargs: Arguments for verifier constructor
            embedding_model: Model for computing embeddings
            w_anchor, w_agree, w_neutral, w_contra: Edge weights
            delta_true, delta_false: Claim categorization thresholds
            output_dir: Directory for saving results
        """
        self.generator_model = generator_model
        self.claim_extractor = claim_extractor
        self.embedding_model = embedding_model

        # Edge weights
        self.w_anchor = w_anchor
        self.w_agree = w_agree
        self.w_neutral = w_neutral
        self.w_contra = w_contra

        # Thresholds
        self.delta_true = delta_true
        self.delta_false = delta_false

        # Initialize verifier
        if expert_verifier is not None:
            self.expert_verifier = expert_verifier
        else:
            verifier_kwargs = verifier_kwargs or {}
            self.expert_verifier = create_verifier(verifier_type, **verifier_kwargs)

        self.output_dir = output_dir
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _create_graph_snapshot(
        self,
        graph: nx.Graph,
        prior_entailed: Set[str],
        all_claims: List[Dict]
    ) -> GraphSnapshot:
        """
        Create a serializable snapshot of the current graph state.

        Args:
            graph: NetworkX graph
            prior_entailed: Set of claim IDs that are grounded
            all_claims: List of all claim dicts

        Returns:
            GraphSnapshot with nodes, edges, and claim status
        """
        # Serialize nodes
        nodes = []
        for node_id, attrs in graph.nodes(data=True):
            node_data = {
                "id": node_id,
                "type": attrs.get("type", "unknown"),
                "bipartite": attrs.get("bipartite"),
                "text": attrs.get("text", "")[:200],  # Truncate for readability
            }
            # Add optional attributes
            if "section_name" in attrs:
                node_data["section_name"] = attrs["section_name"]
            if "iteration" in attrs:
                node_data["iteration"] = attrs["iteration"]
            if "question_id" in attrs:
                node_data["question_id"] = attrs["question_id"]
            if "sample_id" in attrs:
                node_data["sample_id"] = attrs["sample_id"]
            if "source_type" in attrs:
                node_data["source_type"] = attrs["source_type"]
            if "verification_round" in attrs:
                node_data["verification_round"] = attrs["verification_round"]
            nodes.append(node_data)

        # Serialize edges
        edges = []
        for u, v, attrs in graph.edges(data=True):
            edge_data = {
                "source": u,
                "target": v,
                "weight": attrs.get("weight", 1.0),
                "cost": attrs.get("cost"),
                "link_type": attrs.get("link_type", "unknown")
            }
            edges.append(edge_data)

        # Get claim statuses
        prior_entailed_list = list(prior_entailed)
        contested_list = [c["id"] for c in all_claims if c["id"] not in prior_entailed]

        return GraphSnapshot(
            nodes=nodes,
            edges=edges,
            prior_entailed_claims=prior_entailed_list,
            contested_claims=contested_list
        )

    def _get_claims_status(
        self,
        all_claims: List[Dict],
        prior_entailed: Set[str],
        verification_results: List[VerificationResult]
    ) -> List[Dict]:
        """
        Get the status of each claim for this iteration.

        Returns list of dicts with claim id, text, and current status.
        """
        # Build lookup for verification results
        verified_claims = {v.claim_id: v for v in verification_results}

        claims_status = []
        for claim in all_claims:
            claim_id = claim["id"]
            status = {
                "id": claim_id,
                "text": claim.get("text", ""),
                "source_type": claim.get("source_type", "unknown"),
                "is_grounded": claim_id in prior_entailed,
                "was_verified_this_iteration": claim_id in verified_claims
            }

            if claim_id in verified_claims:
                v_result = verified_claims[claim_id]
                status["verification_verdict"] = v_result.verdict
                status["verification_reasoning"] = v_result.reasoning

            claims_status.append(status)

        return claims_status

    def run_single_paper_experiment(
        self,
        paper: dict,
        num_iterations: int = 3,
        claims_per_iteration: int = 10,
        response_samples_per_question: int = 3,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> PaperExperimentResult:
        """
        Run full experiment on a single paper.

        Args:
            paper: QASPER paper dict
            num_iterations: Number of expansion iterations (0, 1, 2)
            claims_per_iteration: Number of claims to verify per iteration
            response_samples_per_question: Number of response samples per question
            progress_callback: Optional callback for progress updates

        Returns:
            PaperExperimentResult with all iteration data
        """
        paper_id = paper["id"]
        paper_title = paper["title"]

        if progress_callback:
            progress_callback(f"Processing paper: {paper_id}")

        try:
            # Get expert context (full paper) for verification
            expert_context = construct_expert_context(paper)
            questions = normalize_qasper_questions(paper)

            # ==============================================================
            # ITERATION 0: Generate responses and build initial graph
            # ==============================================================

            if progress_callback:
                progress_callback("Iteration 0: Generating responses...")

            # Construct P_0 (title + abstract only)
            prior_0 = construct_prior(paper, iteration=0)

            # Generate responses ONCE using P_0
            responses = self._generate_responses(
                questions=questions,
                prior_text=prior_0.text,
                num_samples=response_samples_per_question
            )

            if progress_callback:
                progress_callback(f"Generated {sum(len(r.samples) for r in responses)} responses")

            # Extract claims from responses
            all_claims = self._extract_all_claims(prior_0.text, responses)

            if progress_callback:
                progress_callback(f"Extracted {len(all_claims)} claims")

            # Build initial graph
            graph, prior_entailed = self._build_initial_graph(
                prior_text=prior_0.text,
                responses=responses,
                all_claims=all_claims
            )

            # Store fixed responses and claims
            fixed_responses = responses
            fixed_claims = all_claims

            iteration_results = []

            # ==============================================================
            # ALL ITERATIONS: Add priors, verify claims, measure metrics
            # ==============================================================

            for iteration in range(num_iterations):
                if progress_callback:
                    progress_callback(f"Iteration {iteration}: Processing...")

                # Construct prior for this iteration
                prior_info = construct_prior(paper, iteration)

                # Classify questions by answerability with current prior
                question_classifications = [
                    classify_question_answerability(q, prior_info.sections_included, paper)
                    for q in questions
                ]

                if iteration > 0:
                    # Add new section as verified prior node
                    new_section = get_new_section_for_iteration(paper, iteration)
                    if new_section:
                        section_name, section_text = new_section
                        self._add_section_as_prior(
                            graph=graph,
                            prior_entailed=prior_entailed,
                            all_claims=all_claims,
                            section_name=section_name,
                            section_text=section_text,
                            iteration=iteration
                        )

                # Get contested claims
                contested_claims = self._get_contested_claims(
                    graph, prior_entailed, all_claims
                )

                # Get response node IDs for claim selection
                response_node_ids = [
                    n for n in graph.nodes()
                    if graph.nodes[n].get("type") == "response"
                ]

                # Select claims for verification using utility-based ranking
                selected_claims, selection_diagnostics = self._select_claims_for_verification(
                    contested_claims,
                    k=claims_per_iteration,
                    graph=graph,
                    prior_entailed=prior_entailed,
                    response_node_ids=response_node_ids
                )

                # Expert verification
                verification_results = []
                if selected_claims:
                    claims_to_verify = [
                        {'claim_id': c['id'], 'claim_text': c['text']}
                        for c in selected_claims
                    ]
                    verification_results = self.expert_verifier.verify_batch(
                        claims=claims_to_verify,
                        context=expert_context
                    )

                    # Update graph with verification results
                    self._apply_verification_results(
                        graph=graph,
                        prior_entailed=prior_entailed,
                        verification_results=verification_results,
                        iteration=iteration
                    )

                # Compute metrics
                metrics = self._compute_iteration_metrics(
                    graph=graph,
                    prior_entailed=prior_entailed,
                    all_claims=all_claims,
                    question_classifications=question_classifications,
                    verification_results=verification_results
                )

                # Record iteration result
                accepted = sum(1 for v in verification_results if v.verdict)
                rejected = sum(1 for v in verification_results if not v.verdict)

                # Create graph snapshot for this iteration
                graph_snapshot = self._create_graph_snapshot(
                    graph=graph,
                    prior_entailed=prior_entailed,
                    all_claims=all_claims
                )

                # Get claims status for this iteration
                claims_status = self._get_claims_status(
                    all_claims=all_claims,
                    prior_entailed=prior_entailed,
                    verification_results=verification_results
                )

                iteration_result = IterationResult(
                    iteration=iteration,
                    prior_info=prior_info,
                    num_total_claims=len(fixed_claims),
                    num_contested_claims=len(contested_claims),
                    num_verified=len(verification_results),
                    num_accepted=accepted,
                    num_rejected=rejected,
                    metrics=metrics,
                    question_classifications=question_classifications,
                    verification_results=verification_results,
                    graph_snapshot=graph_snapshot,
                    claims_status=claims_status,
                    selection_diagnostics=selection_diagnostics
                )
                iteration_results.append(iteration_result)

                if progress_callback:
                    progress_callback(
                        f"Iteration {iteration}: PCR={metrics.prior_coverage_ratio:.3f}, "
                        f"KF={metrics.knowledge_frontier_size}"
                    )

            # Compute final metrics
            final_metrics = self._compute_final_metrics(
                graph=graph,
                prior_entailed=prior_entailed,
                all_claims=all_claims,
                questions=questions
            )

            # Create final graph snapshot
            final_graph_snapshot = self._create_graph_snapshot(
                graph=graph,
                prior_entailed=prior_entailed,
                all_claims=all_claims
            )

            return PaperExperimentResult(
                paper_id=paper_id,
                paper_title=paper_title,
                iterations=iteration_results,
                responses=fixed_responses,
                all_claims=fixed_claims,
                final_metrics=final_metrics,
                final_graph_snapshot=final_graph_snapshot
            )

        except Exception as e:
            return PaperExperimentResult(
                paper_id=paper_id,
                paper_title=paper_title,
                iterations=[],
                responses=[],
                all_claims=[],
                final_metrics={},
                error=str(e)
            )

    def _generate_responses(
        self,
        questions: List[dict],
        prior_text: str,
        num_samples: int = 3
    ) -> List[QuestionResponses]:
        """
        Generate LLM responses for each question using only the prior.

        IMPORTANT: This is called ONLY ONCE in Iteration 0.
        Each response answers a SINGLE question.
        """
        all_responses = []

        for q in questions:
            question_id = q.get("question_id", "")
            question_text = q.get("question", "")

            question_responses = QuestionResponses(
                question_id=question_id,
                question_text=question_text
            )

            for sample_idx in range(num_samples):
                prompt = f"""Based on the following paper content, answer the question.
If the answer cannot be determined from the given content, say so and provide
your best reasoning about what the answer might be.

Paper Content:
{prior_text}

Question: {question_text}

Answer:"""

                # Generate response
                if self.generator_model is not None:
                    response_result = self.generator_model.generate_given_prompt(prompt)
                    # Handle both dict (from models.py) and string returns
                    if isinstance(response_result, dict):
                        response_text = response_result.get('generation', str(response_result))
                    else:
                        response_text = str(response_result)
                else:
                    # Mock response for testing
                    response_text = f"Mock response for question: {question_text[:50]}"

                sample = ResponseSample(
                    sample_id=f"{question_id}_sample_{sample_idx}",
                    question_id=question_id,
                    text=response_text
                )
                question_responses.samples.append(sample)

            all_responses.append(question_responses)

        return all_responses

    def _extract_all_claims(
        self,
        prior_text: str,
        responses: List[QuestionResponses]
    ) -> List[Dict]:
        """
        Extract claims from prior and all responses.
        """
        all_claims = []
        claim_id_counter = 0

        # Extract claims from prior
        if self.claim_extractor is not None:
            prior_claims = self._extract_claims_from_text(prior_text, "prior")
        else:
            # Mock extraction
            prior_claims = [
                {"text": f"Prior claim {i}", "source": "prior", "source_type": "prior"}
                for i in range(3)
            ]

        for claim in prior_claims:
            claim["id"] = f"C{claim_id_counter}"
            claim["source_type"] = "prior"
            claim["source_section"] = "abstract"
            claim["source_paragraph"] = prior_text
            all_claims.append(claim)
            claim_id_counter += 1

        # Extract claims from each response
        for qr in responses:
            for sample in qr.samples:
                if self.claim_extractor is not None:
                    sample_claims = self._extract_claims_from_text(
                        sample.text, sample.sample_id
                    )
                else:
                    # Mock extraction
                    sample_claims = [
                        {"text": f"Response claim from {sample.sample_id}",
                         "source": sample.sample_id}
                    ]

                for claim in sample_claims:
                    claim["id"] = f"C{claim_id_counter}"
                    claim["source_type"] = "response"
                    claim["question_id"] = qr.question_id
                    claim["sample_id"] = sample.sample_id
                    all_claims.append(claim)
                    claim_id_counter += 1

                sample.claims = sample_claims

        return all_claims

    def _extract_claims_from_text(self, text: str, source_id: str) -> List[Dict]:
        """
        Extract claims from a single text using the claim extractor.
        """
        if self.claim_extractor is None:
            return []

        # Use the claim extractor interface
        # This matches the BreakdownProcessor interface
        try:
            claims = self.claim_extractor.extract_claims(text)
            return [{"text": c, "source": source_id} for c in claims]
        except AttributeError:
            # Try alternative interface
            try:
                result = self.claim_extractor.break_down_single(
                    {"text": text}, source_id, {}
                )
                if result and "breakdown" in result:
                    return [
                        {"text": c.get("claim", c), "source": source_id}
                        for c in result["breakdown"]
                    ]
            except Exception:
                pass
        return []

    def _build_initial_graph(
        self,
        prior_text: str,
        responses: List[QuestionResponses],
        all_claims: List[Dict]
    ) -> Tuple[nx.Graph, Set[str]]:
        """
        Build initial weighted bipartite graph.
        """
        G = nx.Graph()

        # Add prior node
        G.add_node("S_prior", type="prior", bipartite=0, text=prior_text[:500])

        # Add response nodes
        response_idx = 0
        for qr in responses:
            for sample in qr.samples:
                node_id = f"S_R{response_idx}"
                G.add_node(
                    node_id,
                    type="response",
                    bipartite=0,
                    text=sample.text[:500],
                    question_id=qr.question_id,
                    sample_id=sample.sample_id
                )
                response_idx += 1

        # Add claim nodes
        for claim in all_claims:
            G.add_node(
                claim["id"],
                type="claim",
                bipartite=1,
                text=claim["text"],
                source_type=claim["source_type"],
                source_section=claim.get("source_section", ""),
                source_paragraph=claim.get("source_paragraph", "")
            )

        # Add edges
        prior_entailed = set()

        for claim in all_claims:
            if claim["source_type"] == "prior":
                # Anchor edge from prior to prior claims
                G.add_edge(
                    "S_prior",
                    claim["id"],
                    weight=self.w_anchor,
                    cost=1 / self.w_anchor,
                    link_type="anchor"
                )
                prior_entailed.add(claim["id"])
            else:
                # Find the response node for this claim
                sample_id = claim.get("sample_id", "")
                response_node = None
                response_idx = 0
                for qr in responses:
                    for sample in qr.samples:
                        if sample.sample_id == sample_id:
                            response_node = f"S_R{response_idx}"
                            break
                        response_idx += 1
                    if response_node:
                        break

                if response_node:
                    # Determine edge type based on similarity to prior claims
                    # For now, use neutral weight
                    G.add_edge(
                        response_node,
                        claim["id"],
                        weight=self.w_neutral,
                        cost=1 / self.w_neutral,
                        link_type="neutral"
                    )

        return G, prior_entailed

    def _add_section_as_prior(
        self,
        graph: nx.Graph,
        prior_entailed: Set[str],
        all_claims: List[Dict],
        section_name: str,
        section_text: str,
        iteration: int
    ):
        """
        Add a newly revealed section as a prior node to existing graph.
        This simulates "verifying" that section's content is accurate.
        """
        # Create new prior node for the section
        prior_node_id = f"S_section_{iteration}"
        graph.add_node(
            prior_node_id,
            type="prior",
            bipartite=0,
            text=section_text[:500],
            section_name=section_name,
            iteration=iteration
        )

        # Extract claims from the section
        if self.claim_extractor is not None:
            section_claims = self._extract_claims_from_text(section_text, prior_node_id)
        else:
            section_claims = []

        # Split section into paragraphs for fine-grained provenance
        section_paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        if not section_paragraphs:
            section_paragraphs = [section_text]

        # Find claims in all_claims that are semantically similar to section claims
        # For now, we use a simple text overlap heuristic
        for claim in all_claims:
            if claim["id"] in prior_entailed:
                continue  # Already grounded

            # Check if claim text overlaps with section text
            claim_text_lower = claim["text"].lower()
            section_text_lower = section_text.lower()

            # Simple heuristic: claim is grounded if significant overlap
            words_in_claim = set(claim_text_lower.split())
            words_in_section = set(section_text_lower.split())
            overlap = len(words_in_claim & words_in_section) / max(len(words_in_claim), 1)

            if overlap > 0.5:  # 50% word overlap threshold
                # Find the paragraph within the section that best matches this claim
                best_para = section_text
                best_para_overlap = 0.0
                for para in section_paragraphs:
                    words_in_para = set(para.lower().split())
                    para_overlap = len(words_in_claim & words_in_para) / max(len(words_in_claim), 1)
                    if para_overlap > best_para_overlap:
                        best_para_overlap = para_overlap
                        best_para = para

                # Record provenance on the claim dict
                claim["source_section"] = section_name
                claim["source_paragraph"] = best_para

                # Add anchor edge and store provenance on the graph node too
                graph.add_edge(
                    prior_node_id,
                    claim["id"],
                    weight=self.w_anchor,
                    cost=1 / self.w_anchor,
                    link_type="anchor"
                )
                graph.nodes[claim["id"]]["source_section"] = section_name
                graph.nodes[claim["id"]]["source_paragraph"] = best_para
                prior_entailed.add(claim["id"])

    def _get_contested_claims(
        self,
        graph: nx.Graph,
        prior_entailed: Set[str],
        all_claims: List[Dict]
    ) -> List[Dict]:
        """
        Get all contested (non-grounded) claims.
        """
        contested = []
        for claim in all_claims:
            if claim["id"] not in prior_entailed:
                contested.append(claim)
        return contested

    def _select_claims_for_verification(
        self,
        contested_claims: List[Dict],
        k: int,
        graph: Optional[nx.Graph] = None,
        prior_entailed: Optional[Set[str]] = None,
        response_node_ids: Optional[List[str]] = None
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Select top-k claims for verification using utility-based ranking.

        Uses the IterativeKnowledgeExpansion module to compute:
        - Centrality scores (betweenness in contested subgraph)
        - Novelty scores (graph distance to priors)
        - Priority scores (weighted combination)
        - Costs (graph-based, since embeddings may not be available)
        - Utilities (priority × (1 - sigma(CC)) - cost)

        Args:
            contested_claims: List of contested claim dicts
            k: Number of claims to select
            graph: NetworkX graph (optional, for utility-based selection)
            prior_entailed: Set of grounded claim IDs (optional)
            response_node_ids: List of response node IDs (optional)

        Returns:
            Tuple of (selected_claims, selection_diagnostics)
        """
        if len(contested_claims) == 0:
            return [], {}

        # If graph info not provided, fall back to simple selection
        if graph is None or prior_entailed is None:
            return contested_claims[:k], {"selection_method": "simple"}

        contested_ids = [c["id"] for c in contested_claims]

        # Get response node IDs from graph if not provided
        if response_node_ids is None:
            response_node_ids = [
                n for n in graph.nodes()
                if graph.nodes[n].get("type") == "response"
            ]

        # Compute closeness centrality for all claims in graph
        claim_nodes = [n for n in graph.nodes() if graph.nodes[n].get("type") == "claim"]
        try:
            closeness_centrality = nx.closeness_centrality(graph)
        except Exception:
            closeness_centrality = {n: 0.5 for n in graph.nodes()}

        # Compute centrality scores (betweenness on contested subgraph)
        centrality_raw = compute_centrality_scores(
            graph, contested_ids, response_node_ids
        )

        # Compute novelty scores (graph distance to priors)
        novelty_raw = compute_novelty_scores(
            graph, contested_ids, list(prior_entailed)
        )

        # Compute priority scores (normalized, weighted combination)
        priority_scores, S_centrality, S_novelty = compute_priority_scores(
            centrality_raw, novelty_raw,
            lambda1=1.0, lambda2=1.0
        )

        # Compute graph-based costs (use inverse novelty as proxy for cost)
        # Claims far from priors are harder to verify
        costs = {}
        for cid in contested_ids:
            novelty = novelty_raw.get(cid)
            if novelty is None or novelty == 0:
                costs[cid] = 1.0  # Default cost for disconnected claims
            else:
                # Higher novelty (distance) = higher cost
                costs[cid] = novelty
        costs = z_normalize_positive(costs)

        # Select claims using utility-based ranking
        selection, select_diagnostics = select_claims_for_verification(
            contested_claim_ids=contested_ids,
            closeness_centrality=closeness_centrality,
            priority_scores=priority_scores,
            costs=costs,
            k=k
        )

        # Map selected IDs back to claim dicts
        selected_ids = set(selection['verify'])
        selected_claims = [c for c in contested_claims if c["id"] in selected_ids]

        # Preserve the utility ordering
        id_to_claim = {c["id"]: c for c in contested_claims}
        selected_claims = [id_to_claim[cid] for cid in selection['verify'] if cid in id_to_claim]

        # Build diagnostics
        diagnostics = {
            "selection_method": "utility_based",
            "num_contested": len(contested_claims),
            "num_selected": len(selected_claims),
            "discarded_ids": selection['discard'],
            "priority_scores": priority_scores,
            "costs": costs,
            "S_centrality": S_centrality,
            "S_novelty": S_novelty,
            **select_diagnostics
        }

        return selected_claims, diagnostics

    def _apply_verification_results(
        self,
        graph: nx.Graph,
        prior_entailed: Set[str],
        verification_results: List[VerificationResult],
        iteration: int
    ):
        """
        Apply verification results to update the graph.
        """
        # Create verification prior node
        prior_node_id = f"S_verified_{iteration}"
        graph.add_node(
            prior_node_id,
            type="prior",
            bipartite=0,
            text=f"Verified claims from iteration {iteration}",
            verification_round=iteration
        )

        for result in verification_results:
            claim_id = result.claim_id

            if result.verdict:
                # Accepted: add anchor edge and mark as prior-entailed
                graph.add_edge(
                    prior_node_id,
                    claim_id,
                    weight=self.w_anchor,
                    cost=1 / self.w_anchor,
                    link_type="anchor"
                )
                prior_entailed.add(claim_id)
            else:
                # Rejected: could remove node or mark as false
                # For now, just don't add to prior_entailed
                pass

    def _compute_iteration_metrics(
        self,
        graph: nx.Graph,
        prior_entailed: Set[str],
        all_claims: List[Dict],
        question_classifications: List[QuestionClassification],
        verification_results: List[VerificationResult]
    ) -> IterationMetrics:
        """
        Compute metrics for a single iteration.
        """
        total_claims = len(all_claims)
        grounded_claims = len(prior_entailed)
        contested_claims = total_claims - grounded_claims

        # Prior Coverage Ratio
        pcr = grounded_claims / total_claims if total_claims > 0 else 0

        # Knowledge Frontier Size
        kf_size = contested_claims

        # Question Answerability Rate
        answerable_count = sum(
            1 for q in question_classifications
            if q.answerable_from_prior
        )
        answerability_rate = answerable_count / len(question_classifications) \
            if question_classifications else 0

        # Verification Utility Ratio
        verified_count = len(verification_results)
        accepted_count = sum(1 for v in verification_results if v.verdict)
        vur = accepted_count / verified_count if verified_count > 0 else 0

        return IterationMetrics(
            prior_coverage_ratio=pcr,
            knowledge_frontier_size=kf_size,
            question_answerability_rate=answerability_rate,
            verification_utility_ratio=vur,
            total_claims=total_claims,
            grounded_claims=grounded_claims,
            contested_claims=contested_claims
        )

    def _compute_final_metrics(
        self,
        graph: nx.Graph,
        prior_entailed: Set[str],
        all_claims: List[Dict],
        questions: List[dict]
    ) -> Dict:
        """
        Compute final metrics after all iterations.
        """
        total_claims = len(all_claims)
        grounded_claims = len(prior_entailed)

        return {
            "final_pcr": grounded_claims / total_claims if total_claims > 0 else 0,
            "final_grounded_claims": grounded_claims,
            "final_contested_claims": total_claims - grounded_claims,
            "total_claims": total_claims,
            "total_questions": len(questions),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges()
        }


# ==============================================================================
# BATCH EXPERIMENT RUNNER
# ==============================================================================

def run_batch_experiment(
    papers: List[dict],
    experiment: QASPERExperiment,
    output_path: str,
    num_iterations: int = 3,
    claims_per_iteration: int = 10,
    response_samples_per_question: int = 3,
    max_papers: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict:
    """
    Run experiment on multiple papers and aggregate results.

    Args:
        papers: List of QASPER paper dicts
        experiment: QASPERExperiment instance
        output_path: Path to save results
        num_iterations: Number of iterations per paper
        claims_per_iteration: Claims to verify per iteration
        response_samples_per_question: Response samples per question
        max_papers: Maximum papers to process (None for all)
        progress_callback: Optional callback for progress updates

    Returns:
        Aggregated results dict
    """
    results = []
    n_papers = min(len(papers), max_papers) if max_papers else len(papers)

    for i, paper in enumerate(papers[:n_papers]):
        if progress_callback:
            progress_callback(f"Paper {i + 1}/{n_papers}: {paper['id']}")
        else:
            print(f"Processing paper {i + 1}/{n_papers}: {paper['id']}")

        result = experiment.run_single_paper_experiment(
            paper=paper,
            num_iterations=num_iterations,
            claims_per_iteration=claims_per_iteration,
            response_samples_per_question=response_samples_per_question,
            progress_callback=progress_callback
        )

        if result.error:
            print(f"  Error: {result.error}")
        else:
            results.append(result)

        # Save intermediate results
        if (i + 1) % 10 == 0:
            _save_intermediate_results(results, output_path)

    # Aggregate metrics
    aggregated = aggregate_results(results)

    # Save final results
    final_output = {
        "experiment_config": {
            "num_papers": n_papers,
            "num_iterations": num_iterations,
            "claims_per_iteration": claims_per_iteration,
            "response_samples_per_question": response_samples_per_question
        },
        "individual_results": [r.to_dict() for r in results],
        "aggregated_metrics": aggregated,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Results saved to: {output_path}")

    return aggregated


def aggregate_results(results: List[PaperExperimentResult]) -> Dict:
    """
    Aggregate metrics across multiple papers.
    """
    if not results:
        return {}

    metrics_by_iteration = {}

    for result in results:
        for iteration_result in result.iterations:
            it = iteration_result.iteration
            if it not in metrics_by_iteration:
                metrics_by_iteration[it] = []
            metrics_by_iteration[it].append(iteration_result.metrics)

    aggregated = {}
    for iteration, metrics_list in sorted(metrics_by_iteration.items()):
        if not metrics_list:
            continue

        aggregated[f"iteration_{iteration}"] = {
            "mean_pcr": np.mean([m.prior_coverage_ratio for m in metrics_list]),
            "std_pcr": np.std([m.prior_coverage_ratio for m in metrics_list]),
            "mean_kf_size": np.mean([m.knowledge_frontier_size for m in metrics_list]),
            "std_kf_size": np.std([m.knowledge_frontier_size for m in metrics_list]),
            "mean_answerability": np.mean([m.question_answerability_rate for m in metrics_list]),
            "mean_vur": np.mean([m.verification_utility_ratio for m in metrics_list]),
            "mean_total_claims": np.mean([m.total_claims for m in metrics_list]),
            "mean_grounded_claims": np.mean([m.grounded_claims for m in metrics_list]),
            "num_papers": len(metrics_list)
        }

    return aggregated


def _save_intermediate_results(results: List[PaperExperimentResult], output_path: str):
    """Save intermediate results."""
    intermediate_path = output_path.replace(".json", "_intermediate.json")
    with open(intermediate_path, "w") as f:
        json.dump(
            {"results": [r.to_dict() for r in results]},
            f, indent=2
        )


if __name__ == "__main__":
    # Quick test
    print("Testing QASPERExperiment module...")

    # Create mock experiment
    experiment = QASPERExperiment(
        generator_model=None,  # Mock
        claim_extractor=None,  # Mock
        verifier_type="mock"
    )

    # Create mock paper
    mock_paper = {
        "id": "test_paper",
        "title": "Test Paper Title",
        "abstract": "This is the abstract of the test paper.",
        "full_text": {
            "section_name": ["Introduction", "Methods", "Results"],
            "paragraphs": [
                ["This is the introduction."],
                ["This is the methods section."],
                ["These are the results."]
            ]
        },
        "qas": [
            {
                "question_id": "q1",
                "question": "What is the main contribution?",
                "answers": [{"unanswerable": False, "free_form_answer": "Test answer"}]
            }
        ]
    }

    result = experiment.run_single_paper_experiment(
        paper=mock_paper,
        num_iterations=3,
        claims_per_iteration=5,
        response_samples_per_question=2
    )

    print(f"Paper: {result.paper_id}")
    print(f"Iterations: {len(result.iterations)}")
    for it in result.iterations:
        print(f"  Iteration {it.iteration}: PCR={it.metrics.prior_coverage_ratio:.3f}")

    print("\nModule test complete!")
