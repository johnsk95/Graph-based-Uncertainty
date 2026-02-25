"""
Claims-Based Question Answering Module

Answers questions using different context conditions:
1. VERIFIED_CLAIMS: Grounded/verified claims from knowledge graph
2. BASELINE_ABSTRACT: Title + Abstract only
3. BASELINE_FULL: Full paper text
4. GRAPH_GUIDED_RETRIEVAL: Source passages retrieved via graph centrality + question relevance
"""

import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path


# ==============================================================================
# PROMPTS
# ==============================================================================

QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about scientific papers.
Answer questions using ONLY the information provided in the context.
If the context does not contain enough information to answer the question,
say "Cannot be determined from the given information."

Be concise and specific in your answers.
"""

QA_USER_PROMPT_TEMPLATE = """# Context
{context}

# Question
{question}

# Instructions
Answer the question using only the information provided in the context above.
Provide a concise, specific answer.
"""


# ==============================================================================
# CLAIMS LOADING
# ==============================================================================

def load_verified_claims_from_results(
    results_path: str,
    paper_id: str
) -> List[Dict]:
    """
    Load grounded/verified claims for a paper from experiment results.

    Args:
        results_path: Path to results.json from knowledge expansion experiment
        paper_id: Paper ID to load claims for

    Returns:
        List of {"claim_id": str, "text": str, "status": str}
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Find the paper in results
    paper_result = None
    for p in results.get("individual_results", []):
        if p["paper_id"] == paper_id:
            paper_result = p
            break

    if paper_result is None:
        print(f"Warning: Paper {paper_id} not found in results")
        return []

    # Get the final graph's grounded claims
    final_graph = paper_result.get("final_graph", {})
    grounded_claim_ids = set(final_graph.get("prior_entailed_claims", []))

    # Also get verification results to identify accepted claims
    accepted_claim_ids = set()
    for iteration in paper_result.get("iterations", []):
        for ver_result in iteration.get("verification_results", []):
            if ver_result.get("verdict", False):
                accepted_claim_ids.add(ver_result["claim_id"])

    # All verified claims = grounded OR accepted
    verified_claim_ids = grounded_claim_ids | accepted_claim_ids

    # Get claim texts from the paper's claims list
    claims = paper_result.get("claims", [])
    verified_claims = []

    for claim in claims:
        claim_id = claim.get("id")
        if claim_id in verified_claim_ids:
            status = "grounded" if claim_id in grounded_claim_ids else "accepted"
            verified_claims.append({
                "claim_id": claim_id,
                "text": claim.get("text", ""),
                "status": status,
                "source_type": claim.get("source_type", "unknown")
            })

    return verified_claims


def get_paper_info_from_results(
    results_path: str,
    paper_id: str
) -> Optional[Dict]:
    """
    Get paper info from results file.

    Args:
        results_path: Path to results.json
        paper_id: Paper ID

    Returns:
        Paper result dict or None
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    for p in results.get("individual_results", []):
        if p["paper_id"] == paper_id:
            return p

    return None


# ==============================================================================
# GRAPH-GUIDED PASSAGE RETRIEVAL
# ==============================================================================

def _build_provenance_from_graph(paper_result: Dict) -> Dict[str, Tuple[str, str]]:
    """
    Reconstruct (source_section, source_paragraph) for each claim from the
    final graph snapshot.

    Uses anchor edges: claim → prior source node → node text (section text).
    Falls back to the full source node text when per-paragraph data is absent
    (i.e. results generated before the provenance change).

    Returns:
        Dict mapping claim_id -> (section_name, paragraph_text)
    """
    final_graph = paper_result.get("final_graph", {})
    nodes = final_graph.get("nodes", [])
    edges = final_graph.get("edges", [])

    # Build node lookup: id -> node dict
    node_by_id: Dict[str, Dict] = {n["id"]: n for n in nodes}

    # Map each claim to its best prior source node via anchor edges
    # (prefer section nodes S_section_N over generic S_prior)
    claim_to_prior: Dict[str, str] = {}
    for edge in edges:
        if edge.get("link_type") != "anchor":
            continue
        src, tgt = edge.get("source", ""), edge.get("target", "")
        # anchor edges run prior → claim
        if src in node_by_id and tgt.startswith("C"):
            # Prefer a named section node over a generic one
            existing = claim_to_prior.get(tgt, "")
            if not existing or src.startswith("S_section"):
                claim_to_prior[tgt] = src

    provenance: Dict[str, Tuple[str, str]] = {}
    for claim_id, prior_id in claim_to_prior.items():
        prior_node = node_by_id.get(prior_id, {})
        section_name = prior_node.get("section_name", "")
        text = prior_node.get("text", "")
        # source_paragraph may be stored directly on the node (new results)
        para = prior_node.get("source_paragraph", text)
        provenance[claim_id] = (section_name, para)

    return provenance


def load_verified_claims_with_provenance(
    results_path: str,
    paper_id: str
) -> List[Dict]:
    """
    Load verified claims including their source_paragraph and source_section
    fields needed for passage retrieval.

    Works with both new results (provenance stored on claim dicts) and old
    results (provenance reconstructed from the graph snapshot).

    Returns:
        List of {"claim_id", "text", "status", "source_type",
                 "source_section", "source_paragraph"}
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    paper_result = None
    for p in results.get("individual_results", []):
        if p["paper_id"] == paper_id:
            paper_result = p
            break

    if paper_result is None:
        return []

    final_graph = paper_result.get("final_graph", {})
    grounded_claim_ids = set(final_graph.get("prior_entailed_claims", []))

    accepted_claim_ids = set()
    for iteration in paper_result.get("iterations", []):
        for ver_result in iteration.get("verification_results", []):
            if ver_result.get("verdict", False):
                accepted_claim_ids.add(ver_result["claim_id"])

    verified_claim_ids = grounded_claim_ids | accepted_claim_ids

    # Reconstruct provenance from the graph snapshot as a fallback
    graph_provenance = _build_provenance_from_graph(paper_result)

    claims = paper_result.get("claims", [])
    verified_claims = []
    for claim in claims:
        claim_id = claim.get("id")
        if claim_id not in verified_claim_ids:
            continue

        status = "grounded" if claim_id in grounded_claim_ids else "accepted"

        # Prefer provenance stored directly on the claim (new results),
        # fall back to graph-reconstructed provenance
        source_section = claim.get("source_section", "")
        source_paragraph = claim.get("source_paragraph", "")
        if not source_paragraph and claim_id in graph_provenance:
            source_section, source_paragraph = graph_provenance[claim_id]

        verified_claims.append({
            "claim_id": claim_id,
            "text": claim.get("text", ""),
            "status": status,
            "source_type": claim.get("source_type", "unknown"),
            "source_section": source_section,
            "source_paragraph": source_paragraph,
        })

    return verified_claims


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def retrieve_passages_for_question(
    question: str,
    verified_claims: List[Dict],
    centrality_scores: Optional[Dict[str, float]] = None,
    top_k: int = 5,
    min_paragraph_length: int = 50,
    embedding_model: Optional[Any] = None,
) -> List[Dict]:
    """
    Retrieve the most relevant source passages for a question by combining
    graph centrality (how reliable a claim is) with question-passage relevance.

    For each verified claim that has a non-empty source_paragraph, a score is
    computed as:

        score = relevance_weight * question_relevance
              + centrality_weight * centrality_score

    Passages are deduplicated; when multiple claims point to the same paragraph
    the highest composite score is kept.

    Args:
        question: The question text.
        verified_claims: List of claim dicts with "source_paragraph", etc.
        centrality_scores: Optional dict mapping claim_id -> centrality value
            (e.g. weighted closeness).  If None, all claims are treated equally.
        top_k: Maximum number of passages to return.
        min_paragraph_length: Discard paragraphs shorter than this (chars).
        embedding_model: An EmbeddingModel instance from src/embeddings.py.
            If None, only centrality is used for ranking.

    Returns:
        List of up to top_k dicts:
            {"passage": str, "section": str, "score": float,
             "contributing_claims": [claim_id, ...]}
        ordered by descending score.
    """
    # Collect candidate (paragraph, section, claim) triples
    candidates: List[Dict] = []
    for claim in verified_claims:
        para = claim.get("source_paragraph", "").strip()
        if not para or len(para) < min_paragraph_length:
            continue
        candidates.append({
            "claim_id": claim["claim_id"],
            "paragraph": para,
            "section": claim.get("source_section", ""),
            "centrality": (
                centrality_scores.get(claim["claim_id"], 0.5)
                if centrality_scores else 0.5
            ),
        })

    if not candidates:
        return []

    # Normalise centrality scores to [0, 1] across this candidate set
    raw_centralities = np.array([c["centrality"] for c in candidates])
    c_min, c_max = raw_centralities.min(), raw_centralities.max()
    if c_max > c_min:
        norm_centralities = (raw_centralities - c_min) / (c_max - c_min)
    else:
        norm_centralities = np.ones(len(candidates)) * 0.5

    # Compute question-relevance scores via embedding similarity (if available)
    if embedding_model is not None:
        try:
            paragraphs = [c["paragraph"] for c in candidates]
            texts_to_embed = [question] + paragraphs
            embeddings = embedding_model.encode(texts_to_embed)
            q_emb = embeddings[0]
            p_embs = embeddings[1:]
            relevance_scores = np.array([
                _cosine_similarity(q_emb, p_embs[i])
                for i in range(len(paragraphs))
            ])
            # Normalise to [0, 1]
            r_min, r_max = relevance_scores.min(), relevance_scores.max()
            if r_max > r_min:
                norm_relevance = (relevance_scores - r_min) / (r_max - r_min)
            else:
                norm_relevance = np.ones(len(candidates)) * 0.5
            relevance_weight, centrality_weight = 0.6, 0.4
        except Exception:
            norm_relevance = np.ones(len(candidates)) * 0.5
            relevance_weight, centrality_weight = 0.0, 1.0
    else:
        norm_relevance = np.ones(len(candidates)) * 0.5
        relevance_weight, centrality_weight = 0.0, 1.0

    # Composite score per candidate
    composite = (
        relevance_weight * norm_relevance
        + centrality_weight * norm_centralities
    )

    # Deduplicate: keep one entry per unique paragraph, merging contributing claims
    para_map: Dict[str, Dict] = {}
    for i, cand in enumerate(candidates):
        key = cand["paragraph"]
        if key not in para_map or composite[i] > para_map[key]["score"]:
            para_map[key] = {
                "passage": cand["paragraph"],
                "section": cand["section"],
                "score": float(composite[i]),
                "contributing_claims": [cand["claim_id"]],
            }
        else:
            para_map[key]["contributing_claims"].append(cand["claim_id"])

    ranked = sorted(para_map.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def format_retrieved_passages_as_context(
    passages: List[Dict],
    paper_title: str = ""
) -> str:
    """
    Format retrieved passages into a context string for the QA model.

    Passages are ordered by score (highest first) and labelled with their
    source section so the model has structural cues.
    """
    if not passages:
        return "No relevant passages could be retrieved for this question."

    lines = []
    if paper_title:
        lines += [f"# {paper_title}", ""]

    for i, p in enumerate(passages, 1):
        section_label = f"[{p['section']}]" if p.get("section") else f"[Passage {i}]"
        lines += [section_label, p["passage"], ""]

    return "\n".join(lines).strip()


# ==============================================================================
# CONTEXT FORMATTING
# ==============================================================================

def format_claims_as_context(claims: List[Dict]) -> str:
    """
    Format verified claims as context for the QA model.

    Args:
        claims: List of {"claim_id": str, "text": str, "status": str}

    Returns:
        Formatted context string
    """
    if not claims:
        return "No verified information available about this paper."

    context_lines = [
        "The following are verified facts about this paper:",
        ""
    ]

    for i, claim in enumerate(claims, 1):
        claim_text = claim.get("text", "").strip()
        if claim_text:
            context_lines.append(f"{i}. {claim_text}")

    return "\n".join(context_lines)


def format_abstract_context(paper: Dict) -> str:
    """
    Format title + abstract as context.

    Args:
        paper: Paper dict with 'title' and 'abstract'

    Returns:
        Formatted context string
    """
    title = paper.get("title", "Unknown Title")
    abstract = paper.get("abstract", "No abstract available.")

    return f"# {title}\n\n## Abstract\n{abstract}"


def format_full_paper_context(paper: Dict, max_length: int = 100000) -> str:
    """
    Format full paper as context.

    Args:
        paper: Paper dict with 'title', 'abstract', 'full_text'
        max_length: Maximum context length (truncate if exceeded)

    Returns:
        Formatted context string
    """
    lines = [
        f"# {paper.get('title', 'Unknown Title')}",
        "",
        "## Abstract",
        paper.get("abstract", ""),
        ""
    ]

    full_text = paper.get("full_text", {})
    section_names = full_text.get("section_name", [])
    paragraphs = full_text.get("paragraphs", [])

    for i, section_name in enumerate(section_names):
        lines.append(f"## {section_name}")
        if i < len(paragraphs):
            for para in paragraphs[i]:
                if para and para.strip():
                    lines.append(para)
                    lines.append("")

    context = "\n".join(lines)

    # Truncate if too long
    if len(context) > max_length:
        context = context[:max_length] + "\n\n[Content truncated due to length...]"

    return context


# ==============================================================================
# QA CLASS
# ==============================================================================

class ClaimsBasedQA:
    """
    Answer questions using different context conditions.
    """

    def __init__(self, openai_client, model: str = "gpt-4o-mini"):
        """
        Initialize the QA system.

        Args:
            openai_client: OpenAI client instance
            model: Model to use for answering
        """
        self.client = openai_client
        self.model = model

    def answer_question(
        self,
        question: str,
        context: str
    ) -> Dict:
        """
        Answer a single question given context.

        Args:
            question: Question text
            context: Context string (claims, abstract, or full paper)

        Returns:
            {"answer": str, "tokens_used": int}
        """
        user_prompt = QA_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Deterministic for reproducibility
            max_tokens=500
        )

        return {
            "answer": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens
        }

    def run_qa_for_paper(
        self,
        paper: Dict,
        questions: List[Dict],
        verified_claims: List[Dict],
        conditions: List[str] = None,
        centrality_scores: Optional[Dict[str, float]] = None,
        embedding_model: Optional[Any] = None,
        retrieval_top_k: int = 5,
    ) -> Dict:
        """
        Run QA experiment for a single paper across specified conditions.

        Args:
            paper: QASPER paper dict
            questions: List of generated questions with ground truth
            verified_claims: List of verified claims for this paper.
                For "graph_guided_retrieval" these must include
                "source_paragraph" and "source_section" fields
                (use load_verified_claims_with_provenance).
            conditions: List of conditions to run (default: all four).
                Options: "verified_claims", "baseline_abstract",
                         "baseline_full", "graph_guided_retrieval"
            centrality_scores: Dict mapping claim_id -> centrality value,
                used to weight passage retrieval. If None, all claims are
                weighted equally.
            embedding_model: EmbeddingModel for question-passage similarity
                reranking. If None, ranking is centrality-only.
            retrieval_top_k: Number of passages to retrieve per question.

        Returns:
            Results dict with answers for each condition
        """
        if conditions is None:
            conditions = [
                "verified_claims",
                "baseline_abstract",
                "baseline_full",
                "graph_guided_retrieval",
            ]

        paper_title = paper.get("title", "")

        # Prepare static contexts (same for every question)
        contexts: Dict[str, Any] = {}
        if "verified_claims" in conditions:
            contexts["verified_claims"] = format_claims_as_context(verified_claims)
        if "baseline_abstract" in conditions:
            contexts["baseline_abstract"] = format_abstract_context(paper)
        if "baseline_full" in conditions:
            contexts["baseline_full"] = format_full_paper_context(paper)
        # "graph_guided_retrieval" context is computed per-question below

        results = {
            "paper_id": paper.get("id", "unknown"),
            "num_verified_claims": len(verified_claims),
            "num_questions": len(questions),
            "context_lengths": {
                cond: len(ctx) for cond, ctx in contexts.items()
            },
            "conditions": {cond: [] for cond in conditions}
        }

        for q in questions:
            question_text = q.get("question", "")
            ground_truth = q.get("answer", "")
            question_id = q.get("question_id", "")

            for condition in conditions:
                if condition == "graph_guided_retrieval":
                    # Retrieve passages specific to this question
                    passages = retrieve_passages_for_question(
                        question=question_text,
                        verified_claims=verified_claims,
                        centrality_scores=centrality_scores,
                        top_k=retrieval_top_k,
                        embedding_model=embedding_model,
                    )
                    context = format_retrieved_passages_as_context(passages, paper_title)
                    extra = {"num_passages_retrieved": len(passages)}
                else:
                    context = contexts[condition]
                    extra = {}

                try:
                    qa_result = self.answer_question(question_text, context)

                    results["conditions"][condition].append({
                        "question_id": question_id,
                        "question": question_text,
                        "ground_truth": ground_truth,
                        "predicted_answer": qa_result["answer"],
                        "tokens_used": qa_result["tokens_used"],
                        **extra,
                    })
                except Exception as e:
                    print(f"    Error answering question {question_id} with {condition}: {e}")
                    results["conditions"][condition].append({
                        "question_id": question_id,
                        "question": question_text,
                        "ground_truth": ground_truth,
                        "predicted_answer": f"Error: {str(e)}",
                        "tokens_used": 0,
                        "error": str(e),
                        **extra,
                    })

        return results


# ==============================================================================
# ANSWERABILITY DETECTION
# ==============================================================================

UNANSWERABLE_PATTERNS = [
    "cannot be determined",
    "not enough information",
    "cannot answer",
    "no information",
    "not mentioned",
    "not provided",
    "not specified",
    "unclear from",
    "not stated",
    "information is not",
    "does not provide",
    "does not contain",
    "does not mention",
    "cannot find",
    "no evidence"
]


def is_unanswerable_response(answer: str) -> bool:
    """
    Check if the model indicated it couldn't answer.

    Args:
        answer: Model's answer text

    Returns:
        True if the answer indicates inability to answer
    """
    answer_lower = answer.lower()
    return any(pattern in answer_lower for pattern in UNANSWERABLE_PATTERNS)


# ==============================================================================
# CLI FOR TESTING
# ==============================================================================

if __name__ == "__main__":
    import os

    # Test loading claims
    results_path = "experiments/qasper/experiment_gpt4o/results.json"
    paper_id = "1910.12203"

    claims = load_verified_claims_from_results(results_path, paper_id)
    print(f"Loaded {len(claims)} verified claims for {paper_id}")

    if claims:
        print("\nSample claims:")
        for c in claims[:5]:
            print(f"  [{c['status']}] {c['text'][:80]}...")

        print("\nFormatted context:")
        context = format_claims_as_context(claims)
        print(context[:500] + "...")
