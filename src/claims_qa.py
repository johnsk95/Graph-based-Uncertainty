"""
Claims-Based Question Answering Module

Answers questions using different context conditions:
1. VERIFIED_CLAIMS: Grounded/verified claims from knowledge graph
2. BASELINE_ABSTRACT: Title + Abstract only
3. BASELINE_FULL: Full paper text
"""

import json
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
        conditions: List[str] = None
    ) -> Dict:
        """
        Run QA experiment for a single paper across specified conditions.

        Args:
            paper: QASPER paper dict
            questions: List of generated questions with ground truth
            verified_claims: List of verified claims for this paper
            conditions: List of conditions to run (default: all three)

        Returns:
            Results dict with answers for each condition
        """
        if conditions is None:
            conditions = ["verified_claims", "baseline_abstract", "baseline_full"]

        # Prepare contexts
        contexts = {}
        if "verified_claims" in conditions:
            contexts["verified_claims"] = format_claims_as_context(verified_claims)
        if "baseline_abstract" in conditions:
            contexts["baseline_abstract"] = format_abstract_context(paper)
        if "baseline_full" in conditions:
            contexts["baseline_full"] = format_full_paper_context(paper)

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
                context = contexts[condition]

                try:
                    qa_result = self.answer_question(question_text, context)

                    results["conditions"][condition].append({
                        "question_id": question_id,
                        "question": question_text,
                        "ground_truth": ground_truth,
                        "predicted_answer": qa_result["answer"],
                        "tokens_used": qa_result["tokens_used"]
                    })
                except Exception as e:
                    print(f"    Error answering question {question_id} with {condition}: {e}")
                    results["conditions"][condition].append({
                        "question_id": question_id,
                        "question": question_text,
                        "ground_truth": ground_truth,
                        "predicted_answer": f"Error: {str(e)}",
                        "tokens_used": 0,
                        "error": str(e)
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
