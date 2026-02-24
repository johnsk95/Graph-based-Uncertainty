"""
QASPER Dataset Loader and Utilities

Provides functions for loading the QASPER dataset, identifying sections,
constructing priors for progressive expansion, and classifying questions.

Dataset: Allen AI QASPER
- 5,049 questions over 1,585 NLP papers
- Hugging Face: allenai/qasper
"""

from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datasets import load_dataset


# ==============================================================================
# SECTION NAME VARIANTS
# ==============================================================================

INTRODUCTION_VARIANTS = [
    "introduction",
    "intro",
    "1 introduction",
    "i. introduction",
    "background",
    "1. introduction",
    "overview",
    "motivation",
    "1 intro",
]

METHODS_VARIANTS = [
    "methods",
    "method",
    "methodology",
    "approach",
    "model",
    "proposed method",
    "proposed approach",
    "our approach",
    "our method",
    "system",
    "framework",
    "architecture",
    "technical approach",
    "2 method",
    "3 method",
    "2. method",
    "3. methodology",
    "2 methods",
    "3 methods",
    "the model",
    "our model",
    "system description",
    "system overview",
]

EXPERIMENT_VARIANTS = [
    "experiments",
    "experiment",
    "experimental setup",
    "experimental results",
    "evaluation",
    "results",
    "empirical evaluation",
    "empirical results",
    "4 experiments",
    "5 experiments",
    "4. experiments",
    "5. evaluation",
    "results and discussion",
    "experimental evaluation",
]


# ==============================================================================
# QASPER FORMAT CONVERSION
# ==============================================================================

def normalize_qasper_questions(paper: dict) -> List[dict]:
    """
    Convert QASPER's dict-of-lists format to a list of question dicts.

    QASPER format:
        paper["qas"] = {
            "question": ["q1", "q2", ...],
            "question_id": ["id1", "id2", ...],
            "answers": [{"answer": [...]}, {"answer": [...]}, ...]
        }

    Normalized format:
        [
            {"question_id": "id1", "question": "q1", "answers": [...]},
            {"question_id": "id2", "question": "q2", "answers": [...]},
            ...
        ]

    Args:
        paper: QASPER paper dict

    Returns:
        List of normalized question dicts
    """
    qas = paper.get("qas", {})

    # Handle already normalized format (for testing)
    if isinstance(qas, list):
        return qas

    # QASPER format: dict with parallel lists
    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [])
    answers_list = qas.get("answers", [])

    normalized = []
    for i in range(len(questions)):
        q_id = question_ids[i] if i < len(question_ids) else f"q{i}"
        q_text = questions[i] if i < len(questions) else ""

        # Get answers for this question
        # Each item in answers_list is a dict with "answer" key containing list of annotator answers
        if i < len(answers_list):
            answer_dict = answers_list[i]
            if isinstance(answer_dict, dict) and "answer" in answer_dict:
                # answer_dict["answer"] is a list of annotator answers
                annotator_answers = answer_dict["answer"]
            else:
                annotator_answers = []
        else:
            annotator_answers = []

        normalized.append({
            "question_id": q_id,
            "question": q_text,
            "answers": annotator_answers
        })

    return normalized


def get_num_questions(paper: dict) -> int:
    """Get the number of questions in a paper."""
    qas = paper.get("qas", {})
    if isinstance(qas, list):
        return len(qas)
    return len(qas.get("question", []))


def count_answerable_questions(paper: dict) -> int:
    """Count the number of answerable questions in a paper."""
    questions = normalize_qasper_questions(paper)
    count = 0
    for q in questions:
        answers = q.get("answers", [])
        if answers and len(answers) > 0:
            first_answer = answers[0]
            if isinstance(first_answer, dict) and not first_answer.get("unanswerable", False):
                count += 1
    return count


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class PriorInfo:
    """Information about the constructed prior for an iteration."""
    text: str
    sections_included: List[str]
    token_count: int
    iteration: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "sections_included": self.sections_included,
            "token_count": self.token_count,
            "iteration": self.iteration
        }


@dataclass
class QuestionClassification:
    """Classification of a question's answerability."""
    question_id: str
    question: str
    is_unanswerable: bool
    evidence_sections: List[str]
    answerable_from_prior: bool
    answer_type: str  # "extractive", "yes_no", "free_form", "unanswerable"
    ground_truth_answer: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "is_unanswerable": self.is_unanswerable,
            "evidence_sections": self.evidence_sections,
            "answerable_from_prior": self.answerable_from_prior,
            "answer_type": self.answer_type,
            "ground_truth_answer": self.ground_truth_answer
        }


@dataclass
class PaperStructure:
    """Validated structure information for a paper."""
    paper_id: str
    title: str
    num_sections: int
    section_names: List[str]
    has_intro: bool
    has_methods: bool
    has_experiments: bool
    intro_idx: Optional[int]
    methods_idx: Optional[int]
    experiment_idx: Optional[int]
    num_questions: int
    answerable_questions: int


# ==============================================================================
# SECTION IDENTIFICATION
# ==============================================================================

def identify_section_index(
    section_names: List[str],
    variants: List[str]
) -> Optional[int]:
    """
    Find the index of a section matching any of the variant names.

    Args:
        section_names: List of section names from paper
        variants: List of acceptable variant names (lowercase)

    Returns:
        Index of matching section, or None if not found
    """
    for idx, name in enumerate(section_names):
        if name is None:
            continue
        normalized = name.lower().strip()
        for variant in variants:
            if variant in normalized or normalized in variant:
                return idx
    return None


def get_sections_before_experiments(section_names: List[str]) -> List[int]:
    """
    Get indices of all sections that appear before experiments/results.
    Used as fallback when Methods section is not explicitly named.

    Args:
        section_names: List of section names from paper

    Returns:
        List of section indices to include as P_2
    """
    experiment_idx = identify_section_index(section_names, EXPERIMENT_VARIANTS)

    if experiment_idx is None:
        # No experiment section found - include first half of sections
        return list(range(len(section_names) // 2))

    # Return all sections before experiment section
    return list(range(experiment_idx))


# ==============================================================================
# PRIOR CONSTRUCTION
# ==============================================================================

def construct_prior(paper: dict, iteration: int) -> PriorInfo:
    """
    Construct the prior text for a given iteration.

    Args:
        paper: QASPER paper dict
        iteration: 0, 1, or 2

    Returns:
        PriorInfo with text, sections_included, token_count, iteration
    """
    title = paper["title"]
    abstract = paper["abstract"]
    section_names = paper["full_text"]["section_name"]
    paragraphs = paper["full_text"]["paragraphs"]

    # P_0: Title + Abstract (always included)
    prior_text = f"Title: {title}\n\nAbstract: {abstract}"
    sections_included = ["title", "abstract"]

    if iteration >= 1:
        # P_1: Add Introduction
        intro_idx = identify_section_index(section_names, INTRODUCTION_VARIANTS)
        if intro_idx is not None and intro_idx < len(paragraphs):
            intro_text = "\n\n".join(paragraphs[intro_idx])
            section_name = section_names[intro_idx]
            prior_text += f"\n\n{section_name}:\n{intro_text}"
            sections_included.append(section_name)

    if iteration >= 2:
        # P_2: Add Methods (or pre-experiment sections)
        methods_idx = identify_section_index(section_names, METHODS_VARIANTS)

        if methods_idx is not None and methods_idx < len(paragraphs):
            # Explicit Methods section found
            methods_text = "\n\n".join(paragraphs[methods_idx])
            section_name = section_names[methods_idx]
            prior_text += f"\n\n{section_name}:\n{methods_text}"
            sections_included.append(section_name)
        else:
            # Fallback: include all sections before experiments
            pre_exp_indices = get_sections_before_experiments(section_names)

            # Filter out already-included sections (intro)
            intro_idx = identify_section_index(section_names, INTRODUCTION_VARIANTS)
            for idx in pre_exp_indices:
                if idx != intro_idx and idx < len(paragraphs):
                    section_text = "\n\n".join(paragraphs[idx])
                    section_name = section_names[idx]
                    prior_text += f"\n\n{section_name}:\n{section_text}"
                    sections_included.append(section_name)

    # Estimate token count (rough: 1 token ~ 4 chars)
    token_count = len(prior_text) // 4

    return PriorInfo(
        text=prior_text,
        sections_included=sections_included,
        token_count=token_count,
        iteration=iteration
    )


def construct_expert_context(paper: dict) -> str:
    """
    Construct full paper text for expert verification.

    Args:
        paper: QASPER paper dict

    Returns:
        Full paper text as string
    """
    title = paper["title"]
    abstract = paper["abstract"]
    section_names = paper["full_text"]["section_name"]
    paragraphs = paper["full_text"]["paragraphs"]

    full_text = f"Title: {title}\n\nAbstract: {abstract}"

    for idx, section_name in enumerate(section_names):
        if idx < len(paragraphs):
            section_text = "\n\n".join(paragraphs[idx])
            full_text += f"\n\n{section_name}:\n{section_text}"

    return full_text


def get_section_text(paper: dict, section_name: str) -> Optional[str]:
    """
    Get the text of a specific section from a paper.

    Args:
        paper: QASPER paper dict
        section_name: Name of section to retrieve

    Returns:
        Section text or None if not found
    """
    section_names = paper["full_text"]["section_name"]
    paragraphs = paper["full_text"]["paragraphs"]

    for idx, name in enumerate(section_names):
        if name.lower().strip() == section_name.lower().strip():
            if idx < len(paragraphs):
                return "\n\n".join(paragraphs[idx])
    return None


def get_new_section_for_iteration(paper: dict, iteration: int) -> Optional[Tuple[str, str]]:
    """
    Get the newly added section for a given iteration.

    Args:
        paper: QASPER paper dict
        iteration: 1 or 2 (iteration 0 has no new section)

    Returns:
        Tuple of (section_name, section_text) or None
    """
    if iteration == 0:
        return None

    section_names = paper["full_text"]["section_name"]
    paragraphs = paper["full_text"]["paragraphs"]

    if iteration == 1:
        # Introduction
        intro_idx = identify_section_index(section_names, INTRODUCTION_VARIANTS)
        if intro_idx is not None and intro_idx < len(paragraphs):
            return (section_names[intro_idx], "\n\n".join(paragraphs[intro_idx]))

    elif iteration == 2:
        # Methods
        methods_idx = identify_section_index(section_names, METHODS_VARIANTS)
        if methods_idx is not None and methods_idx < len(paragraphs):
            return (section_names[methods_idx], "\n\n".join(paragraphs[methods_idx]))

    return None


# ==============================================================================
# QUESTION CLASSIFICATION
# ==============================================================================

def find_evidence_sections(
    evidence_paragraphs: List[str],
    full_text: dict
) -> List[str]:
    """
    Map evidence paragraphs back to their source sections.

    Args:
        evidence_paragraphs: List of evidence paragraph texts
        full_text: Paper full_text dict with section_name and paragraphs

    Returns:
        List of section names containing the evidence
    """
    evidence_sections = set()
    section_names = full_text["section_name"]
    paragraphs = full_text["paragraphs"]

    for evidence in evidence_paragraphs:
        if not evidence:
            continue
        evidence_lower = evidence.lower().strip()

        for idx, section_paras in enumerate(paragraphs):
            for para in section_paras:
                # Check if evidence matches or is contained in paragraph
                para_lower = para.lower().strip()
                if evidence_lower in para_lower or para_lower in evidence_lower:
                    if idx < len(section_names):
                        evidence_sections.add(section_names[idx])
                    break

    return list(evidence_sections)


def classify_question_answerability(
    question: dict,
    prior_sections: List[str],
    paper: dict
) -> QuestionClassification:
    """
    Classify whether a question is answerable from the current prior.

    Args:
        question: QASPER question dict with answers
        prior_sections: List of section names included in current prior
        paper: Full paper dict

    Returns:
        QuestionClassification dataclass
    """
    # Get first annotator's answer (if available)
    if not question.get("answers") or len(question["answers"]) == 0:
        return QuestionClassification(
            question_id=question.get("question_id", ""),
            question=question.get("question", ""),
            is_unanswerable=True,
            evidence_sections=[],
            answerable_from_prior=False,
            answer_type="unanswerable",
            ground_truth_answer=None
        )

    answer = question["answers"][0]

    # Check if unanswerable
    if answer.get("unanswerable", False):
        return QuestionClassification(
            question_id=question["question_id"],
            question=question["question"],
            is_unanswerable=True,
            evidence_sections=[],
            answerable_from_prior=False,
            answer_type="unanswerable",
            ground_truth_answer=None
        )

    # Determine answer type and ground truth
    if answer.get("extractive_spans"):
        answer_type = "extractive"
        ground_truth = " ".join(answer["extractive_spans"])
    elif answer.get("yes_no") is not None:
        answer_type = "yes_no"
        ground_truth = "Yes" if answer["yes_no"] else "No"
    else:
        answer_type = "free_form"
        ground_truth = answer.get("free_form_answer", "")

    # Find which sections contain the evidence
    evidence_paragraphs = answer.get("evidence", [])
    evidence_sections = find_evidence_sections(
        evidence_paragraphs,
        paper["full_text"]
    )

    # Check if evidence is in prior
    prior_sections_lower = [s.lower() for s in prior_sections]
    if evidence_sections:
        answerable = all(
            any(p in es.lower() or es.lower() in p
                for p in prior_sections_lower)
            for es in evidence_sections
        )
    else:
        # Evidence might be in abstract
        answerable = "abstract" in prior_sections_lower

    return QuestionClassification(
        question_id=question["question_id"],
        question=question["question"],
        is_unanswerable=False,
        evidence_sections=evidence_sections,
        answerable_from_prior=answerable,
        answer_type=answer_type,
        ground_truth_answer=ground_truth
    )


# ==============================================================================
# DATASET LOADING
# ==============================================================================

def load_qasper_dataset(split: str = "validation") -> List[dict]:
    """
    Load QASPER dataset from Hugging Face.

    Args:
        split: "train", "validation", or "test"

    Returns:
        List of paper dicts
    """
    dataset = load_dataset("allenai/qasper", split=split)
    return list(dataset)


def validate_paper_structure(paper: dict) -> PaperStructure:
    """
    Validate and report paper structure for debugging.

    Args:
        paper: QASPER paper dict

    Returns:
        PaperStructure dataclass with validation results
    """
    section_names = paper["full_text"]["section_name"]

    intro_idx = identify_section_index(section_names, INTRODUCTION_VARIANTS)
    methods_idx = identify_section_index(section_names, METHODS_VARIANTS)
    experiment_idx = identify_section_index(section_names, EXPERIMENT_VARIANTS)

    num_questions = get_num_questions(paper)
    answerable = count_answerable_questions(paper)

    return PaperStructure(
        paper_id=paper["id"],
        title=paper["title"],
        num_sections=len(section_names),
        section_names=section_names,
        has_intro=intro_idx is not None,
        has_methods=methods_idx is not None,
        has_experiments=experiment_idx is not None,
        intro_idx=intro_idx,
        methods_idx=methods_idx,
        experiment_idx=experiment_idx,
        num_questions=num_questions,
        answerable_questions=answerable
    )


def filter_papers_with_complete_structure(
    papers: List[dict],
    require_methods: bool = False
) -> List[dict]:
    """
    Filter papers that have identifiable Introduction and Methods sections.

    Args:
        papers: List of QASPER paper dicts
        require_methods: If True, require explicit Methods section

    Returns:
        Filtered list with papers having clear structure
    """
    filtered = []

    for paper in papers:
        section_names = paper["full_text"]["section_name"]

        # Check for Introduction
        has_intro = identify_section_index(section_names, INTRODUCTION_VARIANTS) is not None

        # Check for Methods OR can identify pre-experiment sections
        has_methods = identify_section_index(section_names, METHODS_VARIANTS) is not None
        has_experiments = identify_section_index(section_names, EXPERIMENT_VARIANTS) is not None

        if has_intro:
            if require_methods:
                if has_methods:
                    filtered.append(paper)
            else:
                if has_methods or has_experiments:
                    filtered.append(paper)

    return filtered


def get_papers_with_answerable_questions(
    papers: List[dict],
    min_answerable: int = 1
) -> List[dict]:
    """
    Filter papers that have at least min_answerable answerable questions.

    Args:
        papers: List of QASPER paper dicts
        min_answerable: Minimum number of answerable questions required

    Returns:
        Filtered list of papers
    """
    filtered = []

    for paper in papers:
        answerable = count_answerable_questions(paper)
        if answerable >= min_answerable:
            filtered.append(paper)

    return filtered


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_paper_summary(paper: dict) -> dict:
    """
    Get a summary of paper for logging/display.

    Args:
        paper: QASPER paper dict

    Returns:
        Summary dict
    """
    structure = validate_paper_structure(paper)
    return {
        "paper_id": paper["id"],
        "title": paper["title"][:80] + "..." if len(paper["title"]) > 80 else paper["title"],
        "num_sections": structure.num_sections,
        "has_intro": structure.has_intro,
        "has_methods": structure.has_methods,
        "num_questions": structure.num_questions,
        "answerable_questions": structure.answerable_questions
    }


def sample_papers(
    papers: List[dict],
    n_samples: int,
    seed: int = 42
) -> List[dict]:
    """
    Sample n papers from the dataset.

    Args:
        papers: List of QASPER paper dicts
        n_samples: Number of papers to sample
        seed: Random seed for reproducibility

    Returns:
        Sampled list of papers
    """
    import random
    random.seed(seed)

    if n_samples >= len(papers):
        return papers

    return random.sample(papers, n_samples)


if __name__ == "__main__":
    # Test the module
    print("Loading QASPER validation set...")
    papers = load_qasper_dataset("validation")
    print(f"Loaded {len(papers)} papers")

    # Filter for complete structure
    complete = filter_papers_with_complete_structure(papers)
    print(f"Papers with complete structure: {len(complete)}")

    # Test prior construction
    if complete:
        paper = complete[0]
        print(f"\nTest paper: {paper['title'][:60]}...")

        for iteration in range(3):
            prior = construct_prior(paper, iteration)
            print(f"\nIteration {iteration}:")
            print(f"  Sections: {prior.sections_included}")
            print(f"  Token count: {prior.token_count}")

        # Test question classification
        print(f"\nQuestions: {len(paper['qas'])}")
        for q in paper["qas"][:3]:
            prior_0 = construct_prior(paper, 0)
            classification = classify_question_answerability(
                q, prior_0.sections_included, paper
            )
            print(f"  Q: {q['question'][:50]}...")
            print(f"    Type: {classification.answer_type}, Answerable: {classification.answerable_from_prior}")
