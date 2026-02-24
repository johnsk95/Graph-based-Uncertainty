"""
Question Generation Module for Verified Claims QA Experiment

Generates new questions for QASPER papers using GPT-4.1 that:
- Cannot be answered from title/abstract alone
- Are distinct from existing QASPER questions
- Have ground truth answers derived from the paper content
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


# ==============================================================================
# PROMPTS
# ==============================================================================

SYSTEM_PROMPT = """You are an expert research paper analyst. Your task is to generate questions
about a scientific paper that:
1. Require information BEYOND the title and abstract to answer
2. Are specific to this paper's methodology, results, or technical details
3. Do NOT overlap with the existing questions provided
4. Have clear, factual answers derivable from the paper content

Focus on questions about:
- Specific experimental settings, hyperparameters, or configurations
- Quantitative results and comparisons
- Technical implementation details
- Limitations or future work mentioned
- Specific datasets, baselines, or evaluation metrics used
"""

USER_PROMPT_TEMPLATE = """# Paper Title
{title}

# Abstract
{abstract}

# Full Paper Content
{full_text_formatted}

# Existing Questions (DO NOT generate similar questions)
{existing_questions_list}

# Task
Generate exactly 20 NEW questions about this paper that:
1. Cannot be answered by reading only the title and abstract
2. Are distinct from the existing questions listed above
3. Have specific, factual answers found in the paper

For each question, also provide the correct answer based on the paper content.

# Output Format
Return a JSON object with a "questions" array containing exactly 20 objects:
{{
    "questions": [
        {{
            "question": "Your question here?",
            "answer": "The correct answer based on the paper",
            "evidence_section": "Name of section containing the answer"
        }},
        ...
    ]
}}
"""


# ==============================================================================
# QUESTION GENERATOR CLASS
# ==============================================================================

class QuestionGenerator:
    """
    Generate new questions for papers using GPT-4.1.
    """

    def __init__(self, openai_client, model: str = "gpt-4.1"):
        """
        Initialize the question generator.

        Args:
            openai_client: OpenAI client instance
            model: Model to use for generation
        """
        self.client = openai_client
        self.model = model

    def format_full_text(self, full_text: dict) -> str:
        """
        Convert QASPER full_text structure to readable string.

        Args:
            full_text: {"section_name": [...], "paragraphs": [[...], ...]}

        Returns:
            Formatted string with section headers and paragraphs
        """
        formatted = []
        section_names = full_text.get("section_name", [])
        paragraphs = full_text.get("paragraphs", [])

        for i, section_name in enumerate(section_names):
            formatted.append(f"\n## {section_name}\n")
            if i < len(paragraphs):
                for para in paragraphs[i]:
                    if para and para.strip():
                        formatted.append(para)
                        formatted.append("")  # Empty line between paragraphs

        return "\n".join(formatted)

    def format_existing_questions(self, qas) -> str:
        """
        Format existing QASPER questions as a numbered list.

        QASPER uses columnar format: qas = {'question': [...], 'question_id': [...], ...}
        """
        if not qas:
            return "None"

        # Handle QASPER columnar format
        if isinstance(qas, dict) and 'question' in qas:
            questions = qas['question']
        elif isinstance(qas, list):
            questions = [qa.get("question", "") for qa in qas]
        else:
            return "None"

        return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions) if q])

    def generate_questions(
        self,
        paper: dict,
        max_retries: int = 2
    ) -> List[Dict]:
        """
        Generate 20 new questions for a paper.

        Args:
            paper: QASPER paper dict with id, title, abstract, full_text, qas
            max_retries: Number of retries on failure

        Returns:
            List of {"question": str, "answer": str, "evidence_section": str}
        """
        full_text_formatted = self.format_full_text(paper.get("full_text", {}))
        existing_questions = self.format_existing_questions(paper.get("qas", []))

        # Truncate if too long (avoid hitting context limits)
        if len(full_text_formatted) > 100000:
            full_text_formatted = full_text_formatted[:100000] + "\n\n[Content truncated...]"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            full_text_formatted=full_text_formatted,
            existing_questions_list=existing_questions
        )

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                result = json.loads(content)

                # Handle both direct array and wrapped object responses
                if isinstance(result, list):
                    questions = result
                elif "questions" in result:
                    questions = result["questions"]
                else:
                    # Try to find any list in the result
                    for key, value in result.items():
                        if isinstance(value, list):
                            questions = value
                            break
                    else:
                        raise ValueError(f"Unexpected response format: {list(result.keys())}")

                if len(questions) != 20:
                    print(f"  Warning: Got {len(questions)} questions instead of 20")

                return questions

            except json.JSONDecodeError as e:
                print(f"  JSON parse error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    raise
            except Exception as e:
                print(f"  Error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    raise

        return []

    def validate_questions(
        self,
        questions: List[Dict],
        existing_qas,
        similarity_threshold: float = 0.85
    ) -> List[Dict]:
        """
        Filter out questions that are too similar to existing ones.
        Uses simple Jaccard similarity for efficiency.

        Args:
            questions: Generated questions
            existing_qas: Existing QASPER questions (columnar dict or list)
            similarity_threshold: Jaccard similarity threshold for filtering

        Returns:
            Filtered list of questions
        """
        # Handle QASPER columnar format
        if isinstance(existing_qas, dict) and 'question' in existing_qas:
            existing_questions = [q.lower() for q in existing_qas['question']]
        elif isinstance(existing_qas, list):
            existing_questions = [
                qa.get("question", "").lower()
                for qa in existing_qas
            ]
        else:
            existing_questions = []

        def jaccard_similarity(q1: str, q2: str) -> float:
            tokens1 = set(q1.lower().split())
            tokens2 = set(q2.lower().split())
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            return len(intersection) / len(union) if union else 0

        filtered = []
        for q in questions:
            q_text = q.get("question", "")
            if not q_text:
                continue

            max_sim = max(
                (jaccard_similarity(q_text, eq) for eq in existing_questions),
                default=0
            )

            if max_sim < similarity_threshold:
                filtered.append(q)
            else:
                print(f"    Filtered similar question: {q_text[:50]}...")

        return filtered


# ==============================================================================
# MAIN GENERATION FUNCTION
# ==============================================================================

def generate_questions_for_papers(
    papers: List[Dict],
    output_path: str,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4.1"
) -> Dict:
    """
    Generate questions for all papers and save to JSON.

    Args:
        papers: List of QASPER paper dicts
        output_path: Path to save JSON output
        openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        model: Model to use for generation

    Returns:
        Generated questions dict
    """
    import os
    from openai import OpenAI

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")

    client = OpenAI(api_key=api_key)
    generator = QuestionGenerator(client, model=model)

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": model,
        "num_papers": 0,
        "total_questions": 0,
        "papers": []
    }

    for i, paper in enumerate(papers):
        paper_id = paper.get("id", f"paper_{i}")
        print(f"[{i+1}/{len(papers)}] Generating questions for paper: {paper_id}")

        try:
            questions = generator.generate_questions(paper)
            questions = generator.validate_questions(
                questions,
                paper.get("qas", [])
            )

            # Add question IDs
            for j, q in enumerate(questions):
                q["question_id"] = f"gen_{paper_id}_{j+1:03d}"

            paper_entry = {
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "num_existing_questions": len(paper.get("qas", [])),
                "num_generated_questions": len(questions),
                "generated_questions": questions
            }
            output["papers"].append(paper_entry)
            output["total_questions"] += len(questions)

            print(f"  Generated {len(questions)} questions")

        except Exception as e:
            print(f"  Error generating questions for {paper_id}: {e}")
            # Add entry with error
            output["papers"].append({
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "error": str(e),
                "generated_questions": []
            })

    output["num_papers"] = len([p for p in output["papers"] if p.get("generated_questions")])

    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {output['num_papers']} papers with {output['total_questions']} questions to {output_path}")
    return output


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.qasper_data import load_qasper_dataset

    parser = argparse.ArgumentParser(description="Generate questions for QASPER papers")
    parser.add_argument("--split", default="test", help="QASPER split to use")
    parser.add_argument("--num_papers", type=int, default=None, help="Number of papers to process")
    parser.add_argument("--paper_ids", nargs="+", help="Specific paper IDs to process")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model for generation")

    args = parser.parse_args()

    # Load papers
    print(f"Loading QASPER {args.split} split...")
    papers = load_qasper_dataset(args.split)

    # Filter by paper IDs if specified
    if args.paper_ids:
        papers = [p for p in papers if p["id"] in args.paper_ids]
        print(f"Filtered to {len(papers)} papers by ID")

    # Limit number of papers
    if args.num_papers:
        papers = papers[:args.num_papers]

    print(f"Processing {len(papers)} papers...")

    # Generate questions
    generate_questions_for_papers(
        papers=papers,
        output_path=args.output,
        model=args.model
    )
