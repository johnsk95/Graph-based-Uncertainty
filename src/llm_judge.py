"""
LLM Judge Module for Answer Evaluation

Uses GPT-4.1 to judge whether predicted answers are correct
by comparing them to ground truth answers.
"""

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


# ==============================================================================
# PROMPTS
# ==============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for question-answering systems. Your task is to
determine whether a predicted answer is correct by comparing it to the ground
truth answer.

Rules for evaluation:
1. The predicted answer does NOT need to match the ground truth exactly
2. The predicted answer is CORRECT if it conveys the same essential information
3. Minor differences in wording, phrasing, or additional context are acceptable
4. The predicted answer is INCORRECT if it:
   - States different facts than the ground truth
   - Misses key information from the ground truth
   - Says the question cannot be answered when a valid answer exists
   - Provides wrong numbers, names, or specific details

Respond with ONLY "True" or "False" - no explanation needed.
"""

JUDGE_USER_PROMPT_TEMPLATE = """Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted_answer}

Is the predicted answer correct? Respond with only "True" or "False".
"""


# ==============================================================================
# LLM JUDGE CLASS
# ==============================================================================

class LLMJudge:
    """
    Use GPT-4.1 to judge answer correctness.
    """

    def __init__(self, openai_client, model: str = "gpt-4.1"):
        """
        Initialize the judge.

        Args:
            openai_client: OpenAI client instance
            model: Model to use for judging
        """
        self.client = openai_client
        self.model = model

    def judge_answer(
        self,
        question: str,
        ground_truth: str,
        predicted_answer: str
    ) -> Dict:
        """
        Judge whether a predicted answer is correct.

        Args:
            question: The question that was asked
            ground_truth: The correct answer
            predicted_answer: The model's predicted answer

        Returns:
            {"is_correct": bool, "raw_response": str}
        """
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted_answer
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Deterministic judgments
            max_tokens=10
        )

        raw_response = response.choices[0].message.content.strip()

        # Parse boolean response
        is_correct = raw_response.lower() in ["true", "yes", "correct"]

        return {
            "is_correct": is_correct,
            "raw_response": raw_response
        }

    def judge_batch(
        self,
        results: List[Dict],
        verbose: bool = False
    ) -> List[Dict]:
        """
        Judge a batch of QA results.

        Args:
            results: List of {"question": str, "ground_truth": str,
                             "predicted_answer": str, ...}
            verbose: Print progress

        Returns:
            List of results with added "is_correct" and "judge_response" fields
        """
        judged_results = []

        for i, r in enumerate(results):
            if verbose and (i + 1) % 10 == 0:
                print(f"    Judging {i + 1}/{len(results)}...")

            try:
                judgment = self.judge_answer(
                    question=r["question"],
                    ground_truth=r["ground_truth"],
                    predicted_answer=r["predicted_answer"]
                )

                judged_result = r.copy()
                judged_result["is_correct"] = judgment["is_correct"]
                judged_result["judge_response"] = judgment["raw_response"]
                judged_results.append(judged_result)

            except Exception as e:
                print(f"    Error judging question {r.get('question_id', i)}: {e}")
                judged_result = r.copy()
                judged_result["is_correct"] = False
                judged_result["judge_response"] = f"Error: {str(e)}"
                judged_result["judge_error"] = str(e)
                judged_results.append(judged_result)

        return judged_results


# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

def evaluate_condition(results: List[Dict]) -> Dict:
    """
    Compute metrics for a single condition after LLM judging.

    Args:
        results: List of judged QA results with "is_correct" field

    Returns:
        Aggregated metrics dict
    """
    from src.claims_qa import is_unanswerable_response

    if not results:
        return {
            "num_questions": 0,
            "num_correct": 0,
            "accuracy": 0.0,
            "num_answerable": 0,
            "answerability_rate": 0.0,
            "accuracy_when_answerable": 0.0,
            "total_tokens": 0,
            "avg_tokens_per_question": 0
        }

    num_correct = sum(1 for r in results if r.get("is_correct", False))
    num_answerable = sum(
        1 for r in results
        if not is_unanswerable_response(r.get("predicted_answer", ""))
    )
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    n = len(results)

    # Accuracy among answerable questions only
    answerable_results = [
        r for r in results
        if not is_unanswerable_response(r.get("predicted_answer", ""))
    ]
    num_correct_answerable = sum(
        1 for r in answerable_results if r.get("is_correct", False)
    )

    return {
        "num_questions": n,
        "num_correct": num_correct,
        "accuracy": num_correct / n if n > 0 else 0.0,
        "num_answerable": num_answerable,
        "answerability_rate": num_answerable / n if n > 0 else 0.0,
        "accuracy_when_answerable": (
            num_correct_answerable / num_answerable
            if num_answerable > 0 else 0.0
        ),
        "total_tokens": total_tokens,
        "avg_tokens_per_question": total_tokens / n if n > 0 else 0
    }


def aggregate_all_metrics(paper_results: List[Dict]) -> Dict:
    """
    Aggregate metrics across all papers.

    Args:
        paper_results: List of paper result dicts

    Returns:
        Aggregated metrics per condition
    """
    conditions = ["verified_claims", "baseline_abstract", "baseline_full"]

    aggregated = {}

    for condition in conditions:
        metrics_key = f"{condition}_metrics"

        total_questions = 0
        total_correct = 0
        total_answerable = 0
        total_correct_answerable = 0
        total_tokens = 0

        for paper in paper_results:
            if metrics_key in paper:
                m = paper[metrics_key]
                total_questions += m["num_questions"]
                total_correct += m["num_correct"]
                total_answerable += m["num_answerable"]
                # Compute correct among answerable for this paper
                if m["num_answerable"] > 0:
                    total_correct_answerable += int(
                        m["accuracy_when_answerable"] * m["num_answerable"]
                    )
                total_tokens += m["total_tokens"]

        aggregated[condition] = {
            "num_papers": len([p for p in paper_results if metrics_key in p]),
            "total_questions": total_questions,
            "total_correct": total_correct,
            "accuracy": total_correct / total_questions if total_questions > 0 else 0,
            "total_answerable": total_answerable,
            "answerability_rate": (
                total_answerable / total_questions if total_questions > 0 else 0
            ),
            "accuracy_when_answerable": (
                total_correct_answerable / total_answerable
                if total_answerable > 0 else 0
            ),
            "total_tokens": total_tokens
        }

    return aggregated


# ==============================================================================
# CLI FOR TESTING
# ==============================================================================

if __name__ == "__main__":
    import os
    from openai import OpenAI

    # Test the judge
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable to test")
        exit(1)

    client = OpenAI(api_key=api_key)
    judge = LLMJudge(client)

    # Test cases
    test_cases = [
        {
            "question": "What learning rate was used?",
            "ground_truth": "2e-5",
            "predicted_answer": "The model used a learning rate of 2e-5"
        },
        {
            "question": "What dataset was used?",
            "ground_truth": "IMDB movie reviews",
            "predicted_answer": "Cannot be determined from the given information."
        },
        {
            "question": "How many layers does the model have?",
            "ground_truth": "12 transformer layers",
            "predicted_answer": "The model has 6 layers."
        }
    ]

    print("Testing LLM Judge:")
    for i, tc in enumerate(test_cases):
        result = judge.judge_answer(
            tc["question"],
            tc["ground_truth"],
            tc["predicted_answer"]
        )
        print(f"\nTest {i + 1}:")
        print(f"  Question: {tc['question']}")
        print(f"  Ground Truth: {tc['ground_truth']}")
        print(f"  Predicted: {tc['predicted_answer']}")
        print(f"  Judgment: {result['is_correct']} ({result['raw_response']})")
