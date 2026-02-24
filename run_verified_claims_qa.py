#!/usr/bin/env python3
"""
Verified Claims QA Experiment Runner

This experiment evaluates whether verified/grounded claims from the knowledge
graph can serve as effective context for question answering.

Conditions:
    1. VERIFIED_CLAIMS: Grounded claims from knowledge graph
    2. BASELINE_ABSTRACT: Title + Abstract only
    3. BASELINE_FULL: Full paper text

All conditions use the same QA model (GPT-4o-mini) for fair comparison.

Usage:
    # Step 1: Generate questions (if not already done)
    python run_verified_claims_qa.py --generate_questions --num_papers 5

    # Step 2: Run QA experiment with judging
    python run_verified_claims_qa.py --run_qa --questions_path <path>

    # Full pipeline (generate + QA + judge)
    python run_verified_claims_qa.py --full_pipeline --num_papers 50
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Verified Claims QA Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument(
        "--generate_questions",
        action="store_true",
        help="Generate new questions for papers"
    )
    parser.add_argument(
        "--run_qa",
        action="store_true",
        help="Run QA experiment (requires generated questions)"
    )
    parser.add_argument(
        "--full_pipeline",
        action="store_true",
        help="Run full pipeline: generate questions + QA + judging"
    )

    # Data paths
    parser.add_argument(
        "--results_path",
        type=str,
        default="experiments/qasper/experiment_gpt4o/results.json",
        help="Path to knowledge expansion experiment results"
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default=None,
        help="Path to generated questions JSON (for --run_qa mode)"
    )

    # Paper selection
    parser.add_argument(
        "--num_papers",
        type=int,
        default=None,
        help="Number of papers to process (None for all)"
    )
    parser.add_argument(
        "--paper_ids",
        nargs="+",
        help="Specific paper IDs to process"
    )

    # Model settings
    parser.add_argument(
        "--question_gen_model",
        type=str,
        default="gpt-4.1",
        help="Model for question generation"
    )
    parser.add_argument(
        "--qa_model",
        type=str,
        default="gpt-4o-mini",
        help="Model for question answering"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4.1",
        help="Model for answer judging"
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/qasper/verified_claims_qa",
        help="Output directory for results"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for this experiment run"
    )

    # Experiment settings
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["verified_claims", "baseline_abstract", "baseline_full"],
        help="Conditions to run"
    )
    parser.add_argument(
        "--skip_judging",
        action="store_true",
        help="Skip LLM judging (useful for debugging)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    return parser.parse_args()


def load_papers_from_qasper(paper_ids: List[str], split: str = "test") -> Dict[str, Dict]:
    """
    Load QASPER papers by ID.

    Returns:
        Dict mapping paper_id -> paper dict
    """
    from src.qasper_data import load_qasper_dataset

    print(f"Loading QASPER {split} split...")
    all_papers = load_qasper_dataset(split)

    papers_by_id = {p["id"]: p for p in all_papers}

    return {pid: papers_by_id[pid] for pid in paper_ids if pid in papers_by_id}


def run_question_generation(
    results_path: str,
    output_path: str,
    num_papers: Optional[int] = None,
    paper_ids: Optional[List[str]] = None,
    model: str = "gpt-4.1"
) -> str:
    """
    Generate questions for papers from the knowledge expansion results.

    Returns:
        Path to generated questions file
    """
    from src.question_generation import generate_questions_for_papers

    print("\n" + "=" * 60)
    print("STEP 1: QUESTION GENERATION")
    print("=" * 60)

    # Load results to get paper IDs
    with open(results_path, 'r') as f:
        results = json.load(f)

    available_paper_ids = [p["paper_id"] for p in results.get("individual_results", [])]

    # Filter paper IDs
    if paper_ids:
        target_ids = [pid for pid in paper_ids if pid in available_paper_ids]
    else:
        target_ids = available_paper_ids

    if num_papers:
        target_ids = target_ids[:num_papers]

    print(f"Generating questions for {len(target_ids)} papers...")

    # Load full paper data from QASPER
    papers = load_papers_from_qasper(target_ids)
    paper_list = [papers[pid] for pid in target_ids if pid in papers]

    print(f"Loaded {len(paper_list)} papers from QASPER")

    # Generate questions
    generate_questions_for_papers(
        papers=paper_list,
        output_path=output_path,
        model=model
    )

    return output_path


def run_qa_experiment(
    results_path: str,
    questions_path: str,
    output_path: str,
    qa_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4.1",
    conditions: List[str] = None,
    skip_judging: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Run the QA experiment with LLM judging.

    Returns:
        Complete results dict
    """
    import os
    from openai import OpenAI
    from src.claims_qa import (
        ClaimsBasedQA,
        load_verified_claims_from_results,
    )
    from src.llm_judge import LLMJudge, evaluate_condition, aggregate_all_metrics

    if conditions is None:
        conditions = ["verified_claims", "baseline_abstract", "baseline_full"]

    print("\n" + "=" * 60)
    print("STEP 2: QUESTION ANSWERING")
    print("=" * 60)

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    client = OpenAI(api_key=api_key)
    qa_system = ClaimsBasedQA(client, model=qa_model)
    judge = LLMJudge(client, model=judge_model)

    # Load generated questions
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)

    questions_by_paper = {
        p["paper_id"]: p["generated_questions"]
        for p in questions_data.get("papers", [])
        if p.get("generated_questions")
    }

    print(f"Loaded questions for {len(questions_by_paper)} papers")

    # Load QASPER papers for full text
    paper_ids = list(questions_by_paper.keys())
    papers = load_papers_from_qasper(paper_ids)

    print(f"Loaded {len(papers)} papers from QASPER")

    # Results storage
    all_results = {
        "experiment_timestamp": datetime.utcnow().isoformat() + "Z",
        "qa_model": qa_model,
        "judge_model": judge_model,
        "conditions": conditions,
        "results_source": results_path,
        "questions_source": questions_path,
        "papers": [],
        "aggregate_metrics": {}
    }

    # Process each paper
    for i, paper_id in enumerate(paper_ids):
        print(f"\n[{i+1}/{len(paper_ids)}] Processing paper: {paper_id}")

        if paper_id not in papers:
            print(f"  Paper not found in QASPER, skipping")
            continue

        paper = papers[paper_id]
        questions = questions_by_paper[paper_id]

        if not questions:
            print(f"  No questions available, skipping")
            continue

        # Load verified claims
        verified_claims = load_verified_claims_from_results(results_path, paper_id)
        print(f"  Verified claims: {len(verified_claims)}")
        print(f"  Questions: {len(questions)}")

        # Run QA for all conditions
        print(f"  Running QA across {len(conditions)} conditions...")
        paper_results = qa_system.run_qa_for_paper(
            paper=paper,
            questions=questions,
            verified_claims=verified_claims,
            conditions=conditions
        )

        # Judge answers for each condition
        if not skip_judging:
            print(f"  Judging answers with {judge_model}...")
            for condition in conditions:
                condition_results = paper_results["conditions"][condition]

                # Judge each answer
                judged_results = judge.judge_batch(condition_results, verbose=verbose)
                paper_results["conditions"][condition] = judged_results

                # Compute metrics
                metrics = evaluate_condition(judged_results)
                paper_results[f"{condition}_metrics"] = metrics

                if verbose:
                    print(f"    {condition}: {metrics['accuracy']:.1%} accuracy "
                          f"({metrics['num_correct']}/{metrics['num_questions']})")

        all_results["papers"].append(paper_results)

    # Aggregate metrics
    if not skip_judging:
        all_results["aggregate_metrics"] = aggregate_all_metrics(all_results["papers"])

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    if not skip_judging and all_results["aggregate_metrics"]:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        print(f"\n{'Condition':<20} {'Accuracy':<12} {'Answerable':<12} {'Acc@Ans':<12}")
        print("-" * 56)

        for condition in conditions:
            if condition in all_results["aggregate_metrics"]:
                m = all_results["aggregate_metrics"][condition]
                print(f"{condition:<20} {m['accuracy']:.1%}        "
                      f"{m['answerability_rate']:.1%}        "
                      f"{m['accuracy_when_answerable']:.1%}")

    return all_results


def main():
    args = parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"verified_claims_qa_{timestamp}"
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["timestamp"] = timestamp
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("VERIFIED CLAIMS QA EXPERIMENT")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    questions_path = args.questions_path

    # Step 1: Generate questions (if requested)
    if args.generate_questions or args.full_pipeline:
        questions_path = str(output_dir / "generated_questions.json")
        run_question_generation(
            results_path=args.results_path,
            output_path=questions_path,
            num_papers=args.num_papers,
            paper_ids=args.paper_ids,
            model=args.question_gen_model
        )

    # Step 2: Run QA experiment (if requested)
    if args.run_qa or args.full_pipeline:
        if questions_path is None:
            print("Error: --questions_path required for --run_qa mode")
            return

        results_path = str(output_dir / "qa_results.json")
        run_qa_experiment(
            results_path=args.results_path,
            questions_path=questions_path,
            output_path=results_path,
            qa_model=args.qa_model,
            judge_model=args.judge_model,
            conditions=args.conditions,
            skip_judging=args.skip_judging,
            verbose=args.verbose
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
