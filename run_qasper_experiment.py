#!/usr/bin/env python3
"""
QASPER Knowledge Expansion Experiment Runner

Run the progressive section expansion experiment on the QASPER dataset.

Usage:
    # Run on 10 papers from validation set
    python run_qasper_experiment.py --num_samples 10

    # Run on all papers with LLM verifier
    python run_qasper_experiment.py --verifier_type llm --openai_model gpt-4

    # Run with custom parameters
    python run_qasper_experiment.py --num_samples 50 --iterations 3 --claims_per_iter 15
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# CLAIM EXTRACTOR
# ==============================================================================

class ClaimExtractor:
    """
    Simple claim extractor that decomposes text into atomic claims using an LLM.

    Uses the same prompt as BreakdownProcessor but with a simpler interface.
    """

    PROMPT_TEMPLATE = """Please deconstruct the following paragraph into the smallest possible standalone self-contained facts without semantic repetition, and return the output as a jsonl, where each line is {{"claim": [CLAIM], "confidence": [CONF]}}.

The confidence score [CONF] should represent your confidence in the claim, where 1.0 is obvious facts like 'The earth is round' and 0.0 is for claims that are very obscure or difficult to verify.

The input is:
'{text}'"""

    def __init__(self, llm_model: Any, max_claims_per_text: int = 20):
        """
        Initialize the claim extractor.

        Args:
            llm_model: LLM model with generate_given_prompt method
            max_claims_per_text: Maximum claims to extract per text
        """
        self.llm_model = llm_model
        self.max_claims_per_text = max_claims_per_text

    def extract_claims(self, text: str) -> List[str]:
        """
        Extract atomic claims from text.

        Args:
            text: Text to decompose into claims

        Returns:
            List of claim strings
        """
        if not text or not text.strip():
            return []

        # Truncate very long texts
        if len(text) > 4000:
            text = text[:4000] + "..."

        prompt = self.PROMPT_TEMPLATE.format(text=text)

        try:
            response = self.llm_model.generate_given_prompt(prompt)

            # Handle dict response from models.py
            if isinstance(response, dict):
                output = response.get('generation', str(response))
            else:
                output = str(response)

            # Parse the JSONL output
            claims = self._parse_claims(output)
            return claims[:self.max_claims_per_text]

        except Exception as e:
            print(f"Error extracting claims: {e}")
            return []

    def _parse_claims(self, output: str) -> List[str]:
        """Parse JSONL output to extract claim strings."""
        claims = []

        # Clean up the output
        output = output.replace("```jsonl", "").replace("```json", "").replace("```", "")
        output = output.strip()

        # Try to find JSON objects
        # First, try line-by-line parsing
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue

            claim = self._extract_claim_from_line(line)
            if claim:
                claims.append(claim)

        # If no claims found, try to find JSON objects in the text
        if not claims:
            claims = self._extract_claims_from_text(output)

        return claims

    def _extract_claim_from_line(self, line: str) -> Optional[str]:
        """Extract a claim from a single line."""
        try:
            # Try to parse as JSON
            data = json.loads(line)
            if isinstance(data, dict):
                return data.get('claim', data.get('Claim', None))
        except json.JSONDecodeError:
            pass

        # Try regex extraction
        match = re.search(r'"claim"\s*:\s*"([^"]+)"', line, re.IGNORECASE)
        if match:
            return match.group(1)

        match = re.search(r'"claim"\s*:\s*\[?"([^"]+)"', line, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_claims_from_text(self, text: str) -> List[str]:
        """Extract claims from unstructured text using regex."""
        claims = []

        # Find all JSON-like objects
        pattern = r'\{[^{}]*"claim"[^{}]*\}'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

        for match in matches:
            try:
                # Clean up common issues
                match = match.replace("'", '"')
                match = re.sub(r'(\w+):', r'"\1":', match)  # Quote keys
                data = json.loads(match)
                if 'claim' in data:
                    claims.append(data['claim'])
                elif 'Claim' in data:
                    claims.append(data['Claim'])
            except json.JSONDecodeError:
                # Try regex as fallback
                claim_match = re.search(r'"claim"\s*:\s*"([^"]+)"', match, re.IGNORECASE)
                if claim_match:
                    claims.append(claim_match.group(1))

        return claims

from src.qasper_data import (
    load_qasper_dataset,
    filter_papers_with_complete_structure,
    get_papers_with_answerable_questions,
    sample_papers,
    get_paper_summary,
    validate_paper_structure
)
from src.qasper_experiment import (
    QASPERExperiment,
    run_batch_experiment,
    aggregate_results
)
from src.expert_verification import create_verifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QASPER Knowledge Expansion Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset options
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="QASPER dataset split to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of papers to process (None for all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling papers"
    )
    parser.add_argument(
        "--require_methods",
        action="store_true",
        help="Only include papers with explicit Methods section"
    )
    parser.add_argument(
        "--min_answerable_questions",
        type=int,
        default=1,
        help="Minimum answerable questions per paper"
    )

    # Experiment parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of expansion iterations (0=baseline, 1=+intro, 2=+methods)"
    )
    parser.add_argument(
        "--claims_per_iter",
        type=int,
        default=10,
        help="Number of claims to verify per iteration"
    )
    parser.add_argument(
        "--response_samples",
        type=int,
        default=3,
        help="Number of response samples per question"
    )

    # Model options
    parser.add_argument(
        "--generator_model",
        type=str,
        default=None,
        help="Model for generating responses (e.g., gpt-3.5-turbo, gpt-4)"
    )
    parser.add_argument(
        "--claim_extractor",
        type=str,
        default="mock",
        choices=["mock", "llm"],
        help="Claim extraction method: 'mock' for placeholder claims, 'llm' for LLM-based extraction"
    )
    parser.add_argument(
        "--claim_extractor_model",
        type=str,
        default=None,
        help="Model for claim extraction (defaults to generator_model if not specified)"
    )
    parser.add_argument(
        "--max_claims_per_text",
        type=int,
        default=20,
        help="Maximum claims to extract per text"
    )
    parser.add_argument(
        "--verifier_type",
        type=str,
        default="mock",
        choices=["mock", "llm", "cli_human"],
        help="Type of expert verifier to use"
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model for LLM verifier"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation"
    )

    # Edge weights
    parser.add_argument("--w_anchor", type=float, default=3.0, help="Anchor edge weight")
    parser.add_argument("--w_agree", type=float, default=2.0, help="Agreement edge weight")
    parser.add_argument("--w_neutral", type=float, default=1.0, help="Neutral edge weight")
    parser.add_argument("--w_contra", type=float, default=0.5, help="Contradiction edge weight")

    # Thresholds
    parser.add_argument("--delta_true", type=float, default=0.6, help="Grounded threshold")
    parser.add_argument("--delta_false", type=float, default=0.2, help="Contradictory threshold")

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/qasper",
        help="Output directory for results"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for this experiment run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    # Mock verifier options
    parser.add_argument(
        "--mock_accept_rate",
        type=float,
        default=0.7,
        help="Acceptance rate for mock verifier"
    )

    # Visualization options
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Generate graph visualizations after experiment"
    )

    return parser.parse_args()


def setup_generator_model(args):
    """Setup the response generator model."""
    if args.generator_model is None:
        print("No generator model specified - using mock responses")
        return None

    try:
        from src.models import get_model

        # Create a simple args-like object for the model
        class ModelArgs:
            def __init__(self):
                self.temperature = args.temperature
                self.num_generations_per_prompt = args.response_samples

        model_args = ModelArgs()
        model = get_model(args.generator_model, model_args)
        print(f"Using generator model: {args.generator_model}")
        return model
    except ImportError as e:
        print(f"Could not import model: {e}")
        print("Using mock responses")
        return None


def setup_claim_extractor(args, generator_model=None):
    """Setup the claim extractor."""
    if args.claim_extractor == "mock":
        print("Using mock claim extraction")
        return None

    # LLM-based claim extraction
    if args.claim_extractor == "llm":
        # Determine which model to use
        extractor_model = None

        if args.claim_extractor_model:
            # Use dedicated model for claim extraction
            try:
                from src.models import get_model

                class ModelArgs:
                    def __init__(self):
                        self.temperature = 0.0  # Deterministic for extraction
                        self.num_generations_per_prompt = 1

                extractor_model = get_model(args.claim_extractor_model, ModelArgs())
                print(f"Using dedicated claim extractor model: {args.claim_extractor_model}")
            except Exception as e:
                print(f"Could not load claim extractor model: {e}")

        elif generator_model is not None:
            # Reuse the generator model
            extractor_model = generator_model
            print("Using generator model for claim extraction")

        else:
            print("No model available for claim extraction - using mock")
            return None

        if extractor_model:
            extractor = ClaimExtractor(
                llm_model=extractor_model,
                max_claims_per_text=args.max_claims_per_text
            )
            return extractor

    return None


def setup_verifier(args):
    """Setup the expert verifier."""
    if args.verifier_type == "mock":
        from src.expert_verification import MockExpertVerifier
        import random
        random.seed(args.seed)

        # Create verdict map based on accept rate
        def make_verdict():
            return random.random() < args.mock_accept_rate

        verifier = MockExpertVerifier(
            default_verdict=True  # Will be overridden by random verdicts
        )
        # Override verify_claim to use random verdicts
        original_verify = verifier.verify_claim

        def random_verify(claim_id, claim_text, context):
            result = original_verify(claim_id, claim_text, context)
            result.verdict = random.random() < args.mock_accept_rate
            return result

        verifier.verify_claim = random_verify
        print(f"Using mock verifier with {args.mock_accept_rate * 100:.0f}% accept rate")
        return verifier

    elif args.verifier_type == "llm":
        return create_verifier(
            "llm",
            model=args.openai_model,
            temperature=0.0
        )

    elif args.verifier_type == "cli_human":
        return create_verifier("cli_human")

    else:
        raise ValueError(f"Unknown verifier type: {args.verifier_type}")


def print_dataset_stats(papers, filtered_papers, args):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Total papers loaded: {len(papers)}")
    print(f"Papers with complete structure: {len(filtered_papers)}")

    if args.num_samples and args.num_samples < len(filtered_papers):
        print(f"Papers to process (sampled): {args.num_samples}")
    else:
        print(f"Papers to process: {len(filtered_papers)}")

    # Sample structure validation
    if filtered_papers:
        sample = filtered_papers[0]
        structure = validate_paper_structure(sample)
        print(f"\nSample paper structure:")
        print(f"  Sections: {structure.num_sections}")
        print(f"  Has intro: {structure.has_intro}")
        print(f"  Has methods: {structure.has_methods}")
        print(f"  Questions: {structure.num_questions}")


def print_experiment_config(args):
    """Print experiment configuration."""
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Claims per iteration: {args.claims_per_iter}")
    print(f"Response samples per question: {args.response_samples}")
    print(f"Verifier type: {args.verifier_type}")
    print(f"Edge weights: anchor={args.w_anchor}, agree={args.w_agree}, "
          f"neutral={args.w_neutral}, contra={args.w_contra}")
    print(f"Thresholds: delta_true={args.delta_true}, delta_false={args.delta_false}")


def main():
    args = parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"qasper_{args.split}_{timestamp}"
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("QASPER KNOWLEDGE EXPANSION EXPERIMENT")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    # Load dataset
    print(f"\nLoading QASPER {args.split} split...")
    try:
        papers = load_qasper_dataset(args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTo install the datasets library:")
        print("  pip install datasets")
        return

    # Filter papers
    print("Filtering papers...")
    filtered_papers = filter_papers_with_complete_structure(
        papers,
        require_methods=args.require_methods
    )
    filtered_papers = get_papers_with_answerable_questions(
        filtered_papers,
        min_answerable=args.min_answerable_questions
    )

    # Sample if requested
    if args.num_samples and args.num_samples < len(filtered_papers):
        filtered_papers = sample_papers(filtered_papers, args.num_samples, args.seed)

    print_dataset_stats(papers, filtered_papers, args)
    print_experiment_config(args)

    if len(filtered_papers) == 0:
        print("\nNo papers match the filtering criteria. Exiting.")
        return

    # Setup components
    print("\nInitializing experiment components...")
    generator_model = setup_generator_model(args)
    claim_extractor = setup_claim_extractor(args, generator_model)
    verifier = setup_verifier(args)

    # Create experiment
    experiment = QASPERExperiment(
        generator_model=generator_model,
        claim_extractor=claim_extractor,
        expert_verifier=verifier,
        w_anchor=args.w_anchor,
        w_agree=args.w_agree,
        w_neutral=args.w_neutral,
        w_contra=args.w_contra,
        delta_true=args.delta_true,
        delta_false=args.delta_false,
        output_dir=str(output_dir)
    )

    # Progress callback
    def progress_callback(msg):
        if args.verbose:
            print(f"  {msg}")

    # Run experiment
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT")
    print("=" * 60)

    output_path = str(output_dir / "results.json")

    try:
        aggregated = run_batch_experiment(
            papers=filtered_papers,
            experiment=experiment,
            output_path=output_path,
            num_iterations=args.iterations,
            claims_per_iteration=args.claims_per_iter,
            response_samples_per_question=args.response_samples,
            progress_callback=progress_callback if args.verbose else None
        )

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        for iteration_key in sorted(aggregated.keys()):
            metrics = aggregated[iteration_key]
            print(f"\n{iteration_key}:")
            print(f"  Mean PCR: {metrics['mean_pcr']:.3f} (+/- {metrics['std_pcr']:.3f})")
            print(f"  Mean KF Size: {metrics['mean_kf_size']:.1f}")
            print(f"  Mean Answerability: {metrics['mean_answerability']:.3f}")
            print(f"  Mean VUR: {metrics['mean_vur']:.3f}")
            print(f"  Papers: {metrics['num_papers']}")

        print(f"\nResults saved to: {output_path}")

        # Generate visualizations if requested
        if args.vis:
            print("\n" + "=" * 60)
            print("GENERATING VISUALIZATIONS")
            print("=" * 60)
            try:
                from src.graph_visualization import visualize_results_file
                visualize_results_file(output_path, output_dir)
                print(f"Visualizations saved to: {output_dir}/vis/")
            except ImportError as e:
                print(f"Could not import visualization module: {e}")
                print("Install matplotlib: pip install matplotlib")
            except Exception as e:
                print(f"Error generating visualizations: {e}")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print(f"Partial results may be available at: {output_path}")

    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
