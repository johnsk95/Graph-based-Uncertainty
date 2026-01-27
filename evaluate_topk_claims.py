"""
Evaluate and compare truthfulness of top-k claims between base and weighted centrality methods.

This script:
1. Loads claims from both methods
2. Filters top-k claims based on their respective thresholds:
   - Base: 40th percentile of SC scores
   - Weighted: closeness score > 0.6
3. Evaluates truthfulness using GPT-4o-mini
4. Auto-labels prior-connected claims as truthful for weighted method
5. Ensures equal number of claims are evaluated across methods
"""

import json
import os
from typing import List, Dict, Tuple
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import argparse

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def calculate_percentile_threshold(claims: List[Dict], percentile: int, score_key: str) -> float:
    """Calculate the threshold value for a given percentile."""
    scores = [claim[score_key] for claim in claims if score_key in claim and claim[score_key] is not None]
    if not scores:
        return 0.0
    return np.percentile(scores, percentile)

def load_base_method_data(file_path: str) -> List[Dict]:
    """Load and parse base method data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_weighted_method_data(file_path: str) -> List[Dict]:
    """Load and parse weighted method data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_top_k_claims_base(data: List[Dict], percentile: int = 40) -> List[Dict]:
    """
    Get top-k claims from base method using percentile threshold on SC scores.
    Returns list of claims with metadata.
    """
    all_claims = []

    for entry in data:
        entity = entry.get('entity', '')
        wiki_title = entry.get('wiki_title', '')

        for claim_dict in entry.get('pointwise_dict', []):
            sc_score = claim_dict.get('sc_score_4samples')
            if sc_score is not None:
                all_claims.append({
                    'claim': claim_dict['claim'],
                    'entity': entity,
                    'wiki_title': wiki_title,
                    'sc_score': sc_score,
                    'method': 'base'
                })

    # Calculate percentile threshold
    threshold = calculate_percentile_threshold(all_claims, percentile, 'sc_score')

    # Filter claims above threshold
    top_claims = [c for c in all_claims if c['sc_score'] >= threshold]

    print(f"Base method: {len(all_claims)} total claims, threshold={threshold:.4f}, {len(top_claims)} top claims")

    return top_claims, threshold

def get_top_k_claims_weighted(data: List[Dict], closeness_threshold: float = 0.6) -> List[Dict]:
    """
    Get top-k claims from weighted method using closeness threshold > 0.6.
    Returns list of claims with metadata.
    """
    all_claims = []

    for entry in data:
        entity = entry.get('entity', '')
        wiki_title = entry.get('wiki_title', '')
        prior_claims = entry.get('prior_claims', [])

        for claim_dict in entry.get('pointwise_dict', []):
            closeness = claim_dict.get('weighted_closeness')
            in_prior = claim_dict.get('in_prior', False)

            if closeness is not None:
                all_claims.append({
                    'claim': claim_dict['claim'],
                    'entity': entity,
                    'wiki_title': wiki_title,
                    'closeness': closeness,
                    'in_prior': in_prior,
                    'category': claim_dict.get('category', ''),
                    'method': 'weighted'
                })

    # Filter claims above threshold
    top_claims = [c for c in all_claims if c['closeness'] >= closeness_threshold]

    # Count prior-connected claims
    prior_connected = sum(1 for c in top_claims if c['in_prior'])

    print(f"Weighted method: {len(all_claims)} total claims, threshold={closeness_threshold}, {len(top_claims)} top claims")
    print(f"  - Prior-connected (auto-labeled as truthful): {prior_connected}")
    print(f"  - Novel claims (need LLM evaluation): {len(top_claims) - prior_connected}")

    return top_claims, closeness_threshold

def evaluate_claim_truthfulness(claim: str, wiki_title: str, entity: str, prior_claims: List[str] = None) -> Dict:
    """
    Use GPT-4o-mini to evaluate if a claim is truthful based on Wikipedia knowledge.

    Returns:
        Dict with keys: 'is_truthful' (bool), 'confidence' (float), 'reasoning' (str)
    """

    prior_context = ""
    if prior_claims:
        prior_context = f"\n\nFor reference, here are some known facts about {entity} from Wikipedia/Wikidata:\n"
        prior_context += "\n".join([f"- {pc}" for pc in prior_claims[:10]])  # Limit to first 10

    prompt = f"""You are evaluating the truthfulness of claims about Wikipedia entities.

Entity: {entity}
Wikipedia Title: {wiki_title}
{prior_context}

Claim to evaluate: "{claim}"

Based on your knowledge of Wikipedia content about "{wiki_title}", evaluate whether this claim is:
1. TRUE - The claim is factually correct according to Wikipedia
2. FALSE - The claim is factually incorrect or contradicts Wikipedia
3. UNVERIFIABLE - Cannot be verified from Wikipedia (too vague, or information not available)

Respond in JSON format:
{{
    "verdict": "TRUE" or "FALSE" or "UNVERIFIABLE",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise fact-checker that evaluates claims against Wikipedia knowledge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        return {
            'is_truthful': result.get('verdict') == 'TRUE',
            'verdict': result.get('verdict', 'UNVERIFIABLE'),
            'confidence': result.get('confidence', 0.0),
            'reasoning': result.get('reasoning', '')
        }

    except Exception as e:
        print(f"Error evaluating claim: {e}")
        return {
            'is_truthful': False,
            'verdict': 'ERROR',
            'confidence': 0.0,
            'reasoning': str(e)
        }

def evaluate_method(claims: List[Dict], method_name: str, weighted_data: List[Dict] = None) -> Dict:
    """
    Evaluate all claims from a method.
    For weighted method, auto-label prior-connected claims as truthful.
    """
    results = []
    truthful_count = 0
    prior_truthful_count = 0

    # Create a lookup for prior claims if weighted method
    prior_lookup = {}
    if method_name == 'weighted' and weighted_data:
        for entry in weighted_data:
            wiki_title = entry.get('wiki_title', '')
            prior_lookup[wiki_title] = entry.get('prior_claims', [])

    for claim_info in tqdm(claims, desc=f"Evaluating {method_name} claims"):
        claim = claim_info['claim']
        wiki_title = claim_info['wiki_title']
        entity = claim_info['entity']

        # For weighted method, check if claim is prior-connected
        if method_name == 'weighted' and claim_info.get('in_prior', False):
            eval_result = {
                'is_truthful': True,
                'verdict': 'PRIOR_CONNECTED',
                'confidence': 1.0,
                'reasoning': 'Claim is connected to prior knowledge (Wikipedia/Wikidata)'
            }
            truthful_count += 1
            prior_truthful_count += 1
        else:
            # Use LLM to evaluate
            prior_claims = prior_lookup.get(wiki_title, []) if method_name == 'weighted' else None
            eval_result = evaluate_claim_truthfulness(claim, wiki_title, entity, prior_claims)
            if eval_result['is_truthful']:
                truthful_count += 1

        results.append({
            'claim': claim,
            'entity': entity,
            'wiki_title': wiki_title,
            'method': method_name,
            'score': claim_info.get('sc_score' if method_name == 'base' else 'closeness'),
            'in_prior': claim_info.get('in_prior', False),
            **eval_result
        })

    return {
        'results': results,
        'total_claims': len(claims),
        'truthful_count': truthful_count,
        'truthful_rate': truthful_count / len(claims) if claims else 0,
        'prior_connected_truthful': prior_truthful_count
    }

def match_claim_counts(base_claims: List[Dict], weighted_claims: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Ensure both methods have the same number of claims by taking top-k from the larger set.
    """
    min_count = min(len(base_claims), len(weighted_claims))

    print(f"\nMatching claim counts: {len(base_claims)} base, {len(weighted_claims)} weighted")
    print(f"Using top {min_count} claims from each method")

    # Sort by score (descending) and take top min_count
    base_claims_sorted = sorted(base_claims, key=lambda x: x['sc_score'], reverse=True)[:min_count]
    weighted_claims_sorted = sorted(weighted_claims, key=lambda x: x['closeness'], reverse=True)[:min_count]

    return base_claims_sorted, weighted_claims_sorted

def main():
    parser = argparse.ArgumentParser(description='Evaluate top-k claims truthfulness')
    parser.add_argument('--base_dir', type=str,
                       default='experiments/pop_qa/llama-3-8b-instruct_base',
                       help='Directory containing base method results')
    parser.add_argument('--weighted_dir', type=str,
                       default='experiments/pop_qa/llama-3-8b-instruct_weighted',
                       help='Directory containing weighted method results')
    parser.add_argument('--base_file', type=str,
                       default='pop_qa_llama-3-8b-instruct_bipartite_sc_5samples_4matches.json',
                       help='Base method result file')
    parser.add_argument('--weighted_file', type=str,
                       default='phase1_weighted_graph_final.json',
                       help='Weighted method result file')
    parser.add_argument('--percentile', type=int, default=40,
                       help='Percentile threshold for base method')
    parser.add_argument('--closeness_threshold', type=float, default=0.6,
                       help='Closeness threshold for weighted method')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("Top-K Claims Truthfulness Evaluation")
    print("="*80)

    # Load data
    print("\n[1] Loading data...")
    base_path = os.path.join(args.base_dir, args.base_file)
    weighted_path = os.path.join(args.weighted_dir, args.weighted_file)

    base_data = load_base_method_data(base_path)
    weighted_data = load_weighted_method_data(weighted_path)

    print(f"Loaded {len(base_data)} entries from base method")
    print(f"Loaded {len(weighted_data)} entries from weighted method")

    # Get top-k claims
    print("\n[2] Filtering top-k claims...")
    base_claims, base_threshold = get_top_k_claims_base(base_data, args.percentile)
    weighted_claims, weighted_threshold = get_top_k_claims_weighted(weighted_data, args.closeness_threshold)

    # Match claim counts
    print("\n[3] Matching claim counts...")
    base_claims, weighted_claims = match_claim_counts(base_claims, weighted_claims)

    # Evaluate claims
    print("\n[4] Evaluating claims...")
    print(f"\nEvaluating {len(base_claims)} claims from base method...")
    base_results = evaluate_method(base_claims, 'base')

    print(f"\nEvaluating {len(weighted_claims)} claims from weighted method...")
    weighted_results = evaluate_method(weighted_claims, 'weighted', weighted_data)

    # Save results
    print("\n[5] Saving results...")
    output = {
        'config': {
            'base_dir': args.base_dir,
            'weighted_dir': args.weighted_dir,
            'base_file': args.base_file,
            'weighted_file': args.weighted_file,
            'percentile': args.percentile,
            'closeness_threshold': args.closeness_threshold,
            'base_threshold_value': base_threshold,
            'weighted_threshold_value': weighted_threshold
        },
        'base_method': base_results,
        'weighted_method': weighted_results,
        'comparison': {
            'base_truthful_rate': base_results['truthful_rate'],
            'weighted_truthful_rate': weighted_results['truthful_rate'],
            'improvement': weighted_results['truthful_rate'] - base_results['truthful_rate'],
            'base_truthful_count': base_results['truthful_count'],
            'weighted_truthful_count': weighted_results['truthful_count'],
            'weighted_prior_connected': weighted_results['prior_connected_truthful'],
            'total_claims_evaluated': len(base_claims)
        }
    }

    output_file = os.path.join(args.output_dir, 'topk_truthfulness_evaluation.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nTotal claims evaluated per method: {len(base_claims)}")
    print(f"\nBase Method (SC score >= {base_threshold:.4f}):")
    print(f"  - Truthful claims: {base_results['truthful_count']}/{base_results['total_claims']}")
    print(f"  - Truthful rate: {base_results['truthful_rate']:.2%}")

    print(f"\nWeighted Method (Closeness >= {weighted_threshold}):")
    print(f"  - Truthful claims: {weighted_results['truthful_count']}/{weighted_results['total_claims']}")
    print(f"  - Truthful rate: {weighted_results['truthful_rate']:.2%}")
    print(f"  - Prior-connected (auto-truthful): {weighted_results['prior_connected_truthful']}")
    print(f"  - LLM-evaluated truthful: {weighted_results['truthful_count'] - weighted_results['prior_connected_truthful']}")

    print(f"\nImprovement: {output['comparison']['improvement']:.2%}")
    print("="*80)

if __name__ == '__main__':
    main()
