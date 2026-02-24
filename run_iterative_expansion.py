"""
Iterative Knowledge Expansion Pipeline (Section 4.4)

This script runs the iterative knowledge expansion stage, which selects
the most informative contested claims for human/expert verification.

This stage operates AFTER Phase 1 (weighted graph construction) and
uses the claim collapsing results to identify boundary claims.

Usage:
    python run_iterative_expansion.py \
        --model gpt-3.5-turbo \
        --dataset pop_qa \
        --k 20 \
        --lambda1 1.0 \
        --lambda2 1.0
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

import src.utils as utils
from src.embeddings import EmbeddingModel, compute_claim_embeddings, compute_prior_embeddings
from src.iterative_expansion import (
    IterativeKnowledgeExpansion,
    build_graph_from_weighted_data,
    get_node_ids_from_graph_data
)

OUTPUT_DIR = 'experiments'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Iterative Knowledge Expansion: Select informative claims for verification'
    )

    # Data source
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='Model name used in Phase 1')
    parser.add_argument('--dataset', type=str, default='pop_qa',
                       help='Dataset name (pop_qa, facts, etc.)')
    parser.add_argument('--phase1_file', type=str, default=None,
                       help='Path to Phase 1 results file. If not specified, uses default path.')

    # Selection parameters
    parser.add_argument('--k', type=int, default=20,
                       help='Number of claims to select for verification per instance')
    parser.add_argument('--lambda1', type=float, default=1.0,
                       help='Weight for centrality in priority score')
    parser.add_argument('--lambda2', type=float, default=1.0,
                       help='Weight for novelty in priority score')
    parser.add_argument('--k_neighbors', type=int, default=5,
                       help='Number of nearest priors for cost computation')

    # Embedding parameters
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence transformer model for embeddings')

    # Output
    parser.add_argument('--output_suffix', type=str, default='',
                       help='Suffix for output file name')

    args = parser.parse_args()
    return args


def load_phase1_results(args) -> List[Dict]:
    """Load results from Phase 1 weighted graph construction."""
    if args.phase1_file:
        phase1_path = args.phase1_file
    else:
        folder_name = f'{OUTPUT_DIR}/{args.dataset}/{args.model}'
        phase1_path = f'{folder_name}/phase1_weighted_graph_final.json'

    if not Path(phase1_path).exists():
        raise FileNotFoundError(
            f"Phase 1 results not found at {phase1_path}. "
            "Please run run_phase1_weighted_graph.py first."
        )

    with open(phase1_path, 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} instances from Phase 1")
    return results


def extract_claims_and_categories(instance: Dict) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Extract claim lists and their categories from an instance.

    Returns:
        Tuple of (all_claims, grounded_claims, boundary_claims, contradictory_claims)
    """
    all_claims = instance.get('breakdown', [])
    pointwise_dict = instance.get('pointwise_dict', [])

    grounded = []
    boundary = []
    contradictory = []

    for i, pd in enumerate(pointwise_dict):
        category = pd.get('category', 'unknown')
        claim_id = f"C{i}"

        if category == 'grounded':
            grounded.append(claim_id)
        elif category == 'boundary':
            boundary.append(claim_id)
        elif category == 'contradictory':
            contradictory.append(claim_id)

    return all_claims, grounded, boundary, contradictory


def extract_closeness_centrality(instance: Dict) -> Dict[str, float]:
    """Extract closeness centrality scores from pointwise_dict."""
    cc_scores = {}
    for i, pd in enumerate(instance.get('pointwise_dict', [])):
        claim_id = f"C{i}"
        # Use weighted closeness if available, otherwise fall back to other metrics
        cc = pd.get('weighted_closeness', pd.get('closeness_centrality', 0.0))
        cc_scores[claim_id] = cc
    return cc_scores


def process_single_instance(
    instance: Dict,
    instance_id: int,
    embedding_model: EmbeddingModel,
    args
) -> Dict[str, Any]:
    """
    Process a single instance for iterative expansion.

    Args:
        instance: Instance data from Phase 1
        instance_id: Index of the instance
        embedding_model: Pre-initialized embedding model
        args: Command line arguments

    Returns:
        Dictionary with selection results and diagnostics
    """
    entity = instance.get('entity', instance.get('subj', f'instance_{instance_id}'))
    print(f"\n[Instance {instance_id}] Processing: {entity}")

    # Extract claims and categories
    all_claims, grounded, boundary, contradictory = extract_claims_and_categories(instance)
    print(f"  Total claims: {len(all_claims)}")
    print(f"  Grounded: {len(grounded)}, Boundary: {len(boundary)}, Contradictory: {len(contradictory)}")

    if len(boundary) == 0:
        print("  No boundary claims to process. Skipping.")
        return {
            'entity': entity,
            'instance_id': instance_id,
            'selection': {'verify': [], 'discard': []},
            'diagnostics': {},
            'num_boundary': 0,
            'skipped': True
        }

    # Get weighted graph data
    weighted_graph = instance.get('weighted_graph', {})
    if not weighted_graph:
        print("  No weighted graph data found. Skipping.")
        return {
            'entity': entity,
            'instance_id': instance_id,
            'selection': {'verify': [], 'discard': []},
            'diagnostics': {},
            'num_boundary': len(boundary),
            'skipped': True,
            'skip_reason': 'no_weighted_graph'
        }

    num_priors = weighted_graph.get('num_priors', 0)
    num_responses = weighted_graph.get('num_responses', 0)
    num_claims = weighted_graph.get('num_claims', len(all_claims))

    # Build NetworkX graph
    graph = build_graph_from_weighted_data(weighted_graph, num_priors, num_responses, num_claims)

    # Get node IDs
    prior_node_ids, response_node_ids, claim_node_ids = get_node_ids_from_graph_data(
        weighted_graph, num_priors, num_responses
    )

    # Compute claim embeddings
    print("  Computing claim embeddings...")
    claim_emb_dict = {}
    for i, claim_text in enumerate(all_claims):
        emb = embedding_model.encode(claim_text)
        claim_emb_dict[i] = emb.flatten()

    # Compute prior embeddings
    prior_claims = instance.get('prior_claims', [])
    if prior_claims:
        print(f"  Computing embeddings for {len(prior_claims)} prior claims...")
        prior_embeddings = embedding_model.encode(prior_claims)
    else:
        prior_embeddings = np.array([])

    # Get closeness centrality scores
    closeness_centrality = extract_closeness_centrality(instance)

    # Initialize the expander
    expander = IterativeKnowledgeExpansion(
        graph=graph,
        claim_embeddings=claim_emb_dict,
        prior_embeddings=prior_embeddings,
        prior_entailed_claim_ids=grounded,
        response_node_ids=response_node_ids,
        closeness_centrality=closeness_centrality,
        contested_claim_ids=boundary,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        k_neighbors=args.k_neighbors
    )

    # Select claims for verification
    k = min(args.k, len(boundary))
    selection, diagnostics = expander.select_claims(k=k)

    print(f"  Selected {len(selection['verify'])} claims for verification")

    # Add claim texts to selection
    selection_with_text = {
        'verify': [
            {
                'claim_id': cid,
                'claim_text': all_claims[int(cid[1:])] if cid.startswith('C') else '',
                'details': expander.get_claim_details(cid, diagnostics)
            }
            for cid in selection['verify']
        ],
        'discard': [
            {
                'claim_id': cid,
                'claim_text': all_claims[int(cid[1:])] if cid.startswith('C') else ''
            }
            for cid in selection['discard']
        ]
    }

    return {
        'entity': entity,
        'instance_id': instance_id,
        'selection': selection_with_text,
        'diagnostics': {
            k: {str(kk): float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in diagnostics.items()
        },
        'num_claims': len(all_claims),
        'num_grounded': len(grounded),
        'num_boundary': len(boundary),
        'num_contradictory': len(contradictory),
        'num_selected': len(selection['verify']),
        'skipped': False
    }


def print_selection_summary(result: Dict):
    """Print a summary of the selection for an instance."""
    if result.get('skipped'):
        return

    print(f"\n  Top selected claims for verification:")
    for i, claim_info in enumerate(result['selection']['verify'][:5]):
        details = claim_info.get('details', {})
        print(f"    {i+1}. [{claim_info['claim_id']}] {claim_info['claim_text'][:60]}...")
        print(f"       Priority: {details.get('priority_score', 0):.3f}, "
              f"Utility: {details.get('utility', 0):.3f}, "
              f"Cost: {details.get('cost', 0):.3f}")


def main():
    """Main pipeline for iterative knowledge expansion."""
    args = parse_args()

    # Create output directory
    folder_name = f'{OUTPUT_DIR}/{args.dataset}/{args.model}'
    os.makedirs(folder_name, exist_ok=True)

    print("=" * 80)
    print("ITERATIVE KNOWLEDGE EXPANSION (Section 4.4)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"k (selection budget): {args.k}")
    print(f"lambda1 (centrality weight): {args.lambda1}")
    print(f"lambda2 (novelty weight): {args.lambda2}")
    print(f"Embedding model: {args.embedding_model}")

    # Load Phase 1 results
    print("\n" + "=" * 80)
    print("LOADING PHASE 1 RESULTS")
    print("=" * 80)
    phase1_results = load_phase1_results(args)

    # Initialize embedding model
    print("\n" + "=" * 80)
    print("INITIALIZING EMBEDDING MODEL")
    print("=" * 80)
    cache_dir = f'{folder_name}/embedding_cache'
    embedding_model = EmbeddingModel(
        model_name=args.embedding_model,
        cache_dir=cache_dir
    )
    print(f"Embedding dimension: {embedding_model.embedding_dim}")

    # Process all instances
    print("\n" + "=" * 80)
    print("PROCESSING INSTANCES")
    print("=" * 80)

    all_results = []
    total_selected = 0
    total_boundary = 0

    for i, instance in enumerate(phase1_results):
        result = process_single_instance(instance, i, embedding_model, args)
        all_results.append(result)

        if not result.get('skipped'):
            total_selected += result['num_selected']
            total_boundary += result['num_boundary']
            print_selection_summary(result)

    # Save results
    output_suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_path = f'{folder_name}/iterative_expansion_results{output_suffix}.json'

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✓ Saved results to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total instances processed: {len(all_results)}")
    print(f"Total boundary claims: {total_boundary}")
    print(f"Total claims selected for verification: {total_selected}")
    if total_boundary > 0:
        print(f"Selection rate: {100 * total_selected / total_boundary:.1f}%")

    # Print aggregate statistics
    processed = [r for r in all_results if not r.get('skipped')]
    if processed:
        avg_boundary = np.mean([r['num_boundary'] for r in processed])
        avg_selected = np.mean([r['num_selected'] for r in processed])
        print(f"\nPer-instance averages:")
        print(f"  Boundary claims: {avg_boundary:.1f}")
        print(f"  Selected claims: {avg_selected:.1f}")

    print("\n" + "=" * 80)
    print("ITERATIVE EXPANSION COMPLETE!")
    print("=" * 80)

    # Save args for reproducibility
    args_path = f'{folder_name}/iterative_expansion_args{output_suffix}.json'
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    main()
