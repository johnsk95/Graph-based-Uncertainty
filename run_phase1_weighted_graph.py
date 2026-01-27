"""
Phase 1: Weighted Bipartite Graph Construction with Knowledge Priors
Main pipeline script for the weighted graph framework

This script implements Phase 1 of the technical specification:
1. Fetch WikiData priors for each subject
2. Generate N responses (all with temperature, no greedy)
3. Decompose responses into claims
4. Build weighted bipartite graph with edge types:
   - Anchor (w=3.0): Prior → Prior claim
   - Agreement (w=2.0): Response → Prior claim
   - Neutral (w=1.0): Response → Novel claim
   - Contradictory (w=0.5): Response → Contradictory claim
5. Calculate weighted centrality metrics
6. Categorize claims (grounded/boundary/contradictory)
"""

import argparse
import os
import datasets
import json
from pathlib import Path
import random
import torch

# Import custom modules
import src.utils as utils
from src.models import get_model
from src.wikidata_prior import WikiDataPriorFetcher
from src.model_generate_weighted import WeightedGeneration
from src.break_and_merge import Break_And_Merge
from src.weighted_edge_construction import WeightedEdgeConstruction
from src.weighted_centrality import compute_all_weighted_metrics
from src.visualize_weighted_graph import visualize_weighted_graphs

# Environment setup
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser('~/.cache/huggingface')
OUTPUT_DIR = 'experiments'
DATA_DIR = 'data'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Phase 1: Weighted Bipartite Graph Construction')

    # Model parameters
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='Model name (gpt-3.5-turbo, gpt-4, llama-3-8b-instruct, etc.)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for all response generation (creates variability)')

    # Data parameters
    parser.add_argument('--dataset', type=str, default='pop_qa',
                       help='Dataset name (pop_qa, factscore, etc.)')
    parser.add_argument('--data_size', type=int, default=10,
                       help='Number of examples to process')

    # Generation parameters
    parser.add_argument('--num_generations_per_prompt', type=int, default=5,
                       help='Number of responses to generate (all with temperature)')

    # Graph parameters
    parser.add_argument('--w_anchor', type=float, default=3.0,
                       help='Weight for anchor links (prior → prior claim)')
    parser.add_argument('--w_agree', type=float, default=2.0,
                       help='Weight for agreement links (response → prior claim)')
    parser.add_argument('--w_neutral', type=float, default=1.0,
                       help='Weight for neutral links (response → novel claim)')
    parser.add_argument('--w_contra', type=float, default=0.5,
                       help='Weight for contradictory links')

    # Thresholds for claim categorization
    parser.add_argument('--delta_true', type=float, default=0.6,
                       help='Threshold for grounded/true claims (high centrality)')
    parser.add_argument('--delta_false', type=float, default=0.2,
                       help='Threshold for contradictory/false claims (low centrality)')

    # Pipeline stages
    parser.add_argument('--fetch_priors', type=utils.str2bool, default=True,
                       help='Fetch WikiData priors')
    parser.add_argument('--generate', type=utils.str2bool, default=True,
                       help='Generate LLM responses')
    parser.add_argument('--breakdown', type=utils.str2bool, default=True,
                       help='Decompose into claims')
    parser.add_argument('--construct_graph', type=utils.str2bool, default=True,
                       help='Construct weighted bipartite graph')
    parser.add_argument('--compute_centrality', type=utils.str2bool, default=True,
                       help='Compute weighted centrality metrics')

    # Visualization parameters
    parser.add_argument('--vis', type=utils.str2bool, default=False,
                       help='Generate graph visualizations')
    parser.add_argument('--vis_max_instances', type=int, default=10,
                       help='Maximum number of instances to visualize')
    parser.add_argument('--vis_max_claims', type=int, default=20,
                       help='Maximum number of claims to show per graph')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_samples_for_claims', type=int, default=4,
                       help='Number of samples used for claim extraction')

    args = parser.parse_args()
    return args


def load_popqa_dataset(data_size):
    """
    Load PopQA dataset.

    Args:
        data_size: Number of examples to load

    Returns:
        Dataset with prompts formatted
    """
    ds = datasets.load_from_disk(f'{DATA_DIR}/pop_qa')
    ds = ds.select(range(min(data_size, len(ds))))

    # Format prompts
    def format_popqa(example):
        example['prompt'] = f"Answer the following question: {example['question']}"
        return example

    ds = ds.map(format_popqa)
    return ds


def load_facts_dataset(data_size):
    """
    Load FACTS grounding dataset.

    Args:
        data_size: Number of examples to load

    Returns:
        Dataset with prompts formatted
    """
    ds = datasets.load_dataset('google/FACTS-grounding-public', split='public')
    ds = ds.select(range(min(data_size, len(ds))))

    # Format prompts - use user_request as the prompt
    def format_facts(example):
        example['prompt'] = example['user_request']
        # Use context_document as prior text (will be converted to claims)
        example['prior_text'] = example['context_document']
        # Create a subject identifier from the user request (first 50 chars)
        example['subj'] = example['user_request'][:50] + '...'
        return example

    ds = ds.map(format_facts)
    return ds


def extract_claims_from_text(text, llm_model):
    """
    Extract atomic claims from context document using LLM.

    Args:
        text: Context document text
        llm_model: LLM model for extraction

    Returns:
        List of extracted claims
    """
    prompt = f"""Extract atomic factual claims from the following text. Each claim should be a single, standalone fact.
Format: Return each claim on a new line, numbered.

Text:
{text}

Extracted claims:"""

    result = llm_model.generate_given_prompt(prompt)
    response = result['generation']

    # Parse numbered claims
    lines = response.strip().split('\n')
    claims = []
    for line in lines:
        line = line.strip()
        # Remove numbering like "1.", "1)", etc.
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove leading numbering
            claim = line.lstrip('0123456789.-) ').strip()
            if claim:
                claims.append(claim)

    return claims


def fetch_all_priors(dataset, args):
    """
    Fetch priors for all examples in the dataset.
    For PopQA: Fetch from WikiData
    For FACTS: Extract claims from context_document

    Args:
        dataset: Dataset (PopQA or FACTS)
        args: Arguments

    Returns:
        List of prior dictionaries
    """
    print("\n" + "="*80)

    if args.dataset == 'pop_qa':
        print("FETCHING WIKIDATA PRIORS")
        print("="*80)

        fetcher = WikiDataPriorFetcher(cache_dir=f"{OUTPUT_DIR}/wikidata_cache")

        priors = []
        for i, example in enumerate(dataset):
            print(f"\n[{i+1}/{len(dataset)}] Fetching prior for: {example.get('subj', 'unknown')}")

            s_uri = example.get('s_uri', '')
            if not s_uri:
                print(f"  Warning: No s_uri found, skipping")
                priors.append({
                    'uri': '',
                    'entity_id': '',
                    'subject': example.get('subj', 'unknown'),
                    'claims': [],
                    'raw_data': None
                })
                continue

            prior = fetcher.get_prior_claims(s_uri, example.get('subj'))
            print(f"  Entity ID: {prior['entity_id']}")
            print(f"  Number of claims: {len(prior['claims'])}")

            if prior['claims']:
                print(f"  Sample claims:")
                for j, claim in enumerate(prior['claims'][:3]):
                    print(f"    {j+1}. {claim}")

            priors.append(prior)

    elif args.dataset == 'facts':
        print("EXTRACTING CLAIMS FROM CONTEXT DOCUMENTS")
        print("="*80)

        # Need LLM model for claim extraction
        llm_model = get_model(args.model, args)

        priors = []
        for i, example in enumerate(dataset):
            print(f"\n[{i+1}/{len(dataset)}] Extracting claims from context document")

            prior_text = example.get('prior_text', '')
            if not prior_text:
                print(f"  Warning: No prior_text found, skipping")
                priors.append({
                    'uri': '',
                    'entity_id': '',
                    'subject': example.get('subj', 'unknown'),
                    'claims': [],
                    'raw_data': None
                })
                continue

            # Extract claims from context document
            claims = extract_claims_from_text(prior_text, llm_model)

            print(f"  Number of claims: {len(claims)}")
            if claims:
                print(f"  Sample claims:")
                for j, claim in enumerate(claims[:3]):
                    print(f"    {j+1}. {claim}")

            priors.append({
                'uri': '',
                'entity_id': f'facts_{i}',
                'subject': example.get('subj', 'unknown'),
                'claims': claims,
                'raw_data': prior_text
            })

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported for prior fetching")

    return priors


def main():
    """Main pipeline for Phase 1."""
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    folder_name = f'{OUTPUT_DIR}/{args.dataset}/{args.model}'
    os.makedirs(folder_name, exist_ok=True)

    # Save args
    with open(f'{folder_name}/args_phase1.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "="*80)
    print("PHASE 1: WEIGHTED BIPARTITE GRAPH CONSTRUCTION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Data size: {args.data_size}")
    print(f"Num generations: {args.num_generations_per_prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Output directory: {folder_name}")

    # Load dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)

    if args.dataset == 'pop_qa':
        dataset = load_popqa_dataset(args.data_size)
    elif args.dataset == 'facts':
        dataset = load_facts_dataset(args.data_size)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    print(f"Loaded {len(dataset)} examples")

    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)

    llm_model = get_model(args.model, args)
    print(f"Model initialized: {args.model}")

    # Step 1: Fetch WikiData priors
    priors = None
    priors_path = f'{folder_name}/priors.json'

    if args.fetch_priors:
        if Path(priors_path).exists():
            print(f"\nLoading priors from cache: {priors_path}")
            with open(priors_path, 'r') as f:
                priors = json.load(f)
        else:
            priors = fetch_all_priors(dataset, args)
            with open(priors_path, 'w') as f:
                json.dump(priors, f, indent=2)
            print(f"\nSaved priors to: {priors_path}")
    else:
        if Path(priors_path).exists():
            with open(priors_path, 'r') as f:
                priors = json.load(f)

    # Step 2: Generate responses (all with temperature)
    sequences = None
    if args.generate:
        print("\n" + "="*80)
        print("GENERATING RESPONSES (ALL WITH TEMPERATURE)")
        print("="*80)

        generator = WeightedGeneration(args, dataset, llm_model, folder_name=folder_name)
        sequences = generator.generate()
        print(f"\nGenerated responses for {len(sequences)} examples")
    else:
        # Load from cache
        gen_path = f'{folder_name}/{args.dataset}_{args.model}_weighted_generations.json'
        if Path(gen_path).exists():
            with open(gen_path, 'r') as f:
                sequences = json.load(f)
            print(f"\nLoaded generations from cache: {gen_path}")

    # Add priors to sequences
    if priors and sequences:
        print("\nAdding priors to sequences...")
        for i, (seq, prior) in enumerate(zip(sequences, priors)):
            seq['prior_claims'] = prior['claims']
            seq['prior_uri'] = prior['uri']
            seq['prior_entity_id'] = prior['entity_id']

    # Convert all_generations format to expected format for Break_And_Merge
    if sequences and 'all_generations' in sequences[0]:
        print("\nConverting generation format for breakdown compatibility...")
        for seq in sequences:
            # Break_And_Merge expects 'most_likely_generation' and 'more_generations'
            # We have 'all_generations', so split it
            if 'all_generations' in seq:
                all_gens = seq['all_generations']
                seq['most_likely_generation'] = all_gens[0] if all_gens else ""
                seq['more_generations'] = all_gens[1:] if len(all_gens) > 1 else []
                print(f"  Converted {len(all_gens)} generations (1 primary + {len(all_gens)-1} additional)")

    # Step 3: Break down into claims
    if args.breakdown:
        print("\n" + "="*80)
        print("DECOMPOSING INTO CLAIMS")
        print("="*80)

        breakdown = Break_And_Merge(args, sequences, llm_model, folder_name=folder_name)
        sequences = breakdown.break_down_match()
        print(f"\nDecomposed {len(sequences)} sequences into claims")

        # Restore all_generations format for weighted edge construction
        print("\nRestoring all_generations format...")
        for seq in sequences:
            if 'most_likely_generation' in seq and 'more_generations' in seq:
                seq['all_generations'] = [seq['most_likely_generation']] + seq['more_generations']
                print(f"  Restored {len(seq['all_generations'])} generations")

    # Step 4: Construct weighted bipartite graph
    if args.construct_graph:
        print("\n" + "="*80)
        print("CONSTRUCTING WEIGHTED BIPARTITE GRAPH")
        print("="*80)

        # Update edge weights if custom values provided
        WeightedEdgeConstruction.W_ANCHOR = args.w_anchor
        WeightedEdgeConstruction.W_AGREE = args.w_agree
        WeightedEdgeConstruction.W_NEUTRAL = args.w_neutral
        WeightedEdgeConstruction.W_CONTRA = args.w_contra

        graph_constructor = WeightedEdgeConstruction(args, sequences, llm_model, folder_name=folder_name)
        sequences = graph_constructor.construct_all_weighted_graphs()
        print(f"\nConstructed weighted graphs for {len(sequences)} examples")

    # Step 5: Compute weighted centrality metrics
    if args.compute_centrality:
        print("\n" + "="*80)
        print("COMPUTING WEIGHTED CENTRALITY METRICS")
        print("="*80)

        for i, seq in enumerate(sequences):
            if 'weighted_graph' not in seq:
                print(f"  Warning: No weighted graph for sequence {i}, skipping")
                continue

            graph_data = seq['weighted_graph']
            edge_weights = torch.tensor(graph_data['edge_weights'])

            # Compute all weighted metrics
            metrics = compute_all_weighted_metrics(
                edge_weights.numpy(),
                graph_data['num_priors'],
                graph_data['num_responses'],
                graph_data['num_claims'],
                args.delta_true,
                args.delta_false
            )

            # Add metrics to sequence
            seq['weighted_metrics'] = {
                k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in metrics.items()
            }

            # Add metrics to pointwise_dict
            for claim_idx, pointwise_dict in enumerate(seq['pointwise_dict']):
                pointwise_dict['weighted_closeness'] = float(metrics['weighted_closeness'][claim_idx])
                pointwise_dict['avg_distance_to_sources'] = float(metrics['avg_distance_to_sources'][claim_idx])
                pointwise_dict['weighted_betweenness'] = float(metrics['betweenness_centrality'][claim_idx])
                pointwise_dict['weighted_eigenvector'] = float(metrics['eigenvector_centrality'][claim_idx])
                pointwise_dict['weighted_pagerank'] = float(metrics['pagerank'][claim_idx])
                pointwise_dict['weighted_degree'] = float(metrics['weighted_degree'][claim_idx])

                # Claim category
                if claim_idx in metrics['claim_categories']['grounded']:
                    pointwise_dict['category'] = 'grounded'
                elif claim_idx in metrics['claim_categories']['boundary']:
                    pointwise_dict['category'] = 'boundary'
                elif claim_idx in metrics['claim_categories']['contradictory']:
                    pointwise_dict['category'] = 'contradictory'
                else:
                    pointwise_dict['category'] = 'unknown'

        # Save final results
        final_path = f'{folder_name}/phase1_weighted_graph_final.json'
        with open(final_path, 'w') as f:
            json.dump(sequences, f, indent=2)

        print(f"\n✓ Saved final results to: {final_path}")

        # Print statistics
        print("\n" + "="*80)
        print("PHASE 1 STATISTICS")
        print("="*80)

        total_claims = sum(len(seq['breakdown']) for seq in sequences)
        total_grounded = sum(len(seq['weighted_metrics']['claim_categories']['grounded']) for seq in sequences if 'weighted_metrics' in seq)
        total_boundary = sum(len(seq['weighted_metrics']['claim_categories']['boundary']) for seq in sequences if 'weighted_metrics' in seq)
        total_contradictory = sum(len(seq['weighted_metrics']['claim_categories']['contradictory']) for seq in sequences if 'weighted_metrics' in seq)

        print(f"Total examples processed: {len(sequences)}")
        print(f"Total claims: {total_claims}")
        print(f"Grounded claims: {total_grounded} ({100*total_grounded/total_claims:.1f}%)")
        print(f"Boundary claims: {total_boundary} ({100*total_boundary/total_claims:.1f}%)")
        print(f"Contradictory claims: {total_contradictory} ({100*total_contradictory/total_claims:.1f}%)")

    # Step 6: Generate visualizations (optional)
    if args.vis:
        print("\n" + "="*80)
        print("GENERATING GRAPH VISUALIZATIONS")
        print("="*80)

        vis_dir = f'{folder_name}/visualizations'

        try:
            visualize_weighted_graphs(
                sequences,
                output_dir=vis_dir,
                max_instances=args.vis_max_instances,
                max_claims=args.vis_max_claims
            )
            print(f"\n✓ Visualizations saved to: {vis_dir}")
        except Exception as e:
            print(f"\n✗ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
