"""
Weighted Edge Construction Module for Phase 1
Implements weighted bipartite graph construction with knowledge priors
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import src.utils as utils


class WeightedEdgeConstruction():
    """
    Constructs weighted bipartite graph with priors.

    Edge weights are assigned based on:
    - Anchor Links (w=3.0): Prior → Claim from prior
    - Agreement Links (w=2.0): Response → Claim also in prior
    - Neutral Links (w=1.0): Response → Novel claim (not in prior)
    - Contradictory Links (w=0.5): Response → Claim contradicting prior
    """

    # Edge weight constants (Section 4.2, Step 1.1)
    W_ANCHOR = 3.0
    W_AGREE = 2.0
    W_NEUTRAL = 1.0
    W_CONTRA = 0.5

    def __init__(self, args, source_data, llm_model, folder_name=''):
        self.args = args
        self.source_data = source_data
        self.llm_model = llm_model
        self.folder_name = folder_name

        # Cache files
        self.raw_results_path = f'{folder_name}/weighted_graph_raw_return.json'
        self.collected_results_path = f'{folder_name}/{self.args.dataset}_{self.args.model}_weighted_graph.json'

        # Cached results structure
        self.cached_results = {
            'faithfulness': [],  # Faithfulness checks (Response/Prior → Claim)
            'contradiction': []  # Contradiction checks (Claim vs Prior claims)
        }

        # Load cache if exists
        if Path(self.raw_results_path).exists():
            with open(self.raw_results_path, 'r') as f:
                self.cached_results = json.load(f)
            print(f'Loaded from cache: {self.raw_results_path}')

    def perform_faithfulness_check(self, source, claim):
        """
        Check if a source (response or prior) entails/supports a claim.

        Args:
            source: Response text or prior text
            claim: Claim text

        Returns:
            Dict with 'return' and 'prompt'
        """
        prompt = f"""Context: {source}
Claim: {claim}
Is the claim supported by the context above?
Answer Yes or No:
"""

        model_response = self.llm_model.generate_given_prompt([
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ])

        return {
            'return': model_response['generation'],
            'prompt': prompt
        }

    def perform_contradiction_check(self, claim, prior_claims):
        """
        Check if a claim contradicts any of the prior claims using LLM judge.

        Args:
            claim: Claim to check
            prior_claims: List of claims from the prior

        Returns:
            Dict with 'contradicts' (bool) and 'raw_response'
        """
        # Combine all prior claims into context
        prior_context = "\n".join([f"- {pc}" for pc in prior_claims])

        prompt = f"""Prior knowledge (established facts):
{prior_context}

New claim to evaluate: {claim}

Does the new claim contradict any of the prior knowledge listed above?
Answer Yes if the claim contradicts the prior knowledge, or No if it is consistent or unrelated.
Answer Yes or No:
"""

        model_response = self.llm_model.generate_given_prompt([
            {'role': 'system', 'content': 'You are a helpful assistant that identifies contradictions in factual claims.'},
            {'role': 'user', 'content': prompt}
        ])

        response_text = model_response['generation'].strip()

        # Parse response
        response_clean = ''.join([char for char in response_text if char.isalpha()]).lower()
        contradicts = response_clean == 'yes'

        return {
            'contradicts': contradicts,
            'raw_response': response_text,
            'prompt': prompt
        }

    def construct_weighted_graph_single(self, instance_data, instance_id):
        """
        Construct weighted bipartite graph for a single instance.

        Args:
            instance_data: Dictionary containing:
                - 'all_generations': List of N response texts
                - 'breakdown': List of all claims (from responses)
                - 'prior_claims': List of claims from prior (WikiData)
                - 'pointwise_dict': List of dicts for each claim
            instance_id: Index for caching

        Returns:
            Dictionary with weighted adjacency matrices and edge types
        """
        all_claims = instance_data['breakdown']
        prior_claims = instance_data.get('prior_claims', [])
        all_generations = instance_data['all_generations']

        num_responses = len(all_generations)
        num_priors = 1 if prior_claims else 0  # We treat all prior claims as coming from one "prior" source
        num_claims = len(all_claims)

        # Initialize edge weight matrix
        # Rows: [Prior_0, Response_0, Response_1, ..., Response_{N-1}]
        # Cols: Claims
        edge_weights = np.zeros((num_priors + num_responses, num_claims))
        edge_types = [['' for _ in range(num_claims)] for _ in range(num_priors + num_responses)]

        # Track which claims are in the prior
        claims_in_prior = set()
        contradictory_claims = set()

        # Initialize cache structures if needed
        if instance_id >= len(self.cached_results['faithfulness']):
            self.cached_results['faithfulness'].append({})
        if instance_id >= len(self.cached_results['contradiction']):
            self.cached_results['contradiction'].append({})

        instance_faith_cache = self.cached_results['faithfulness'][instance_id]
        instance_contra_cache = self.cached_results['contradiction'][instance_id]

        # Step 1: Process Prior → Claims (Anchor Links)
        if prior_claims:
            print(f"  Processing prior with {len(prior_claims)} claims...")

            # Create a combined prior text for faithfulness checking
            prior_text = " ".join(prior_claims)

            for claim_idx, claim in enumerate(all_claims):
                cache_key = f"prior_0_claim_{claim_idx}"

                # Check if this claim is entailed by prior
                if cache_key in instance_faith_cache:
                    faith_result = instance_faith_cache[cache_key]
                else:
                    faith_result = self.perform_faithfulness_check(prior_text, claim)
                    instance_faith_cache[cache_key] = faith_result

                # Parse response
                response = faith_result['return'].strip()
                response_clean = ''.join([char for char in response if char.isalpha()]).lower()
                is_supported = response_clean == 'yes'

                if is_supported:
                    # Anchor link: Prior entails this claim
                    edge_weights[0, claim_idx] = self.W_ANCHOR
                    edge_types[0][claim_idx] = 'anchor'
                    claims_in_prior.add(claim_idx)

        # Step 2: Identify contradictory claims
        print(f"  Checking for contradictions...")
        for claim_idx, claim in enumerate(all_claims):
            if claim_idx in claims_in_prior:
                # Skip claims already in prior (they can't contradict it)
                continue

            cache_key = f"claim_{claim_idx}_contradiction"

            if cache_key in instance_contra_cache:
                contra_result = instance_contra_cache[cache_key]
            else:
                contra_result = self.perform_contradiction_check(claim, prior_claims)
                instance_contra_cache[cache_key] = contra_result

            if contra_result['contradicts']:
                contradictory_claims.add(claim_idx)

        # Step 3: Process Responses → Claims
        print(f"  Processing {num_responses} responses...")
        for resp_idx, response in enumerate(all_generations):
            for claim_idx, claim in enumerate(all_claims):
                cache_key = f"resp_{resp_idx}_claim_{claim_idx}"

                # Check faithfulness
                if cache_key in instance_faith_cache:
                    faith_result = instance_faith_cache[cache_key]
                else:
                    faith_result = self.perform_faithfulness_check(response, claim)
                    instance_faith_cache[cache_key] = faith_result

                # Parse response
                response_text = faith_result['return'].strip()
                response_clean = ''.join([char for char in response_text if char.isalpha()]).lower()
                is_supported = response_clean == 'yes'

                if not is_supported:
                    # No edge (weight = 0)
                    edge_weights[num_priors + resp_idx, claim_idx] = 0.0
                    edge_types[num_priors + resp_idx][claim_idx] = 'none'
                    continue

                # Determine edge type based on claim category
                if claim_idx in contradictory_claims:
                    # Contradictory link
                    edge_weights[num_priors + resp_idx, claim_idx] = self.W_CONTRA
                    edge_types[num_priors + resp_idx][claim_idx] = 'contradictory'
                elif claim_idx in claims_in_prior:
                    # Agreement link
                    edge_weights[num_priors + resp_idx, claim_idx] = self.W_AGREE
                    edge_types[num_priors + resp_idx][claim_idx] = 'agreement'
                else:
                    # Neutral link (novel claim)
                    edge_weights[num_priors + resp_idx, claim_idx] = self.W_NEUTRAL
                    edge_types[num_priors + resp_idx][claim_idx] = 'neutral'

        # Save cache
        with open(self.raw_results_path, 'w') as f:
            json.dump(self.cached_results, f, indent=2)

        # Compute statistics
        edge_type_counts = {
            'anchor': 0,
            'agreement': 0,
            'neutral': 0,
            'contradictory': 0,
            'none': 0
        }

        for row in edge_types:
            for etype in row:
                if etype in edge_type_counts:
                    edge_type_counts[etype] += 1

        return {
            'edge_weights': edge_weights.tolist(),
            'edge_types': edge_types,
            'claims_in_prior': list(claims_in_prior),
            'contradictory_claims': list(contradictory_claims),
            'edge_type_counts': edge_type_counts,
            'num_priors': num_priors,
            'num_responses': num_responses,
            'num_claims': num_claims
        }

    def construct_all_weighted_graphs(self):
        """
        Construct weighted bipartite graphs for all instances.

        Returns:
            List of instance dictionaries with weighted graph data
        """
        all_results = []

        for instance_id, instance_data in enumerate(tqdm(self.source_data, desc="Constructing Weighted Graphs")):
            print(f"\n[Instance {instance_id + 1}/{len(self.source_data)}]")
            print(f"  Entity: {instance_data.get('entity', 'unknown')}")
            print(f"  Claims: {len(instance_data['breakdown'])}")
            print(f"  Prior claims: {len(instance_data.get('prior_claims', []))}")
            print(f"  Responses: {len(instance_data['all_generations'])}")

            # Construct weighted graph
            graph_result = self.construct_weighted_graph_single(instance_data, instance_id)

            # Add graph data to instance
            instance_result = instance_data.copy()
            instance_result['weighted_graph'] = graph_result

            # Update pointwise_dict with edge information
            for claim_idx, pointwise_dict in enumerate(instance_result['pointwise_dict']):
                # Mark if claim is in prior
                pointwise_dict['in_prior'] = claim_idx in graph_result['claims_in_prior']
                pointwise_dict['is_contradictory'] = claim_idx in graph_result['contradictory_claims']

                # Count edge types for this claim
                claim_edge_types = [row[claim_idx] for row in graph_result['edge_types']]
                pointwise_dict['edge_type_counts'] = {
                    'anchor': claim_edge_types.count('anchor'),
                    'agreement': claim_edge_types.count('agreement'),
                    'neutral': claim_edge_types.count('neutral'),
                    'contradictory': claim_edge_types.count('contradictory'),
                    'none': claim_edge_types.count('none')
                }

                # Average edge weight for this claim
                claim_weights = [graph_result['edge_weights'][i][claim_idx]
                                for i in range(len(graph_result['edge_weights']))]
                pointwise_dict['avg_edge_weight'] = np.mean([w for w in claim_weights if w > 0]) if any(w > 0 for w in claim_weights) else 0.0

            all_results.append(instance_result)

            # Print statistics
            print(f"  Edge types: {graph_result['edge_type_counts']}")
            print(f"  Claims in prior: {len(graph_result['claims_in_prior'])}")
            print(f"  Contradictory claims: {len(graph_result['contradictory_claims'])}")

        # Save results
        with open(self.collected_results_path, 'w') as f:
            json.dump(all_results, f, indent=4)

        print(f"\n✓ Saved weighted graph results to: {self.collected_results_path}")
        return all_results
