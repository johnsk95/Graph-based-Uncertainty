"""
Modified Generation Module for Weighted Bipartite Graph
All responses are generated with the same temperature settings (no greedy generation)
"""

import json
from pathlib import Path


class WeightedGeneration():
    """
    Generation class for weighted bipartite graph framework.
    Unlike the original implementation which generates one greedy response and N-1 stochastic responses,
    this generates all N responses with the same temperature for variability.
    """

    def __init__(self, args, dataset, llm_model, folder_name=''):
        self.args = args
        self.llm_model = llm_model
        self.dataset = dataset
        self.foldername = folder_name
        self.raw_results_path = f'{folder_name}/{self.args.dataset}_raw_return_weighted.json'
        self.collected_results_path = f'{folder_name}/{self.args.dataset}_{self.args.model}_weighted_generations.json'
        self.all_results = []  # Simpler structure: just list of results

        if Path(self.raw_results_path).exists():
            with open(self.raw_results_path, 'r') as f:
                self.all_results = json.load(f)
            print('Loaded from cache', self.raw_results_path)

    def _generate_all_responses(self, prompt, gen_id):
        """
        Generate all N responses with temperature (no greedy generation).

        Args:
            prompt: Input prompt
            gen_id: Generation ID for caching

        Returns:
            List of N generated responses
        """
        if gen_id < len(self.all_results):
            results = self.all_results[gen_id]
        else:
            results = {'generation': [], 'prompt': prompt}
            try_id = 0

            # Generate all responses with temperature
            while len(results['generation']) < self.args.num_generations_per_prompt and try_id < 5:
                # Use generate_n_given_prompt which applies temperature
                batch_generations = self.llm_model.generate_n_given_prompt(prompt)['generation']

                # Filter out refusal responses
                filtered_generations = [g for g in batch_generations if not self.ignore_generation(g)]

                # Add filtered generations up to the required number
                results['generation'] += filtered_generations[:self.args.num_generations_per_prompt - len(results['generation'])]
                try_id += 1
                print(f'Try {try_id}: Generated {len(filtered_generations)} valid responses')

            # Padding with placeholder if needed (though we want to avoid this)
            while len(results['generation']) < self.args.num_generations_per_prompt:
                results['generation'].append('I apologize, I do not know.')
                print(f'Warning: Padded generation {gen_id} with placeholder')

            self.all_results.append(results)

        # Extract generated texts
        generated_texts = []
        for j in range(self.args.num_generations_per_prompt):
            if 'generation' in results:
                text_generation = results['generation'][j]
            else:
                # Legacy format
                text_generation = results["choices"][j]['message']['content'].strip()
            generated_texts.append(text_generation)

        return generated_texts

    def generate(self):
        """
        Generate responses for all prompts in the dataset.

        Returns:
            List of sequence dictionaries containing all generations
        """
        sequences = []

        for gen_id, data in enumerate(self.dataset):
            print(f"\nGenerating responses for example {gen_id + 1}/{len(self.dataset)}")

            # Generate all N responses with temperature
            all_generations = self._generate_all_responses(data['prompt'], gen_id)

            # Save cache after each generation
            with open(self.raw_results_path, 'w') as f:
                json.dump(self.all_results, f, indent=2)

            # Check if we have valid generations
            valid_generations = [g for g in all_generations if not self.ignore_generation(g)]
            if len(valid_generations) < self.args.num_generations_per_prompt / 2:
                print(f"Warning: Too many invalid generations for example {gen_id}, skipping")
                continue

            # Build sequence dictionary
            # For FACTS: use subject (truncated user_request); for PopQA: use question or entity
            entity_name = data.get('subj', data.get('question', data.get('entity', 'unknown')))

            sequence_dict = {
                'prompt': data['prompt'],
                'entity': entity_name,
                'wiki_title': data.get('wiki_title', entity_name),
                'all_generations': all_generations,  # All N responses (no distinction between greedy/stochastic)
            }

            # Add WikiData URI if available (for PopQA dataset)
            if 's_uri' in data:
                sequence_dict['s_uri'] = data['s_uri']

            # Add subject name if available
            if 'subj' in data:
                sequence_dict['subject'] = data['subj']

            # Add ground truth answer if available
            if 'obj' in data:
                sequence_dict['ground_truth'] = data['obj']

            if 'possible_answers' in data:
                sequence_dict['possible_answers'] = data['possible_answers']

            sequences.append(sequence_dict)

        # Save collected results
        with open(self.collected_results_path, 'w') as outfile:
            json.dump(sequences, outfile, indent=4)

        print(f"\nGeneration complete. Saved {len(sequences)} sequences to {self.collected_results_path}")
        return sequences

    def ignore_generation(self, text):
        """
        Check if a generation should be ignored (refusal responses).

        Args:
            text: Generated text

        Returns:
            Boolean indicating if the generation should be ignored
        """
        if not text or not isinstance(text, str):
            return True

        text_lower = text.lower()
        refusal_phrases = [
            'i apologize',
            "i don't know",
            "i'm not sure",
            "i'm sorry",
            "i'm not familiar",
            'unfortunately',
            'i cannot',
            'i am not able',
        ]

        return any(phrase in text_lower for phrase in refusal_phrases)
