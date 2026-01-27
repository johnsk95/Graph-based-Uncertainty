"""
WikiData Prior Fetcher Module
Fetches structured knowledge from WikiData entities to serve as knowledge priors
for weighted bipartite graph construction.
"""

import requests
import time
from typing import List, Dict, Optional
import json


class WikiDataPriorFetcher:
    """Fetches claims from WikiData entities to use as knowledge priors."""

    def __init__(self, cache_dir: str = "data/wikidata_cache"):
        """
        Initialize WikiData fetcher with caching.

        Args:
            cache_dir: Directory to cache WikiData responses
        """
        self.cache_dir = cache_dir
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.entity_url = "https://www.wikidata.org/wiki/Special:EntityData/"

        # Create cache directory if it doesn't exist
        import os
        os.makedirs(cache_dir, exist_ok=True)

    def extract_entity_id(self, wikidata_uri: str) -> str:
        """
        Extract entity ID from WikiData URI.

        Args:
            wikidata_uri: URI like "http://www.wikidata.org/entity/Q123456"

        Returns:
            Entity ID like "Q123456"
        """
        return wikidata_uri.split('/')[-1]

    def get_cache_path(self, entity_id: str) -> str:
        """Get cache file path for an entity."""
        import os
        return os.path.join(self.cache_dir, f"{entity_id}.json")

    def fetch_entity_data(self, wikidata_uri: str) -> Optional[Dict]:
        """
        Fetch entity data from WikiData API with caching.

        Args:
            wikidata_uri: WikiData entity URI

        Returns:
            Dictionary containing entity data, or None if fetch fails
        """
        entity_id = self.extract_entity_id(wikidata_uri)
        cache_path = self.get_cache_path(entity_id)

        # Check cache first
        import os
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Fetch from API
        try:
            # Use the Wikidata API action endpoint (more reliable)
            params = {
                'action': 'wbgetentities',
                'ids': entity_id,
                'format': 'json',
                'languages': 'en'
            }

            headers = {
                'User-Agent': 'WikiDataPriorFetcher/1.0 (Educational Research Project)'
            }

            response = requests.get(self.base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            entity_data = response.json()

            # Cache the response
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(entity_data, f, ensure_ascii=False, indent=2)

            # Rate limiting
            time.sleep(0.1)

            return entity_data

        except Exception as e:
            print(f"Error fetching WikiData entity {entity_id}: {e}")
            return None

    def extract_claims_from_entity(self, entity_data: Dict, entity_id: str) -> List[str]:
        """
        Extract natural language claims from WikiData entity data.

        Args:
            entity_data: Raw WikiData entity data
            entity_id: Entity ID (e.g., "Q123456")

        Returns:
            List of natural language claims
        """
        if not entity_data or 'entities' not in entity_data:
            return []

        entity = entity_data['entities'].get(entity_id, {})
        claims_list = []

        # Get entity label (name)
        labels = entity.get('labels', {})
        entity_label = labels.get('en', {}).get('value', entity_id)

        # Get description
        descriptions = entity.get('descriptions', {})
        entity_description = descriptions.get('en', {}).get('value', None)

        if entity_description:
            claims_list.append(f"{entity_label} is {entity_description}.")

        # Extract claims from statements/claims
        claims = entity.get('claims', {})

        # Map of common property IDs to natural language
        property_mappings = {
            'P31': 'is a',  # instance of
            'P106': 'has occupation',  # occupation
            'P27': 'has country of citizenship',  # country of citizenship
            'P19': 'was born in',  # place of birth
            'P569': 'was born on',  # date of birth
            'P570': 'died on',  # date of death
            'P21': 'has sex or gender',  # sex or gender
            'P735': 'has given name',  # given name
            'P734': 'has family name',  # family name
            'P39': 'held position',  # position held
            'P108': 'worked for',  # employer
            'P69': 'was educated at',  # educated at
            'P166': 'received award',  # award received
            'P50': 'was written by',  # author
            'P800': 'created notable work',  # notable work
        }

        for prop_id, prop_claims in claims.items():
            prop_relation = property_mappings.get(prop_id, f"has property {prop_id}")

            for claim in prop_claims:
                try:
                    mainsnak = claim.get('mainsnak', {})
                    datatype = mainsnak.get('datatype', '')
                    datavalue = mainsnak.get('datavalue', {})

                    # Extract value based on datatype
                    value_str = None

                    if datatype == 'wikibase-item':
                        # Reference to another entity
                        value_id = datavalue.get('value', {}).get('id', '')
                        # For now, just use the ID; could be enhanced to fetch labels
                        value_str = value_id

                    elif datatype == 'string':
                        value_str = datavalue.get('value', '')

                    elif datatype == 'time':
                        time_value = datavalue.get('value', {}).get('time', '')
                        # Parse time format like "+1850-01-01T00:00:00Z"
                        if time_value:
                            # Extract year
                            value_str = time_value.split('-')[0].replace('+', '')

                    elif datatype == 'quantity':
                        amount = datavalue.get('value', {}).get('amount', '')
                        value_str = amount.replace('+', '')

                    if value_str:
                        claim_text = f"{entity_label} {prop_relation} {value_str}."
                        claims_list.append(claim_text)

                except Exception as e:
                    # Skip malformed claims
                    continue

        return claims_list

    def get_prior_claims(self, wikidata_uri: str, subject_name: str = None) -> Dict:
        """
        Get prior claims for a WikiData entity.

        Args:
            wikidata_uri: WikiData entity URI
            subject_name: Optional human-readable name for the subject

        Returns:
            Dictionary containing:
                - 'uri': WikiData URI
                - 'entity_id': Entity ID
                - 'subject': Subject name
                - 'claims': List of natural language claims
                - 'raw_data': Raw WikiData entity data (for reference)
        """
        entity_id = self.extract_entity_id(wikidata_uri)
        entity_data = self.fetch_entity_data(wikidata_uri)

        if not entity_data:
            return {
                'uri': wikidata_uri,
                'entity_id': entity_id,
                'subject': subject_name or entity_id,
                'claims': [],
                'raw_data': None
            }

        # Extract entity label if subject_name not provided
        if not subject_name:
            entity = entity_data.get('entities', {}).get(entity_id, {})
            labels = entity.get('labels', {})
            subject_name = labels.get('en', {}).get('value', entity_id)

        claims = self.extract_claims_from_entity(entity_data, entity_id)

        return {
            'uri': wikidata_uri,
            'entity_id': entity_id,
            'subject': subject_name,
            'claims': claims,
            'raw_data': entity_data
        }

    def get_batch_priors(self, wikidata_uris: List[str], subject_names: List[str] = None) -> List[Dict]:
        """
        Fetch prior claims for multiple WikiData entities.

        Args:
            wikidata_uris: List of WikiData URIs
            subject_names: Optional list of subject names (same order as URIs)

        Returns:
            List of prior dictionaries
        """
        if subject_names is None:
            subject_names = [None] * len(wikidata_uris)

        priors = []
        for uri, name in zip(wikidata_uris, subject_names):
            prior = self.get_prior_claims(uri, name)
            priors.append(prior)

        return priors


def test_wikidata_fetcher():
    """Test the WikiData fetcher with a sample entity."""
    fetcher = WikiDataPriorFetcher()

    # Test with Albert Einstein (Q937)
    test_uri = "http://www.wikidata.org/entity/Q937"

    print("Fetching WikiData prior for Albert Einstein...")
    prior = fetcher.get_prior_claims(test_uri, "Albert Einstein")

    print(f"\nEntity: {prior['subject']}")
    print(f"Entity ID: {prior['entity_id']}")
    print(f"Number of claims: {len(prior['claims'])}")
    print("\nSample claims:")
    for i, claim in enumerate(prior['claims'][:10]):
        print(f"  {i+1}. {claim}")


if __name__ == "__main__":
    test_wikidata_fetcher()
