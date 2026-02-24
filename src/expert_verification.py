"""
Expert Verification Module

Provides interfaces and implementations for claim verification by experts
(LLM or human). This module is designed to be abstract and extensible.

Key components:
1. VerificationResult - Data class for verification outcomes
2. BaseExpertVerifier - Abstract base class for verifiers
3. LLMExpertVerifier - GPT-based verifier implementation
4. HumanExpertVerifier - Placeholder for human verification interface
5. MockExpertVerifier - For testing without API calls
"""

import json
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable, Any

# Try to import openai, but don't fail if not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class VerificationResult:
    """
    Result of verifying a single claim.

    Attributes:
        claim_id: Unique identifier for the claim
        claim_text: The text content of the claim
        verdict: True if accepted, False if rejected
        reasoning: Explanation for the decision
        confidence: Optional confidence score (0-1)
        raw_response: Full response data for logging
        timestamp: ISO format timestamp of verification
        verifier_type: Type of verifier used (e.g., 'llm', 'human')
        metadata: Additional metadata
    """
    claim_id: str
    claim_text: str
    verdict: bool
    reasoning: str
    confidence: Optional[float] = None
    raw_response: Optional[dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    verifier_type: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'claim_id': self.claim_id,
            'claim_text': self.claim_text,
            'verdict': self.verdict,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'raw_response': self.raw_response,
            'timestamp': self.timestamp,
            'verifier_type': self.verifier_type,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VerificationResult':
        """Create from dictionary."""
        return cls(**data)


# ==============================================================================
# VERIFICATION PROMPTS
# ==============================================================================

VERIFICATION_SYSTEM_PROMPT = """You are an expert fact-checker. Your task is to verify whether a claim is supported by the provided context.

CRITICAL RULES:
1. You must ONLY use information from the provided context to make your decision.
2. Do NOT use any prior knowledge or information not explicitly stated in the context.
3. If the context does not contain sufficient information to verify the claim, you must REJECT it.
4. Be strict: partial support or ambiguous evidence should result in REJECT.

Your response must be in the following JSON format:
{
    "decision": "ACCEPT" or "REJECT",
    "confidence": <float between 0 and 1>,
    "reasoning": "Brief explanation (1-2 sentences)"
}

Decision criteria:
- ACCEPT: The claim is directly and clearly supported by the context.
- REJECT: The claim is contradicted by the context, not mentioned, only partially supported, or ambiguous."""


VERIFICATION_USER_PROMPT_TEMPLATE = """Context:
{context}

---

Claim to verify:
"{claim_text}"

---

Based ONLY on the context above, verify whether this claim should be accepted or rejected.
Respond in JSON format with "decision", "confidence", and "reasoning" fields."""


# ==============================================================================
# BASE VERIFIER INTERFACE
# ==============================================================================

class BaseExpertVerifier(ABC):
    """
    Abstract base class for expert verifiers.

    Subclass this to implement different verification backends:
    - LLM-based verification (GPT-4, Claude, etc.)
    - Human verification interface
    - Rule-based verification
    - Retrieval-based verification
    """

    @abstractmethod
    def verify_claim(
        self,
        claim_id: str,
        claim_text: str,
        context: str
    ) -> VerificationResult:
        """
        Verify a single claim against the provided context.

        Args:
            claim_id: Unique identifier for the claim
            claim_text: The claim text to verify
            context: Context document(s) for verification

        Returns:
            VerificationResult with verdict and reasoning
        """
        pass

    def verify_batch(
        self,
        claims: List[Dict[str, str]],
        context: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[VerificationResult]:
        """
        Verify a batch of claims with shared context.

        Args:
            claims: List of dicts with 'claim_id' and 'claim_text' keys
            context: Context document (shared across all claims)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of VerificationResult objects
        """
        results = []
        total = len(claims)

        for i, claim in enumerate(claims):
            result = self.verify_claim(
                claim_id=claim['claim_id'],
                claim_text=claim['claim_text'],
                context=context
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def verify_batch_individual_context(
        self,
        claims_with_context: List[Dict[str, str]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[VerificationResult]:
        """
        Verify claims where each has its own context.

        Args:
            claims_with_context: List of dicts with 'claim_id', 'claim_text', 'context'
            progress_callback: Optional callback(current, total)

        Returns:
            List of VerificationResult objects
        """
        results = []
        total = len(claims_with_context)

        for i, item in enumerate(claims_with_context):
            result = self.verify_claim(
                claim_id=item['claim_id'],
                claim_text=item['claim_text'],
                context=item['context']
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results


# ==============================================================================
# LLM EXPERT VERIFIER
# ==============================================================================

class LLMExpertVerifier(BaseExpertVerifier):
    """
    LLM-based expert verifier using OpenAI API (GPT-4 or similar).

    This verifier uses a language model to judge whether claims are
    supported by the provided context, following strict verification rules.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.0,
        requests_per_minute: int = 50,
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None
    ):
        """
        Initialize the LLM verifier.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model identifier (e.g., 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo')
            temperature: Sampling temperature (0.0 for deterministic)
            requests_per_minute: Rate limit for API calls
            max_retries: Number of retries on failure
            system_prompt: Custom system prompt (uses default if None)
            user_prompt_template: Custom user prompt template (uses default if None)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for LLMExpertVerifier")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.request_interval = 60.0 / requests_per_minute
        self.max_retries = max_retries
        self.last_request_time = 0.0

        self.system_prompt = system_prompt or VERIFICATION_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or VERIFICATION_USER_PROMPT_TEMPLATE

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()

    def _parse_response(self, response_text: str) -> Tuple[bool, float, str]:
        """
        Parse LLM response into verdict, confidence, and reasoning.

        Args:
            response_text: JSON response from LLM

        Returns:
            Tuple of (verdict, confidence, reasoning)

        Raises:
            ValueError: If response cannot be parsed
        """
        parsed = json.loads(response_text)

        decision = parsed.get("decision", "").upper()
        if decision not in ["ACCEPT", "REJECT"]:
            raise ValueError(f"Invalid decision: {decision}")

        verdict = (decision == "ACCEPT")
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "")

        return verdict, confidence, reasoning

    def verify_claim(
        self,
        claim_id: str,
        claim_text: str,
        context: str
    ) -> VerificationResult:
        """
        Verify a single claim using the LLM.

        Args:
            claim_id: Unique identifier for the claim
            claim_text: The claim text to verify
            context: Context document for verification

        Returns:
            VerificationResult with verdict and reasoning
        """
        self._rate_limit_wait()

        user_prompt = self.user_prompt_template.format(
            context=context,
            claim_text=claim_text
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )

                response_text = response.choices[0].message.content
                verdict, confidence, reasoning = self._parse_response(response_text)

                raw_response = {
                    "model": self.model,
                    "response_text": response_text,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens
                    }
                }

                return VerificationResult(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    verdict=verdict,
                    reasoning=reasoning,
                    confidence=confidence,
                    raw_response=raw_response,
                    verifier_type="llm",
                    metadata={"model": self.model, "attempt": attempt + 1}
                )

            except json.JSONDecodeError as e:
                if attempt == self.max_retries - 1:
                    return VerificationResult(
                        claim_id=claim_id,
                        claim_text=claim_text,
                        verdict=False,
                        reasoning=f"Failed to parse LLM response: {str(e)}",
                        confidence=0.0,
                        raw_response={"error": str(e)},
                        verifier_type="llm",
                        metadata={"model": self.model, "error": "parse_error"}
                    )
                continue

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return VerificationResult(
                        claim_id=claim_id,
                        claim_text=claim_text,
                        verdict=False,
                        reasoning=f"Verification failed: {str(e)}",
                        confidence=0.0,
                        raw_response={"error": str(e)},
                        verifier_type="llm",
                        metadata={"model": self.model, "error": "api_error"}
                    )
                continue

        # Should not reach here, but handle just in case
        return VerificationResult(
            claim_id=claim_id,
            claim_text=claim_text,
            verdict=False,
            reasoning="Max retries exceeded",
            confidence=0.0,
            verifier_type="llm",
            metadata={"model": self.model, "error": "max_retries"}
        )


# ==============================================================================
# HUMAN EXPERT VERIFIER (ABSTRACT INTERFACE)
# ==============================================================================

class HumanExpertVerifier(BaseExpertVerifier):
    """
    Abstract interface for human expert verification.

    This class provides a framework for integrating human verification.
    Subclass and implement the _get_human_verdict method for your specific
    UI/workflow (web interface, CLI prompts, labeling tool integration, etc.)
    """

    def __init__(self, verifier_id: str = "human_expert"):
        """
        Args:
            verifier_id: Identifier for the human verifier
        """
        self.verifier_id = verifier_id

    def _get_human_verdict(
        self,
        claim_id: str,
        claim_text: str,
        context: str
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Get verdict from human expert.

        Override this method in subclasses to implement actual human interaction.

        Args:
            claim_id: Claim identifier
            claim_text: Claim text
            context: Context for verification

        Returns:
            Tuple of (verdict, reasoning, optional_confidence)
        """
        raise NotImplementedError(
            "Subclass HumanExpertVerifier and implement _get_human_verdict "
            "for your specific UI/workflow"
        )

    def verify_claim(
        self,
        claim_id: str,
        claim_text: str,
        context: str
    ) -> VerificationResult:
        """Verify using human expert."""
        verdict, reasoning, confidence = self._get_human_verdict(
            claim_id, claim_text, context
        )

        return VerificationResult(
            claim_id=claim_id,
            claim_text=claim_text,
            verdict=verdict,
            reasoning=reasoning,
            confidence=confidence,
            verifier_type="human",
            metadata={"verifier_id": self.verifier_id}
        )


class CLIHumanVerifier(HumanExpertVerifier):
    """
    Simple CLI-based human verification for testing/demo purposes.

    Presents claims via command line and accepts y/n input.
    """

    def _get_human_verdict(
        self,
        claim_id: str,
        claim_text: str,
        context: str
    ) -> Tuple[bool, str, Optional[float]]:
        """Get verdict via CLI prompts."""
        print("\n" + "=" * 60)
        print("CLAIM VERIFICATION")
        print("=" * 60)
        print(f"\nClaim ID: {claim_id}")
        print(f"\nContext:\n{context[:500]}..." if len(context) > 500 else f"\nContext:\n{context}")
        print(f"\nClaim: {claim_text}")
        print("\n" + "-" * 60)

        while True:
            verdict_input = input("Accept this claim? (y/n): ").strip().lower()
            if verdict_input in ['y', 'yes']:
                verdict = True
                break
            elif verdict_input in ['n', 'no']:
                verdict = False
                break
            print("Please enter 'y' or 'n'")

        reasoning = input("Reasoning (optional, press Enter to skip): ").strip()
        if not reasoning:
            reasoning = "Human expert decision"

        return verdict, reasoning, None


# ==============================================================================
# MOCK VERIFIER FOR TESTING
# ==============================================================================

class MockExpertVerifier(BaseExpertVerifier):
    """
    Mock verifier for testing without API calls.

    Can be configured to return specific verdicts or use simple heuristics.
    """

    def __init__(
        self,
        default_verdict: bool = True,
        verdict_map: Optional[Dict[str, bool]] = None,
        delay: float = 0.0
    ):
        """
        Args:
            default_verdict: Default verdict if claim not in verdict_map
            verdict_map: Dict mapping claim_id -> verdict
            delay: Simulated delay per verification (seconds)
        """
        self.default_verdict = default_verdict
        self.verdict_map = verdict_map or {}
        self.delay = delay
        self.verification_count = 0

    def verify_claim(
        self,
        claim_id: str,
        claim_text: str,
        context: str
    ) -> VerificationResult:
        """Return mock verification result."""
        if self.delay > 0:
            time.sleep(self.delay)

        self.verification_count += 1

        verdict = self.verdict_map.get(claim_id, self.default_verdict)

        return VerificationResult(
            claim_id=claim_id,
            claim_text=claim_text,
            verdict=verdict,
            reasoning="Mock verification",
            confidence=1.0 if verdict else 0.0,
            verifier_type="mock",
            metadata={"verification_number": self.verification_count}
        )


# ==============================================================================
# CONTEXT PREPARATION UTILITIES
# ==============================================================================

def prepare_context_for_claim(
    claim_id: str,
    context_documents: Dict[str, str],
    claim_to_context_mapping: Optional[Dict[str, List[str]]] = None,
    max_context_length: int = 8000
) -> str:
    """
    Prepare context string for a specific claim.

    Args:
        claim_id: ID of the claim
        context_documents: Dict mapping doc_id -> document text
        claim_to_context_mapping: Optional dict mapping claim_id -> list of relevant doc_ids
        max_context_length: Maximum context length in characters

    Returns:
        Concatenated context string
    """
    if claim_to_context_mapping:
        relevant_doc_ids = claim_to_context_mapping.get(claim_id, [])
    else:
        relevant_doc_ids = []

    if not relevant_doc_ids:
        # If no mapping, use all documents
        relevant_doc_ids = list(context_documents.keys())

    context_parts = []
    total_length = 0

    for doc_id in relevant_doc_ids:
        doc_text = context_documents.get(doc_id, "")

        if total_length + len(doc_text) > max_context_length:
            # Truncate if necessary
            remaining = max_context_length - total_length
            if remaining > 100:  # Only add if meaningful length
                context_parts.append(f"[Document {doc_id}]:\n{doc_text[:remaining]}...")
            break

        context_parts.append(f"[Document {doc_id}]:\n{doc_text}")
        total_length += len(doc_text) + 50  # Account for header

    return "\n\n---\n\n".join(context_parts)


def prepare_batch_with_individual_context(
    claims: List[Dict[str, Any]],
    context_documents: Dict[str, str],
    claim_to_context_mapping: Dict[str, List[str]],
    max_context_length: int = 8000
) -> List[Dict[str, str]]:
    """
    Prepare claims with individual context for batch verification.

    Args:
        claims: List of claim dicts with 'claim_id' and 'claim_text'
        context_documents: Dict mapping doc_id -> document text
        claim_to_context_mapping: Dict mapping claim_id -> list of doc_ids
        max_context_length: Max context length per claim

    Returns:
        List of dicts with 'claim_id', 'claim_text', and 'context'
    """
    prepared = []

    for claim in claims:
        context = prepare_context_for_claim(
            claim_id=claim['claim_id'],
            context_documents=context_documents,
            claim_to_context_mapping=claim_to_context_mapping,
            max_context_length=max_context_length
        )

        prepared.append({
            'claim_id': claim['claim_id'],
            'claim_text': claim['claim_text'],
            'context': context
        })

    return prepared


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_verifier(
    verifier_type: str = "mock",
    **kwargs
) -> BaseExpertVerifier:
    """
    Factory function to create a verifier instance.

    Args:
        verifier_type: Type of verifier ('llm', 'mock', 'cli_human')
        **kwargs: Arguments passed to verifier constructor

    Returns:
        BaseExpertVerifier instance
    """
    if verifier_type == "llm":
        return LLMExpertVerifier(**kwargs)
    elif verifier_type == "mock":
        return MockExpertVerifier(**kwargs)
    elif verifier_type == "cli_human":
        return CLIHumanVerifier(**kwargs)
    else:
        raise ValueError(f"Unknown verifier type: {verifier_type}")


if __name__ == "__main__":
    # Test the mock verifier
    print("Testing Expert Verification Module...")

    # Test MockExpertVerifier
    mock_verifier = MockExpertVerifier(
        default_verdict=True,
        verdict_map={'C1': False, 'C3': False}
    )

    claims = [
        {'claim_id': 'C0', 'claim_text': 'The sky is blue.'},
        {'claim_id': 'C1', 'claim_text': 'The moon is made of cheese.'},
        {'claim_id': 'C2', 'claim_text': 'Water is wet.'},
        {'claim_id': 'C3', 'claim_text': 'Pigs can fly.'},
    ]

    context = "The sky appears blue due to Rayleigh scattering. Water is a liquid."

    results = mock_verifier.verify_batch(claims, context)

    print("\nVerification Results:")
    for r in results:
        print(f"  {r.claim_id}: {'ACCEPT' if r.verdict else 'REJECT'} - {r.claim_text}")

    # Test serialization
    print("\nSerialization test:")
    result_dict = results[0].to_dict()
    print(f"  Serialized: {result_dict['claim_id']} -> {result_dict['verdict']}")

    restored = VerificationResult.from_dict(result_dict)
    print(f"  Restored: {restored.claim_id} -> {restored.verdict}")

    print("\n✓ Expert Verification module working!")
