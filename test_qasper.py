"""
Test Suite for QASPER Data and Experiment Modules

Tests all components of the QASPER experiment pipeline:
1. Section identification
2. Prior construction
3. Question classification
4. Experiment pipeline with mock components
"""

import sys

# Test result tracking
test_results = []


def log_test(name: str, passed: bool, message: str = ""):
    """Log a test result."""
    status = "✓" if passed else "✗"
    test_results.append((name, passed, message))
    print(f"  {status} {name}" + (f": {message}" if message else ""))


# ==============================================================================
# TEST DATA
# ==============================================================================

MOCK_PAPER = {
    "id": "test_paper_001",
    "title": "A Novel Approach to Neural Machine Translation",
    "abstract": "We present a new method for neural machine translation that improves upon existing approaches. Our method achieves state-of-the-art results on multiple benchmarks.",
    "full_text": {
        "section_name": [
            "Introduction",
            "Related Work",
            "Methodology",
            "Experiments",
            "Results",
            "Conclusion"
        ],
        "paragraphs": [
            ["Machine translation has seen significant advances in recent years.",
             "We propose a novel attention mechanism for improved translation quality."],
            ["Previous work on neural machine translation includes the transformer architecture.",
             "Attention mechanisms have been widely studied."],
            ["Our approach uses a modified attention mechanism.",
             "We introduce a new loss function for training."],
            ["We evaluate our method on WMT datasets.",
             "We compare against strong baselines."],
            ["Our method achieves 32.5 BLEU on WMT14 En-De.",
             "This represents a 2.1 point improvement."],
            ["We have presented a novel approach to neural machine translation.",
             "Future work will explore multilingual settings."]
        ]
    },
    "qas": [
        {
            "question_id": "q1",
            "question": "What is the main contribution of this paper?",
            "answers": [{
                "unanswerable": False,
                "extractive_spans": [],
                "yes_no": None,
                "free_form_answer": "A novel attention mechanism for machine translation",
                "evidence": ["We propose a novel attention mechanism for improved translation quality."],
                "highlighted_evidence": []
            }]
        },
        {
            "question_id": "q2",
            "question": "What BLEU score does the method achieve?",
            "answers": [{
                "unanswerable": False,
                "extractive_spans": ["32.5 BLEU"],
                "yes_no": None,
                "free_form_answer": "",
                "evidence": ["Our method achieves 32.5 BLEU on WMT14 En-De."],
                "highlighted_evidence": []
            }]
        },
        {
            "question_id": "q3",
            "question": "Is this about image classification?",
            "answers": [{
                "unanswerable": False,
                "extractive_spans": [],
                "yes_no": False,
                "free_form_answer": "",
                "evidence": [],
                "highlighted_evidence": []
            }]
        },
        {
            "question_id": "q4",
            "question": "What datasets are not included in this paper?",
            "answers": [{
                "unanswerable": True,
                "extractive_spans": [],
                "yes_no": None,
                "free_form_answer": "",
                "evidence": [],
                "highlighted_evidence": []
            }]
        }
    ]
}

MOCK_PAPER_ALTERNATE_SECTIONS = {
    "id": "test_paper_002",
    "title": "Test Paper with Alternate Section Names",
    "abstract": "Abstract text here.",
    "full_text": {
        "section_name": [
            "1 Intro",
            "Background",
            "Our Approach",
            "Evaluation",
            "Discussion"
        ],
        "paragraphs": [
            ["Introduction paragraph."],
            ["Background paragraph."],
            ["Approach paragraph."],
            ["Evaluation paragraph."],
            ["Discussion paragraph."]
        ]
    },
    "qas": []
}


# ==============================================================================
# TESTS
# ==============================================================================

def test_section_identification():
    """Test section name variant matching."""
    print("\n[1/6] Testing section identification...")

    from src.qasper_data import (
        identify_section_index,
        INTRODUCTION_VARIANTS,
        METHODS_VARIANTS,
        EXPERIMENT_VARIANTS
    )

    # Test standard names
    section_names = MOCK_PAPER["full_text"]["section_name"]

    intro_idx = identify_section_index(section_names, INTRODUCTION_VARIANTS)
    log_test("Find Introduction", intro_idx == 0)

    methods_idx = identify_section_index(section_names, METHODS_VARIANTS)
    log_test("Find Methodology", methods_idx == 2)

    exp_idx = identify_section_index(section_names, EXPERIMENT_VARIANTS)
    log_test("Find Experiments", exp_idx == 3)

    # Test alternate names
    alt_sections = MOCK_PAPER_ALTERNATE_SECTIONS["full_text"]["section_name"]

    alt_intro = identify_section_index(alt_sections, INTRODUCTION_VARIANTS)
    log_test("Find '1 Intro' variant", alt_intro == 0)

    alt_methods = identify_section_index(alt_sections, METHODS_VARIANTS)
    log_test("Find 'Our Approach' variant", alt_methods == 2)

    # Test missing section
    no_methods = ["Abstract", "Introduction", "Conclusion"]
    missing = identify_section_index(no_methods, METHODS_VARIANTS)
    log_test("Handle missing section", missing is None)


def test_prior_construction():
    """Test prior construction for each iteration."""
    print("\n[2/6] Testing prior construction...")

    from src.qasper_data import construct_prior, construct_expert_context

    # Iteration 0: Title + Abstract only
    prior_0 = construct_prior(MOCK_PAPER, iteration=0)
    has_title = "Neural Machine Translation" in prior_0.text
    has_abstract = "state-of-the-art" in prior_0.text
    no_intro = "significant advances" not in prior_0.text
    log_test("P_0 has title", has_title)
    log_test("P_0 has abstract", has_abstract)
    log_test("P_0 excludes intro", no_intro)
    log_test("P_0 sections correct", prior_0.sections_included == ["title", "abstract"])

    # Iteration 1: Title + Abstract + Introduction
    prior_1 = construct_prior(MOCK_PAPER, iteration=1)
    has_intro = "significant advances" in prior_1.text
    log_test("P_1 includes introduction", has_intro)
    log_test("P_1 sections include intro", "Introduction" in prior_1.sections_included)

    # Iteration 2: Title + Abstract + Introduction + Methods
    prior_2 = construct_prior(MOCK_PAPER, iteration=2)
    has_methods = "modified attention mechanism" in prior_2.text
    log_test("P_2 includes methodology", has_methods)

    # Token count estimation
    log_test("Token count positive", prior_2.token_count > 0)
    log_test("Token count increases", prior_2.token_count > prior_0.token_count)

    # Expert context (full paper)
    expert_ctx = construct_expert_context(MOCK_PAPER)
    has_results = "32.5 BLEU" in expert_ctx
    log_test("Expert context has full paper", has_results)


def test_question_classification():
    """Test question answerability classification."""
    print("\n[3/6] Testing question classification...")

    from src.qasper_data import (
        classify_question_answerability,
        construct_prior,
        QuestionClassification
    )

    questions = MOCK_PAPER["qas"]

    # Test with P_0 (title + abstract only)
    prior_0 = construct_prior(MOCK_PAPER, iteration=0)

    # Q1: Free-form answer, evidence in intro
    q1_class = classify_question_answerability(
        questions[0], prior_0.sections_included, MOCK_PAPER
    )
    log_test("Q1 not unanswerable", not q1_class.is_unanswerable)
    log_test("Q1 is free_form type", q1_class.answer_type == "free_form")

    # Q2: Extractive answer
    q2_class = classify_question_answerability(
        questions[1], prior_0.sections_included, MOCK_PAPER
    )
    log_test("Q2 is extractive type", q2_class.answer_type == "extractive")
    log_test("Q2 has ground truth", q2_class.ground_truth_answer == "32.5 BLEU")

    # Q3: Yes/No answer
    q3_class = classify_question_answerability(
        questions[2], prior_0.sections_included, MOCK_PAPER
    )
    log_test("Q3 is yes_no type", q3_class.answer_type == "yes_no")
    log_test("Q3 ground truth is No", q3_class.ground_truth_answer == "No")

    # Q4: Unanswerable
    q4_class = classify_question_answerability(
        questions[3], prior_0.sections_included, MOCK_PAPER
    )
    log_test("Q4 is unanswerable", q4_class.is_unanswerable)
    log_test("Q4 type is unanswerable", q4_class.answer_type == "unanswerable")

    # Test serialization
    q1_dict = q1_class.to_dict()
    log_test("Classification serializable", "question_id" in q1_dict)


def test_paper_validation():
    """Test paper structure validation."""
    print("\n[4/6] Testing paper validation...")

    from src.qasper_data import (
        validate_paper_structure,
        filter_papers_with_complete_structure,
        get_papers_with_answerable_questions
    )

    # Validate structure
    structure = validate_paper_structure(MOCK_PAPER)
    log_test("Structure has paper_id", structure.paper_id == "test_paper_001")
    log_test("Structure has 6 sections", structure.num_sections == 6)
    log_test("Structure has_intro", structure.has_intro)
    log_test("Structure has_methods", structure.has_methods)
    log_test("Structure has 4 questions", structure.num_questions == 4)
    log_test("Structure has 3 answerable", structure.answerable_questions == 3)

    # Filter papers
    papers = [MOCK_PAPER, MOCK_PAPER_ALTERNATE_SECTIONS]
    filtered = filter_papers_with_complete_structure(papers)
    log_test("Both papers pass filter", len(filtered) == 2)

    # Filter by answerable questions
    answerable_filtered = get_papers_with_answerable_questions(papers, min_answerable=1)
    log_test("Filter by answerable questions", len(answerable_filtered) >= 1)


def test_experiment_mock():
    """Test experiment with mock components."""
    print("\n[5/6] Testing experiment with mock components...")

    from src.qasper_experiment import (
        QASPERExperiment,
        PaperExperimentResult,
        IterationMetrics
    )
    from src.expert_verification import MockExpertVerifier

    # Create mock verifier
    mock_verifier = MockExpertVerifier(default_verdict=True)

    # Create experiment with mock components
    experiment = QASPERExperiment(
        generator_model=None,  # Mock
        claim_extractor=None,  # Mock
        expert_verifier=mock_verifier,
        output_dir=None
    )
    log_test("Experiment created", experiment is not None)

    # Run on mock paper
    result = experiment.run_single_paper_experiment(
        paper=MOCK_PAPER,
        num_iterations=3,
        claims_per_iteration=5,
        response_samples_per_question=2
    )

    log_test("Result has paper_id", result.paper_id == "test_paper_001")
    log_test("Result has 3 iterations", len(result.iterations) == 3)
    log_test("Result has responses", len(result.responses) > 0)
    log_test("No error", result.error is None)

    # Check iteration results
    it0 = result.iterations[0]
    log_test("Iteration 0 has metrics", it0.metrics is not None)
    log_test("Metrics has PCR", 0 <= it0.metrics.prior_coverage_ratio <= 1)
    log_test("Metrics has KF size", it0.metrics.knowledge_frontier_size >= 0)

    # Check serialization
    result_dict = result.to_dict()
    log_test("Result serializable", "paper_id" in result_dict)
    log_test("Iterations serializable", len(result_dict["iterations"]) == 3)


def test_metrics_computation():
    """Test metrics computation logic."""
    print("\n[6/6] Testing metrics computation...")

    from src.qasper_experiment import IterationMetrics

    # Create test metrics
    metrics = IterationMetrics(
        prior_coverage_ratio=0.6,
        knowledge_frontier_size=10,
        question_answerability_rate=0.5,
        verification_utility_ratio=0.8,
        total_claims=25,
        grounded_claims=15,
        contested_claims=10
    )

    log_test("PCR correct", metrics.prior_coverage_ratio == 0.6)
    log_test("KF size correct", metrics.knowledge_frontier_size == 10)
    log_test("Grounded + contested = total",
             metrics.grounded_claims + metrics.contested_claims == metrics.total_claims)

    # Test serialization
    metrics_dict = metrics.to_dict()
    log_test("Metrics serializable", "prior_coverage_ratio" in metrics_dict)
    log_test("Dict has all keys", len(metrics_dict) == 7)


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING QASPER DATA AND EXPERIMENT MODULES")
    print("=" * 70)

    try:
        test_section_identification()
        test_prior_construction()
        test_question_classification()
        test_paper_validation()
        test_experiment_mock()
        test_metrics_computation()

        # Summary
        print("\n" + "=" * 70)
        passed = sum(1 for _, p, _ in test_results if p)
        total = len(test_results)

        if passed == total:
            print(f"ALL TESTS PASSED! ({passed}/{total})")
        else:
            print(f"TESTS: {passed}/{total} passed")
            print("\nFailed tests:")
            for name, p, msg in test_results:
                if not p:
                    print(f"  ✗ {name}: {msg}")
            sys.exit(1)

        print("=" * 70)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
