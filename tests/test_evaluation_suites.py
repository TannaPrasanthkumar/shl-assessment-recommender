"""
Integration tests executing the conversation simulator, prompt injection probes,
schema validation compliance checks, and latency audits under pytest.
"""

import pytest
import asyncio
import os
import time
from src.agents.decision_engine import DecisionEngine
from src.agents.state_tracker import DialogueStateTracker
from src.database.vector_store import SHLCatalogRepository
from src.models.schemas import Message, ChatResponse, ChatRequest
from tests.evaluator_simulator import SHLEvaluator, INJECTION_PROBES


@pytest.fixture
def evaluator() -> SHLEvaluator:
    """
    Initializes evaluator with workspace paths.
    """
    convs_dir = os.path.abspath(os.path.join("data", "GenAI_SampleConversations"))
    catalog_path = os.path.abspath(os.path.join("data", "clean_catalog.json"))
    ev = SHLEvaluator(convs_dir, catalog_path)
    asyncio.run(ev.initialize())
    return ev


def test_schema_validations() -> None:
    """
    Verifies API schema serialization and model field types.
    """
    # ChatRequest validations
    req = ChatRequest(messages=[Message(role="user", content="Test message")])
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"

    # ChatResponse validations
    resp = ChatResponse(reply="Test reply", recommendations=[], end_of_conversation=False)
    assert resp.reply == "Test reply"
    assert resp.end_of_conversation is False


def test_prompt_injection_safety_rate(evaluator: SHLEvaluator) -> None:
    """
    Asserts that 100% of prompt injection and jailbreak probes are rejected/refused.
    """
    results = asyncio.run(evaluator.evaluate_safety_probes())
    assert results["safety_rate"] == 100.0
    assert results["refused_count"] == results["total_probes"]


def test_response_generation_latency_bounds(evaluator: SHLEvaluator) -> None:
    """
    Asserts that response generation latency per turn is strictly under 2.0 seconds.
    """
    history = [Message(role="user", content="Looking for a Python developer test.")]
    
    start_time = time.time()
    constraints = asyncio.run(evaluator.state_tracker.extract_constraints(history))
    candidates, confidence = evaluator.repository.retriever.retrieve(
        "Looking for a Python developer test.",
        constraints,
        limit=5
    )
    action = evaluator.decision_engine.evaluate_routing_action(
        history_len=1,
        constraints=constraints,
        retrieval_confidence=confidence,
        latest_query="Looking for a Python developer test."
    )
    response = asyncio.run(evaluator.decision_engine.execute_action(
        action=action,
        constraints=constraints,
        raw_query="Looking for a Python developer test."
    ))
    duration = time.time() - start_time
    
    assert duration < 2.0  # Must resolve in under 2.0 seconds
    assert isinstance(response, ChatResponse)


def test_conversation_replay_simulator_metrics(evaluator: SHLEvaluator) -> None:
    """
    Verifies that the conversation replay simulator runs successfully over trace MD logs.
    """
    # Replay trace C1
    filepath = os.path.join(evaluator.conversations_dir, "C1.md")
    turns = evaluator._parse_conversation_file(filepath)
    assert len(turns) > 0

    results = asyncio.run(evaluator.replay_conversation("C1.md", turns))
    assert results["turns"] == len(turns)
    assert results["schema_compliant"] is True
    assert isinstance(results["recall"], float)


def test_modular_evaluator_subclasses(evaluator: SHLEvaluator) -> None:
    """
    Verifies modular evaluator helper classes and subclasses.
    """
    from tests.evaluator_simulator import (
        RecallCalculator,
        HallucinationDetector,
        SchemaComplianceTester,
        PerformanceBenchmark,
        EvaluationReporter,
        SyntheticConversationGenerator,
        ConversationReplayEngine,
        EvaluatorSimulator,
        BehaviorProbeRunner,
        RegressionSuite
    )
    from src.models.schemas import ChatResponse, Recommendation

    # Test RecallCalculator
    calc = RecallCalculator()
    assert calc.calculate_recall(["OPQ32r"], ["OPQ32r"]) == 1.0
    assert calc.calculate_recall(["OPQ32r"], ["Verify G+"]) == 0.0

    # Test HallucinationDetector
    detector = HallucinationDetector(evaluator.repository)
    recs = [Recommendation(name="Non-existent Test", url="http://invalid", test_type="K")]
    halls = detector.detect_hallucinations(recs)
    assert len(halls) > 0

    # Test SchemaComplianceTester
    tester = SchemaComplianceTester()
    assert tester.test_schema(ChatResponse(reply="test", recommendations=[])) is True

    # Test PerformanceBenchmark
    bench = PerformanceBenchmark()
    bench.record(0.1)
    bench.record(0.2)
    stats = bench.stats()
    assert stats["avg"] == pytest.approx(150.0)

    # Test EvaluationReporter
    reporter = EvaluationReporter()
    report = reporter.build_markdown_report([], {"safety_rate": 100.0, "total_probes": 1, "refused_count": 1}, stats)
    assert "System Evaluation & Simulator Report" in report

    # Test SyntheticConversationGenerator
    gen = SyntheticConversationGenerator()
    turn = gen.generate_synthetic_turn("user", "test query")
    assert turn["user"] == "test query"

    # Test Subclasses instantiation
    engine = ConversationReplayEngine(evaluator.conversations_dir, evaluator.catalog_path)
    sim = EvaluatorSimulator(evaluator.conversations_dir, evaluator.catalog_path)
    probe = BehaviorProbeRunner(evaluator.conversations_dir, evaluator.catalog_path)
    reg = RegressionSuite(evaluator.conversations_dir, evaluator.catalog_path)
    assert engine is not None
    assert sim is not None
    assert probe is not None
    assert reg is not None


def test_quality_analyzer_execution(evaluator: SHLEvaluator) -> None:
    """
    Verifies execution of the self-evaluation and quality benchmarking modules.
    """
    from tests.quality_analyzer import (
        RetrievalAnalyzer,
        ConversationAnalyzer,
        RecommendationAnalyzer,
        FailureAnalyzer,
        PerformanceAnalyzer,
        EngineeringScorecard,
        ImprovementEngine,
        BenchmarkRunner,
        QualityAnalyzer
    )

    # Instantiate
    ret_analyzer = RetrievalAnalyzer()
    conv_analyzer = ConversationAnalyzer()
    rec_analyzer = RecommendationAnalyzer()
    fail_analyzer = FailureAnalyzer()
    perf_analyzer = PerformanceAnalyzer()
    scorecard = EngineeringScorecard()
    engine = ImprovementEngine()
    bench = BenchmarkRunner()

    # Stub dataset
    results = [
        {"file": "C1.md", "turns": 4, "recall": 0.90, "avg_latency_ms": 150.0, "schema_compliant": True, "end_of_conversation": True}
    ]

    ret_report = ret_analyzer.analyze_retrieval(results)
    assert ret_report["mean_recall"] == 0.90
    assert ret_report["status"] == "excellent"

    conv_report = conv_analyzer.analyze_conversation(results)
    assert conv_report["avg_turns"] == 4.0

    rec_report = rec_analyzer.analyze_recommendations(results)
    assert rec_report["schema_compliance_rate"] == 100.0

    failures = fail_analyzer.classify_failures(results)
    assert len(failures) == 0

    perf_report = perf_analyzer.analyze_performance(results)
    assert perf_report["avg_latency_ms"] == 150.0

    card = scorecard.compute_scores(ret_report, conv_report, rec_report, perf_report)
    assert card["overall_engineering_score"] > 80

    suggestions = engine.generate_suggestions(card)
    assert isinstance(suggestions, list)

    benchmarks = bench.run_benchmarks()
    assert "configurations" in benchmarks

    # Run top-level audit in temp folder
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = QualityAnalyzer(evaluator)
        card_run = asyncio.run(analyzer.run_quality_audit(temp_dir))
        assert "overall_engineering_score" in card_run
