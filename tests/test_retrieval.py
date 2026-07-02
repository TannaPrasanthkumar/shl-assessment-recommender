"""
Unit and Integration tests for the SHL Catalog Hybrid Retrieval system.
Validates BM25 indexing, FAISS dense vector search, Rank Fusion (RRF),
metadata filtering, and Cross-Encoder reranking operations.
"""

import pytest
import os
from src.models.schemas import CatalogItem, ConstraintState
from src.database.vector_store import SHLCatalogRepository
from src.database.retrieval import (
    BM25Retriever,
    DenseRetriever,
    CrossEncoderRanker,
    HybridRetriever,
    simple_tokenize
)

# Mock dataset representing catalog entries for testing
MOCK_CATALOG = [
    CatalogItem(
        entity_id="c1",
        name="Java Developer Assessment",
        link="https://www.shl.com/java",
        description="Measures Java coding skills, OOP principles, concurrency, and debugging.",
        job_levels=["Mid-Professional"],
        languages=["English"],
        duration="30 minutes",
        keys=["Knowledge & Skills"],
        competencies=["problem solving"],
        skills=["Java", "OOP"],
        test_type="K"
    ),
    CatalogItem(
        entity_id="c2",
        name="Occupational Personality Questionnaire OPQ32r",
        link="https://www.shl.com/opq32r",
        description="Measures behavioral styles, teamwork, leadership style, and decision-making preferences.",
        job_levels=["Director", "Executive"],
        languages=["English"],
        duration="25 minutes",
        keys=["Personality & Behavior"],
        competencies=["teamwork", "leadership"],
        skills=[],
        test_type="P"
    ),
    CatalogItem(
        entity_id="c3",
        name="Cognitive Ability Test G+",
        link="https://www.shl.com/gplus",
        description="Measures abstract reasoning, logical thinking, and numerical calculation.",
        job_levels=["Entry-Level", "Graduate"],
        languages=["English", "Spanish"],
        duration="35 minutes",
        keys=["Ability & Aptitude"],
        competencies=["problem solving"],
        skills=["Reasoning", "Math"],
        test_type="A"
    )
]


def test_simple_tokenize() -> None:
    """
    Verifies tokenizer extracts cleaned alphanumeric strings.
    """
    text = "Java, Developer; assessment -- (New)!"
    assert simple_tokenize(text) == ["java", "developer", "assessment", "new"]


def test_bm25_retriever_search() -> None:
    """
    Verifies BM25 lexical match scoring.
    """
    retriever = BM25Retriever()
    retriever.build_index(MOCK_CATALOG)
    
    # Searching for exact keyword
    results = retriever.search("Java coding")
    assert len(results) >= 1
    # Top result should be the Java assessment
    assert results[0][0].entity_id == "c1"


def test_dense_retriever_search() -> None:
    """
    Verifies Dense vector search matches semantic concepts.
    """
    retriever = DenseRetriever()
    retriever.build_index(MOCK_CATALOG)
    
    # Search with similar semantic query
    results = retriever.search("reasoning logic")
    assert len(results) >= 1
    # Should resolve to Cognitive Ability G+ (L2 index or fallback)
    assert any(item.entity_id == "c3" for item, _ in results)


def test_cross_encoder_reranker() -> None:
    """
    Verifies Cross-Encoder score generation and ranking.
    """
    ranker = CrossEncoderRanker()
    # Check that rerank runs and orders by score
    results = ranker.rerank("hiring leader for senior role", MOCK_CATALOG)
    assert len(results) == len(MOCK_CATALOG)
    assert results[0][1] >= results[-1][1]


def test_hybrid_retriever_pipeline() -> None:
    """
    Verifies the hybrid retriever pipeline, including RRF, filters, and boosting.
    """
    retriever = HybridRetriever(MOCK_CATALOG)
    
    # Case 1: Search with no constraints
    constraints = ConstraintState()
    candidates, confidence = retriever.retrieve("OOP Java Developer", constraints)
    
    assert len(candidates) >= 1
    assert candidates[0].entity_id == "c1"
    assert 0.0 <= confidence <= 1.0

    # Case 2: Filter by test type preference 'P' (Personality)
    constraints_p = ConstraintState(test_type_preferences=["P"])
    candidates_p, confidence_p = retriever.retrieve("Java Developer", constraints_p)
    # Java (type K) should be filtered out, leaving OPQ (type P)
    assert len(candidates_p) == 1
    assert candidates_p[0].entity_id == "c2"


def test_repository_search_integration() -> None:
    """
    Validates SHLCatalogRepository load and search integration.
    """
    import asyncio
    clean_cat_path = "data/clean_catalog.json"
    
    if os.path.exists(clean_cat_path):
        repo = SHLCatalogRepository(
            catalog_path=clean_cat_path,
            vector_db_path="data/vector_db"
        )
        
        async def run_test():
            await repo.load_catalog()
            constraints = ConstraintState(job_role="Java Developer")
            results = await repo.hybrid_search(
                query="Java developer assessment",
                constraints=constraints,
                limit=10
            )
            return results
            
        results = asyncio.run(run_test())
        
        assert isinstance(results, list)
        if results:
            assert len(results) <= 10
            # Test type properties should exist
            assert hasattr(results[0], "test_type")
            assert hasattr(results[0], "competencies")
            assert hasattr(results[0], "skills")
