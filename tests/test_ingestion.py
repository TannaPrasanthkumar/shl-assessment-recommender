"""
Unit tests for the catalog scraper and parser ingestion pipeline.
Asserts that network retries, job solution exclusions, and metadata normalizations function correctly.
"""

import pytest
from unittest.mock import patch, MagicMock
import urllib.error

from src.scraper.catalog_parser import CatalogScraper, CatalogParser


def test_is_job_solution() -> None:
    """
    Verifies that pre-packaged Job Solutions are identified and flagged.
    """
    parser = CatalogParser()
    
    # Positive cases (should flag as Job Solution)
    assert parser.is_job_solution({"name": "Software Developer Job Solution", "description": "Package for hiring"})
    assert parser.is_job_solution({"name": "Standard Test", "keys": ["Job Solutions"]})
    assert parser.is_job_solution({"name": "Pre-packaged Accounting", "description": "Prepackaged role"})
    
    # Negative cases (should NOT flag as Job Solution)
    assert not parser.is_job_solution({"name": "Occupational Personality Questionnaire OPQ32r", "keys": ["Personality & Behavior"]})
    assert not parser.is_job_solution({"name": "MS Excel (New)", "keys": ["Knowledge & Skills"]})


def test_map_keys_to_type() -> None:
    """
    Verifies key-to-type abbreviation mappings.
    """
    parser = CatalogParser()
    
    assert parser.map_keys_to_type(["Personality & Behavior"]) == "P"
    assert parser.map_keys_to_type(["Knowledge & Skills"]) == "K"
    assert parser.map_keys_to_type(["Simulations"]) == "S"
    assert parser.map_keys_to_type(["Competencies", "Knowledge & Skills"]) == "C, K"
    assert parser.map_keys_to_type(["Ability & Aptitude", "Simulations"]) == "A,S"
    assert parser.map_keys_to_type(["Ability & Aptitude", "Development & 360"]) == "D"  # Precedence rule


def test_normalize_duration() -> None:
    """
    Verifies text extraction and normalization of test durations.
    """
    parser = CatalogParser()
    
    assert parser.normalize_duration("30 minutes", "") == "30 minutes"
    assert parser.normalize_duration("", "Approximate Completion Time in minutes = 17") == "17 minutes"
    assert parser.normalize_duration("", "") == "Not Specified"


def test_clean_items_deduplication_and_filtering() -> None:
    """
    Verifies that parsing deduplicates matching IDs and URLs, and drops Job Solutions.
    """
    parser = CatalogParser()
    raw_data = [
        # Normal Item
        {
            "entity_id": "1001",
            "name": "General Cognitive Ability",
            "link": "https://www.shl.com/view/cog-ability/",
            "description": "Measures cognitive power.",
            "keys": ["Ability & Aptitude"],
            "duration": "20m",
            "adaptive": "yes",
            "remote": "yes",
            "languages": ["English"]
        },
        # Duplicate ID
        {
            "entity_id": "1001",
            "name": "General Cognitive Ability v2",
            "link": "https://www.shl.com/view/cog-ability-2/",
            "description": "Measures cognitive power.",
            "keys": ["Ability & Aptitude"]
        },
        # Duplicate URL
        {
            "entity_id": "1002",
            "name": "Cognitive Alternate",
            "link": "https://www.shl.com/view/cog-ability/",
            "description": "Measures cognitive power.",
            "keys": ["Ability & Aptitude"]
        },
        # Job Solution Item (should be excluded)
        {
            "entity_id": "1003",
            "name": "Java Developer Job Solution",
            "link": "https://www.shl.com/view/java-job/",
            "description": "Developer hiring package.",
            "keys": ["Job Solutions"]
        }
    ]
    
    cleaned = parser.clean_items(raw_data)
    
    # Check that duplicates and Job Solutions were filtered
    assert len(cleaned) == 1
    assert cleaned[0]["id"] == "1001"
    assert cleaned[0]["name"] == "General Cognitive Ability"
    assert cleaned[0]["adaptive_testing"] is True
    assert cleaned[0]["remote_support"] is True


@patch("urllib.request.urlopen")
def test_scraper_retry_on_network_failure(mock_urlopen: MagicMock) -> None:
    """
    Verifies that the scraper retries failed network connections.
    """
    # Configure mock to fail twice and then succeed
    mock_response = MagicMock()
    mock_response.read.return_value = b'[{"entity_id": "1", "name": "Test", "link": "http://test"}]'
    # Ensure the context manager return value is configured correctly
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.side_effect = [
        urllib.error.URLError("Connection refused"),
        urllib.error.URLError("Gateway Timeout"),
        mock_response
    ]
    
    scraper = CatalogScraper(max_retries=3, backoff_factor=0.01)
    data = scraper.fetch_raw_catalog()
    
    assert len(data) == 1
    assert data[0]["entity_id"] == "1"
    assert mock_urlopen.call_count == 3


def test_repository_extended_apis() -> None:
    """
    Verifies the compliance helper functions inside SHLCatalogRepository.
    """
    import asyncio
    import json
    from src.database.vector_store import SHLCatalogRepository
    
    # Create temp catalog path for testing
    import tempfile
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json", encoding="utf-8") as temp_file:
        test_data = [
            {
                "id": "720",
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "description": "Widely used personality questionnaire.",
                "assessment_category": ["Personality & Behavior"],
                "test_type": "P",
                "duration": "25 minutes",
                "languages": ["English"]
            },
            {
                "id": "740",
                "name": "SHL Verify Interactive G+",
                "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "description": "Adaptive ability test.",
                "assessment_category": ["Ability & Aptitude"],
                "test_type": "A",
                "duration": "36 minutes",
                "languages": ["English"]
            }
        ]
        json.dump(test_data, temp_file)
        temp_file_name = temp_file.name

    try:
        repo = SHLCatalogRepository(catalog_path=temp_file_name, vector_db_path="")
        asyncio.run(repo.load_catalog())
        
        # Test get_assessment_by_id
        item = asyncio.run(repo.get_assessment_by_id("720"))
        assert item is not None
        assert item.name == "Occupational Personality Questionnaire OPQ32r"
        
        # Test get_assessment_by_url
        item_url = asyncio.run(repo.get_assessment_by_url("https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/"))
        assert item_url is not None
        assert item_url.entity_id == "740"
        
        # Test list_all_assessments
        all_items = asyncio.run(repo.list_all_assessments())
        assert len(all_items) == 2
        
        # Test lookup_fuzzy
        item_fuzzy = asyncio.run(repo.lookup_fuzzy("Verify Interactive G+"))
        assert item_fuzzy is not None
        assert item_fuzzy.entity_id == "740"
        
        # Test validate_assessment / validate_url
        assert asyncio.run(repo.validate_assessment("SHL Verify Interactive G+")) is True
        assert asyncio.run(repo.validate_url("https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/")) is True
        assert asyncio.run(repo.validate_assessment("Non-existent Test")) is False
        
        # Test compare_assessments
        comparison = asyncio.run(repo.compare_assessments("OPQ32r", "Verify Interactive G+"))
        assert comparison is not None
        assert comparison["item1"]["name"] == "Occupational Personality Questionnaire OPQ32r"
        assert comparison["item2"]["name"] == "SHL Verify Interactive G+"
        
    finally:
        import os
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)
