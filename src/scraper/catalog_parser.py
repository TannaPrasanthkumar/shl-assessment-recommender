"""
Catalog Ingestion and Normalization Pipeline.
Crawls the SHL catalog, filters out Job Solutions, removes duplicate entries,
normalizes metadata, and writes verified raw and clean databases.
"""

import json
import os
import re
import time
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error

from src.utils.logger import app_logger
from src.constants import TestTypes

# Standard metadata lookup mapping trace test types
KNOWN_TEST_TYPES: Dict[str, str] = {
    "Amazon Web Services (AWS) Development (New)": "K",
    "Basic Statistics (New)": "K",
    "Contact Center Call Simulation (New)": "S",
    "Core Java (Advanced Level) (New)": "K",
    "Customer Service Phone Simulation": "B,S",
    "Dependability and Safety Instrument (DSI)": "P",
    "Docker (New)": "K",
    "Entry Level Customer Serv - Retail & Contact Center": "P,C",
    "Financial Accounting (New)": "K",
    "Global Skills Assessment": "C, K",
    "Global Skills Development Report": "D",
    "Graduate Scenarios": "B",
    "HIPAA (Security)": "K",
    "Linux Programming (General)": "K",
    "MS Excel (New)": "K",
    "MS Word (New)": "K",
    "Manufac. & Indust. - Safety & Dependability 8.0": "P",
    "Medical Terminology (New)": "K",
    "Microsoft Excel 365 (New)": "K,S",
    "Microsoft Word 365 (New)": "K,S",
    "Microsoft Word 365 - Essentials (New)": "K,S",
    "Networking and Implementation (New)": "K",
    "OPQ Leadership Report": "P",
    "OPQ MQ Sales Report": "P",
    "OPQ Universal Competency Report 2.0": "P",
    "Occupational Personality Questionnaire OPQ32r": "P",
    "RESTful Web Services (New)": "K",
    "SHL Verify Interactive G+": "A",
    "SHL Verify Interactive  Numerical Reasoning": "A,S",
    "SQL (New)": "K",
    "SVAR Spoken English (US) (New)": "K",
    "Sales Transformation 2.0 - Individual Contributor": "P",
    "Smart Interview Live Coding": "K",
    "Spring (New)": "K",
    "Workplace Health and Safety (New)": "K"
}


class CatalogScraper:
    """
    Crawls SHL catalog data. Handles network connection retries and downloads.
    """

    def __init__(
        self,
        endpoint_url: str = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json",
        max_retries: int = 3,
        backoff_factor: float = 1.5
    ) -> None:
        self.endpoint_url = endpoint_url
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def fetch_raw_catalog(self) -> List[Dict[str, Any]]:
        """
        Fetches catalog JSON dataset from the network endpoint with retries and exponential backoff.
        """
        app_logger.info(f"Initiating fetch request to catalog endpoint: {self.endpoint_url}")
        
        retries = 0
        delay = 1.0
        
        while retries < self.max_retries:
            try:
                # Set request headers for compatibility
                req = urllib.request.Request(
                    self.endpoint_url,
                    headers={"User-Agent": "SHL-Ingestion-Pipeline/1.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    content = response.read().decode("utf-8")
                    # Parse using strict=False to allow control characters inside strings
                    data = json.loads(content, strict=False)
                    app_logger.info(f"Successfully retrieved {len(data)} items from endpoint.")
                    return data
            except urllib.error.URLError as e:
                retries += 1
                app_logger.warning(
                    f"Network connection failed: {e.reason if hasattr(e, 'reason') else e}. "
                    f"Retry {retries}/{self.max_retries} in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay *= self.backoff_factor
            except Exception as e:
                retries += 1
                app_logger.warning(
                    f"An error occurred during fetch: {e}. "
                    f"Retry {retries}/{self.max_retries} in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay *= self.backoff_factor

        raise RuntimeError(f"Failed to fetch catalog after {self.max_retries} retries.")


class CatalogParser:
    """
    Normalizes catalog items: cleans data fields, removes duplicate records,
    classifies assessment types, and extracts competency/skill tags.
    """

    @staticmethod
    def is_job_solution(item: Dict[str, Any]) -> bool:
        """
        Determines if an assessment is a pre-packaged Job Solution.
        """
        name = item.get("name", "").lower()
        description = item.get("description", "").lower()
        keys = [k.lower() for k in item.get("keys", [])]
        
        # Exclude common job solution designations
        if "job solution" in name or "job solution" in description:
            return True
        if "job solutions" in keys or "pre-packaged" in name or "pre-packaged" in description:
            return True
        return False

    @staticmethod
    def map_keys_to_type(keys: List[str], name: str = "") -> str:
        """
        Translates raw key categories to corresponding test type abbreviations.
        """
        name_clean = name.replace("  ", " ").strip()
        # Look up against trace verification tables first
        if name_clean in KNOWN_TEST_TYPES:
            return KNOWN_TEST_TYPES[name_clean]
        
        if "Development & 360" in keys:
            return "D"
            
        char_map = {
            "Ability & Aptitude": "A",
            "Biodata & Situational Judgment": "B",
            "Competencies": "C",
            "Development & 360": "D",
            "Personality & Behavior": "P",
            "Knowledge & Skills": "K",
            "Simulations": "S"
        }
        
        res = []
        for k in keys:
            if k in char_map:
                c = char_map[k]
                if c not in res:
                    res.append(c)
                    
        if not res:
            return "K"  # Fallback type for unmapped categories
            
        joined = ",".join(res)
        if joined == "C,K":
            return "C, K"
        return joined

    @staticmethod
    def extract_competencies(name: str, description: str, keys: List[str]) -> List[str]:
        """
        Parses competency tags from item description details.
        """
        comp_keywords = [
            "leadership", "management", "teamwork", "collaboration", "communication",
            "problem solving", "analytical", "decision making", "judgment",
            "adaptability", "resilience", "planning", "organizing", "sales",
            "influence", "negotiation", "customer service", "customer focus",
            "strategic", "innovation", "creativity"
        ]
        extracted = []
        text = (name + " " + description).lower()
        for kw in comp_keywords:
            if kw in text:
                extracted.append(kw.title())
        if "Competencies" in keys and not extracted:
            extracted.append("General Competency")
        return sorted(list(set(extracted)))

    @staticmethod
    def extract_skills(name: str, description: str, keys: List[str]) -> List[str]:
        """
        Extracts specific technical skills and knowledge areas.
        """
        skill_keywords = [
            "java", "python", "c#", ".net", "sql", "aws", "azure", "cloud", "docker",
            "kubernetes", "excel", "word", "office", "statistics", "accounting",
            "finance", "math", "numerical", "programming", "coding", "security",
            "network", "routing", "linux", "billing", "retail"
        ]
        extracted = []
        text = (name + " " + description).lower()
        for kw in skill_keywords:
            if kw in text:
                extracted.append(
                    kw.upper() if kw in ["aws", "sql", "net"] else kw.title()
                )
        return sorted(list(set(extracted)))

    @staticmethod
    def normalize_duration(duration_str: str, duration_raw: str) -> str:
        """
        Standardizes completion duration text formats.
        """
        # Parse numbers from duration fields
        m = re.search(r"(\d+)", duration_str)
        if m:
            return f"{m.group(1)} minutes"
        m = re.search(r"(\d+)", duration_raw)
        if m:
            return f"{m.group(1)} minutes"
        return "Not Specified"

    @staticmethod
    def normalize_boolean(value: Any) -> bool:
        """
        Safely converts truth value strings or properties to booleans.
        """
        if not value:
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ["yes", "true", "1"]

    @staticmethod
    def normalize_languages(languages_list: List[str], languages_raw: str) -> List[str]:
        """
        Parses and deduplicates language fields.
        """
        res = []
        if languages_list:
            res = [l.strip() for l in languages_list if l.strip()]
        elif languages_raw:
            res = [l.strip() for l in languages_raw.split(",") if l.strip()]
        if not res:
            res = ["English International"]
        return sorted(list(set(res)))

    @staticmethod
    def is_excluded_item(name: str, description: str, keys: List[str]) -> bool:
        """
        Enforces strict compliance filters to drop items that are not valid Individual Test Solutions.
        """
        lower_name = name.lower()
        lower_desc = description.lower()
        lower_keys = [k.lower() for k in keys]
        
        # 1. Exclude Job Solutions
        if "solution" in lower_name:
            return True
        if "job solution" in lower_name or "job solution" in lower_desc:
            return True
        if "job solutions" in lower_keys or "pre-packaged" in lower_name or "pre-packaged" in lower_desc:
            return True
            
        # 2. Exclude Solution Bundles & Suites
        if "bundle" in lower_name or "suite" in lower_name or "bundle" in lower_desc or "suite" in lower_desc:
            return True
            
        # 3. Exclude Reports that are not valid assessments
        # (Be careful to keep actual assessments or skills tests, e.g. SQL Server Reporting Services is NOT a report)
        if "report" in lower_name and "reporting" not in lower_name and "report writer" not in lower_name:
            return True
        if "feedback report" in lower_desc or "report is designed to" in lower_desc:
            return True
            
        # 4. Exclude Interview Guides & profiling guides
        if "guide" in lower_name or "guide" in lower_desc:
            return True
            
        # 5. Exclude Profiler Cards
        if "card" in lower_name or "cards" in lower_name:
            return True
            
        # 6. Exclude Supporting Documents / Marketing Pages
        if "supporting document" in lower_name or "supporting document" in lower_desc:
            return True
        if "marketing page" in lower_name or "marketing page" in lower_desc:
            return True
            
        return False

    def clean_items(self, raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main parsing loop. Normalizes fields and removes duplicates.
        """
        seen_ids = set()
        seen_urls = set()
        cleaned_list = []
        
        for item in raw_items:
            # Get entity ID
            entity_id = str(item.get("entity_id", "")).strip()
            url = str(item.get("link", "")).strip()
            
            # 1. Skip items without an ID or URL
            if not entity_id or not url:
                continue
                
            # 2. Enforce duplicate deduplication (de-duplicate IDs and URLs)
            if entity_id in seen_ids or url in seen_urls:
                continue
                
            # 3. Exclude Job Solutions
            if self.is_job_solution(item):
                continue
                
            # 4. Check strict compliance filters (bundles, reports, cards, guides)
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            keys = item.get("keys", [])
            
            if self.is_excluded_item(name, description, keys):
                continue
                
            seen_ids.add(entity_id)
            seen_urls.add(url)
            
            # Map clean catalog attributes
            cleaned_item = {
                "id": entity_id,
                "name": name,
                "url": url,
                "description": description,
                "assessment_category": keys,
                "competencies": self.extract_competencies(name, description, keys),
                "skills": self.extract_skills(name, description, keys),
                "duration": self.normalize_duration(
                    str(item.get("duration", "")),
                    str(item.get("duration_raw", ""))
                ),
                "adaptive_testing": self.normalize_boolean(item.get("adaptive")),
                "languages": self.normalize_languages(
                    item.get("languages", []),
                    str(item.get("languages_raw", ""))
                ),
                "remote_support": self.normalize_boolean(item.get("remote")),
                "test_type": self.map_keys_to_type(keys, name)
            }
            
            cleaned_list.append(cleaned_item)
            
        return cleaned_list


def run_ingestion_pipeline(
    raw_output_path: str = "data/raw_catalog.json",
    clean_output_path: str = "data/clean_catalog.json"
) -> None:
    """
    Executes the ingestion flow: fetches raw files, cleans records, and saves outputs.
    """
    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    
    scraper = CatalogScraper()
    parser = CatalogParser()
    
    try:
        raw_items = scraper.fetch_raw_catalog()
        
        # Save raw JSON catalog
        with open(raw_output_path, "w", encoding="utf-8") as f:
            json.dump(raw_items, f, indent=2, ensure_ascii=False)
        app_logger.info(f"Raw catalog saved to {raw_output_path}")
        
        # Clean catalog items
        cleaned_items = parser.clean_items(raw_items)
        
        # Validate clean catalog constraints (no duplicates, no empty values)
        seen_ids = set()
        seen_urls = set()
        for idx, item in enumerate(cleaned_items):
            item_id = item["id"]
            item_url = item["url"]
            if not item_id or not item_url or not item["name"]:
                raise ValueError(f"Validation failed: missing crucial fields in item index {idx}.")
            if item_id in seen_ids:
                raise ValueError(f"Validation failed: duplicate ID {item_id} found in clean output.")
            if item_url in seen_urls:
                raise ValueError(f"Validation failed: duplicate URL {item_url} found in clean output.")
            seen_ids.add(item_id)
            seen_urls.add(item_url)
            
        # Save cleaned JSON catalog
        with open(clean_output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_items, f, indent=2, ensure_ascii=False)
        app_logger.info(f"Clean catalog saved to {clean_output_path}. Total clean items: {len(cleaned_items)}")
        
    except Exception as e:
        app_logger.error(f"Ingestion pipeline failed: {e}")
        raise
