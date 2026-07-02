import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.config import settings
from src.models.schemas import CatalogItem, Recommendation, ConstraintState
from src.utils.logger import app_logger


class ICatalogRepository(ABC):
    """
    Interface for catalog operations and query retrieval.
    Following Dependency Inversion Principle.
    """

    @abstractmethod
    async def load_catalog(self) -> None:
        """
        Asynchronously loads catalog items into memory or vector index.
        Typically executed during health checks or startup warmup.
        """
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        constraints: ConstraintState,
        limit: int = 10
    ) -> List[CatalogItem]:
        """
        Retrieves matching catalog items based on semantic query and metadata filters.
        """
        pass

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[CatalogItem]:
        """
        Resolves an assessment by its exact catalog name.
        """
        pass

    async def get_assessment_by_id(self, assessment_id: str) -> Optional[CatalogItem]:
        """Resolves an assessment by its unique ID."""
        return None

    async def get_assessment_by_name(self, name: str) -> Optional[CatalogItem]:
        """Resolves an assessment by its catalog name."""
        return None

    async def get_assessment_by_url(self, url: str) -> Optional[CatalogItem]:
        """Resolves an assessment by its official URL link."""
        return None

    async def list_all_assessments(self) -> List[CatalogItem]:
        """Lists all parsed assessments in the catalog."""
        return []

    async def compare_assessments(self, name1: str, name2: str) -> Optional[Dict[str, Any]]:
        """Retrieves and compares metadata of two assessments."""
        return None

    async def lookup_exact(self, name: str) -> Optional[CatalogItem]:
        """Performs case-insensitive exact catalog lookup."""
        return None

    async def lookup_fuzzy(self, name: str) -> Optional[CatalogItem]:
        """Performs fuzzy/substring catalog lookup."""
        return None

    async def validate_assessment(self, name: str) -> bool:
        """Verifies if the assessment exists by name."""
        return False

    async def validate_url(self, url: str) -> bool:
        """Verifies if the URL matches an official catalog link."""
        return False


class SHLCatalogRepository(ICatalogRepository):
    """
    Production implementation of the catalog repository.
    Combines BM25 indexing with dense embedding vectors for candidate retrieval.
    """

    def __init__(self, catalog_path: str, vector_db_path: str) -> None:
        """
        Initializes catalog database settings.
        
        Args:
            catalog_path: Filepath pointing to the clean/raw catalog JSON database.
            vector_db_path: Directory path for the vector index persistence.
        """
        self.catalog_path = catalog_path
        self.vector_db_path = vector_db_path
        self._items: Dict[str, CatalogItem] = {}
        self.retriever: Optional[HybridRetriever] = None
        self._lock = asyncio.Lock()

    async def load_catalog(self) -> None:
        """
        Parses cleaned catalog data and initializes the retrieval search indexes.
        """
        async with self._lock:
            if self._items:
                return

            import json
            from src.database.retrieval import HybridRetriever

            if not os.path.exists(self.catalog_path):
                resolved_path = os.path.join(settings.BASE_DIR, "data", "clean_catalog.json")
                if os.path.exists(resolved_path):
                    self.catalog_path = resolved_path
                else:
                    app_logger.error(f"Catalog file does not exist at path: {self.catalog_path}")
                    raise FileNotFoundError(f"Catalog file not found: {self.catalog_path}")

            try:
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    raw_data = json.loads(f.read(), strict=False)
                
                items_list = []
                for record in raw_data:
                    item = CatalogItem(
                        entity_id=record["id"],
                        name=record["name"],
                        link=record["url"],
                        job_levels=record.get("job_levels", []),
                        languages=record.get("languages", []),
                        duration=record.get("duration", ""),
                        description=record["description"],
                        keys=record.get("assessment_category", []),
                        competencies=record.get("competencies", []),
                        skills=record.get("skills", []),
                        test_type=record.get("test_type", "K")
                    )
                    self._items[item.name.lower()] = item
                    items_list.append(item)
                    
                self.retriever = HybridRetriever(items_list)
                app_logger.info(f"Catalog database loaded. Indexed {len(items_list)} items in repository.")
                
            except Exception as e:
                app_logger.error(f"Failed to load catalog into repository: {e}")
                raise

    async def hybrid_search(
        self,
        query: str,
        constraints: ConstraintState,
        limit: int = 10
    ) -> List[CatalogItem]:
        """
        Performs hybrid search using RRF, metadata filters, and Cross-Encoder reranking.
        
        Args:
            query: The formulated search query string.
            constraints: Filters to apply (test_type, exclusions).
            limit: Maximum items to retrieve.
            
        Returns:
            A list of CatalogItem results.
        """
        if not self.retriever:
            app_logger.warning("Search executed before loading catalog index. Triggering load_catalog.")
            await self.load_catalog()
            
        if self.retriever:
            # Hybrid search retrieves the top 20 candidates (Recall@10 target limit)
            candidates, confidence = self.retriever.retrieve(query, constraints, limit=limit)
            return candidates
        return []

    async def get_by_name(self, name: str) -> Optional[CatalogItem]:
        """
        Resolves an assessment by its exact catalog name.
        
        Args:
            name: Case-insensitive search name.
        """
        if not self._items:
            await self.load_catalog()
        return self._items.get(name.lower())

    async def get_assessment_by_id(self, assessment_id: str) -> Optional[CatalogItem]:
        """Resolves an assessment by its unique ID."""
        if not self._items:
            await self.load_catalog()
        for item in self._items.values():
            if item.entity_id == assessment_id:
                return item
        return None

    async def get_assessment_by_name(self, name: str) -> Optional[CatalogItem]:
        """Resolves an assessment by its catalog name."""
        return await self.get_by_name(name)

    async def get_assessment_by_url(self, url: str) -> Optional[CatalogItem]:
        """Resolves an assessment by its official URL link."""
        if not self._items:
            await self.load_catalog()
        clean_url = url.strip().rstrip("/").lower()
        for item in self._items.values():
            if item.link.strip().rstrip("/").lower() == clean_url:
                return item
        return None

    async def list_all_assessments(self) -> List[CatalogItem]:
        """Lists all parsed assessments in the catalog."""
        if not self._items:
            await self.load_catalog()
        return list(self._items.values())

    async def compare_assessments(self, name1: str, name2: str) -> Optional[Dict[str, Any]]:
        """Retrieves and compares metadata of two assessments."""
        item1 = await self.lookup_fuzzy(name1)
        item2 = await self.lookup_fuzzy(name2)
        if not item1 or not item2:
            return None
        return {
            "item1": {
                "name": item1.name,
                "type": item1.test_type,
                "duration": item1.duration,
                "url": item1.link,
                "description": item1.description
            },
            "item2": {
                "name": item2.name,
                "type": item2.test_type,
                "duration": item2.duration,
                "url": item2.link,
                "description": item2.description
            }
        }

    async def lookup_exact(self, name: str) -> Optional[CatalogItem]:
        """Performs case-insensitive exact catalog lookup."""
        return await self.get_by_name(name)

    async def lookup_fuzzy(self, name: str) -> Optional[CatalogItem]:
        """Performs fuzzy/substring catalog lookup."""
        if not self._items:
            await self.load_catalog()
        name_lower = name.lower()
        # Direct match
        if name_lower in self._items:
            return self._items[name_lower]
        # Substring matches
        for key, item in self._items.items():
            if name_lower in key or key in name_lower:
                return item
        return None

    async def validate_assessment(self, name: str) -> bool:
        """Verifies if the assessment exists by name."""
        item = await self.get_by_name(name)
        return item is not None

    async def validate_url(self, url: str) -> bool:
        """Verifies if the URL matches an official catalog link."""
        item = await self.get_assessment_by_url(url)
        return item is not None
