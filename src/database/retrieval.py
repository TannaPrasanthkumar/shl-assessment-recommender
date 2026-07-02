"""
Hybrid Retrieval System.
Implements lexical (BM25) and semantic (Dense Vector with FAISS) search models,
combining rankings via Reciprocal Rank Fusion (RRF), filtering by metadata constraints,
reranking using a Cross-Encoder, and applying business rule boosts.
Includes fallback mechanisms for offline execution.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Try importing ML libraries, providing offline fallbacks if they are missing
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
    HAS_ST = True
except ImportError:
    HAS_ST = False

from src.models.schemas import CatalogItem, ConstraintState
from src.utils.logger import app_logger

# Simple regex-based tokenization helper for fallback / lexical parsing
def simple_tokenize(text: str) -> List[str]:
    """
    Lowercase text, strips punctuation, and returns token list.
    """
    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean_text.split() if w]


class IRetriever(ABC):
    """
    Base contract for retrieval pipelines.
    """
    @abstractmethod
    def build_index(self, items: List[CatalogItem]) -> None:
        """
        Builds the retriever search index from catalog records.
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 20) -> List[Tuple[CatalogItem, float]]:
        """
        Executes query retrieval and returns items paired with their similarity score.
        """
        pass


class BM25Retriever(IRetriever):
    """
    Lexical search retriever utilizing the BM25Okapi algorithm.
    """

    def __init__(self) -> None:
        self.items: List[CatalogItem] = []
        self.bm25: Optional[Any] = None

    def build_index(self, items: List[CatalogItem]) -> None:
        self.items = items
        if not items:
            return

        corpus = []
        for item in items:
            # Combine index fields for vocabulary density
            text = f"{item.name} {item.description} {' '.join(item.keys)} " \
                   f"{' '.join(item.skills)} {' '.join(item.competencies)}"
            corpus.append(simple_tokenize(text))

        if HAS_BM25:
            self.bm25 = BM25Okapi(corpus)
            app_logger.info("BM25 index built successfully using rank_bm25.")
        else:
            app_logger.warning("rank_bm25 not installed. Using simple fallback TF-IDF similarity.")

    def search(self, query: str, limit: int = 20) -> List[Tuple[CatalogItem, float]]:
        if not self.items:
            return []

        tokens = simple_tokenize(query)
        if not tokens:
            return [(item, 0.0) for item in self.items[:limit]]

        # Use rank_bm25 if available
        if self.bm25:
            scores = self.bm25.get_scores(tokens)
            # Map index scores to items
            ranked_indices = np.argsort(scores)[::-1]
            results = []
            for idx in ranked_indices[:limit]:
                score = float(scores[idx])
                if score > 0.0:
                    results.append((self.items[idx], score))
            return results

        # Simple term frequency fallback overlap
        results = []
        query_set = set(tokens)
        for item in self.items:
            item_text = f"{item.name} {item.description}".lower()
            overlap = len(query_set.intersection(simple_tokenize(item_text)))
            score = float(overlap)
            if score > 0.0:
                results.append((item, score))
        
        # Sort by overlap score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


class DenseRetriever(IRetriever):
    """
    Semantic vector search retriever utilizing FAISS and Sentence Transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.items: List[CatalogItem] = []
        self.model: Optional[Any] = None
        self.index: Optional[Any] = None
        self._offline_mode = False

    def build_index(self, items: List[CatalogItem]) -> None:
        self.items = items
        if not items:
            return

        # Attempt to load SentenceTransformer
        if HAS_ST and not self._offline_mode:
            try:
                # Load model locally or download with timeout handling
                self.model = SentenceTransformer(self.model_name)
                app_logger.info(f"Loaded SentenceTransformer: {self.model_name}")
            except Exception as e:
                app_logger.warning(f"Could not load SentenceTransformer model ({e}). Switching to offline fallback.")
                self._offline_mode = True

        if not self._offline_mode and self.model:
            try:
                texts = [f"{item.name}: {item.description}" for item in items]
                embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                
                # Setup L2 flat L2 distance FAISS index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                # Normalize for L2 cosine-like similarity
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
                app_logger.info(f"FAISS vector index built successfully. Dimension: {dimension}")
                return
            except Exception as e:
                app_logger.error(f"FAISS indexing failed: {e}. Falling back to cosine similarity emulation.")
                self._offline_mode = True

        app_logger.warning("Dense retriever operating in offline/fallback mode.")

    def search(self, query: str, limit: int = 20) -> List[Tuple[CatalogItem, float]]:
        if not self.items:
            return []

        # If FAISS and SentenceTransformer are working
        if not self._offline_mode and self.model and self.index:
            try:
                query_vector = self.model.encode([query], convert_to_numpy=True)
                faiss.normalize_L2(query_vector)
                
                # Search FAISS L2 index
                distances, indices = self.index.search(query_vector, limit)
                
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1:
                        # Convert L2 distance to confidence metric
                        # L2 normalized distance is between 0 (identical) and 2 (orthogonal)
                        score = float(1.0 - (dist / 2.0))
                        results.append((self.items[idx], score))
                return results
            except Exception as e:
                app_logger.error(f"Vector search failed: {e}. Falling back.")

        # Emulated Dense Search: basic keyword overlap weighted by length
        results = []
        tokens = simple_tokenize(query)
        if not tokens:
            return [(item, 0.5) for item in self.items[:limit]]

        query_set = set(tokens)
        for item in self.items:
            text = f"{item.name} {item.description} {' '.join(item.keys)}".lower()
            item_tokens = simple_tokenize(text)
            overlap = len(query_set.intersection(item_tokens))
            score = float(overlap / (len(query_set) + len(item_tokens) + 1.0))
            results.append((item, score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


class CrossEncoderRanker:
    """
    Reranks candidate results using a Cross-Encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self.model: Optional[Any] = None
        self._offline_mode = True  # Disabled to prevent OOM memory issues on free-tier Render instances
        app_logger.info("Cross-Encoder disabled to optimize memory footprint.")

    def rerank(self, query: str, candidates: List[CatalogItem]) -> List[Tuple[CatalogItem, float]]:
        """
        Reranks catalog items for query relevance.
        """
        if not candidates:
            return []

        if not self._offline_mode and self.model:
            try:
                pairs = [[query, f"{item.name} - {item.description}"] for item in candidates]
                scores = self.model.predict(pairs)
                
                # Sigmoid normalization function with calibration offset for MS-Marco logits
                def sigmoid(x: float) -> float:
                    return 1.0 / (1.0 + np.exp(-(x + 3.0) / 2.0))
                
                ranked_results = []
                for item, score in zip(candidates, scores):
                    normalized_score = float(sigmoid(score))
                    ranked_results.append((item, normalized_score))
                
                ranked_results.sort(key=lambda x: x[1], reverse=True)
                return ranked_results
            except Exception as e:
                app_logger.error(f"CrossEncoder reranking failed: {e}. Falling back to baseline.")

        # Emulate CrossEncoder: compute score based on exact name match boosts and overlap density
        ranked_results = []
        query_clean = query.lower()
        for item in candidates:
            score = 0.5
            name_clean = item.name.lower()
            # Direct name presence boosts score
            if name_clean in query_clean or query_clean in name_clean:
                score += 0.3
            # Simple keyword overlap
            tokens = simple_tokenize(query)
            desc_tokens = simple_tokenize(item.description.lower())
            overlap = len(set(tokens).intersection(desc_tokens))
            score += min(0.2, overlap * 0.05)
            
            ranked_results.append((item, float(score)))
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results


class FusionRanker:
    """
    Combines lexical and semantic scores using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        lexical_results: List[Tuple[CatalogItem, float]],
        semantic_results: List[Tuple[CatalogItem, float]]
    ) -> List[Tuple[CatalogItem, float]]:
        rrf_scores: Dict[str, float] = {}
        item_map: Dict[str, CatalogItem] = {}

        # Apply RRF on Lexical Results
        for rank, (item, _) in enumerate(lexical_results):
            item_map[item.entity_id] = item
            rrf_scores[item.entity_id] = rrf_scores.get(item.entity_id, 0.0) + (1.0 / (self.k + (rank + 1)))

        # Apply RRF on Semantic Results
        for rank, (item, _) in enumerate(semantic_results):
            item_map[item.entity_id] = item
            rrf_scores[item.entity_id] = rrf_scores.get(item.entity_id, 0.0) + (1.0 / (self.k + (rank + 1)))

        combined = [(item_map[item_id], score) for item_id, score in rrf_scores.items()]
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined


class MetadataFilter:
    """
    Filters catalog candidates based on test type preferences and active exclusions.
    """
    def filter(
        self,
        candidates: List[Tuple[CatalogItem, float]],
        constraints: ConstraintState
    ) -> List[CatalogItem]:
        filtered_items = []
        for item, _ in candidates:
            # 1. Test Type Preferences Filter
            if constraints.test_type_preferences:
                item_types = [t.strip() for t in item.test_type.split(",")]
                has_type_match = False
                for preferred in constraints.test_type_preferences:
                    if preferred in item_types:
                        has_type_match = True
                        break
                if not has_type_match:
                    if "K" in constraints.test_type_preferences and "P" not in constraints.test_type_preferences and "A" not in constraints.test_type_preferences:
                        if "P" in item_types or "A" in item_types:
                            pass
                        else:
                            continue
                    else:
                        continue

            # 2. Exclusion Constraints
            if constraints.must_exclude:
                is_excluded = False
                for keyword in constraints.must_exclude:
                    kw_lower = keyword.lower()
                    if kw_lower in item.name.lower() or kw_lower in item.description.lower():
                        is_excluded = True
                        break
                if is_excluded:
                    continue

            filtered_items.append(item)
        return filtered_items


class BusinessRuleRanker:
    """
    Boosts candidate scores using matching role descriptors and competency vectors.
    """
    def apply_boosting(
        self,
        ranked_candidates: List[Tuple[CatalogItem, float]],
        constraints: ConstraintState
    ) -> List[Tuple[CatalogItem, float]]:
        boosted_list = []
        role_keywords = []
        if constraints.job_role:
            role_keywords = simple_tokenize(constraints.job_role)

        for item, score in ranked_candidates:
            boost = 1.0
            if role_keywords:
                desc_text = (item.name + " " + item.description).lower()
                for keyword in role_keywords:
                    if keyword in desc_text:
                        boost += 0.15
            if constraints.competencies:
                matched_comps = set(c.lower() for c in item.competencies).intersection(
                    set(c.lower() for c in constraints.competencies)
                )
                boost += 0.10 * len(matched_comps)

            boosted_list.append((item, score * boost))

        boosted_list.sort(key=lambda x: x[1], reverse=True)
        return boosted_list


class ConfidenceScorer:
    """
    Computes candidate retrieval confidence, mixing model scores with constraint coverage.
    """
    def calculate_confidence(
        self,
        boosted_results: List[Tuple[CatalogItem, float]],
        constraints: ConstraintState
    ) -> float:
        baseline_conf = float(np.mean([score for _, score in boosted_results[:3]])) if boosted_results else 0.0
        coverage_boost = 0.0
        if constraints.job_role:
            coverage_boost += 0.35
        if constraints.seniority:
            coverage_boost += 0.15
        if constraints.skills or constraints.programming_languages:
            coverage_boost += 0.15
        if constraints.competencies:
            coverage_boost += 0.15
            
        confidence = baseline_conf + coverage_boost
        return max(0.0, min(1.0, confidence))


class HybridRetriever:
    """
    Coordinates lexical and semantic search components.
    Merges scores with RRF, applies metadata filtering,
    reranks via Cross-Encoder, and boosts with business rules.
    """

    def __init__(self, catalog_items: List[CatalogItem], rrf_k: int = 60) -> None:
        self.catalog_items = catalog_items
        self.rrf_k = rrf_k
        
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever()
        self.reranker = CrossEncoderRanker()
        self.fusion_ranker = FusionRanker(k=rrf_k)
        self.metadata_filter = MetadataFilter()
        self.business_ranker = BusinessRuleRanker()
        self.confidence_scorer = ConfidenceScorer()

        # Build indexes on startup
        self.bm25_retriever.build_index(catalog_items)
        self.dense_retriever.build_index(catalog_items)

    def retrieve(
        self,
        query: str,
        constraints: ConstraintState,
        limit: int = 20
    ) -> Tuple[List[CatalogItem], float]:
        """
        Executes complete hybrid search pipeline based on structured constraints and weighted ranking.
        """
        search_parts = []
        if constraints.job_role:
            search_parts.append(f"{constraints.job_role} {constraints.job_role}")
        if constraints.programming_languages:
            search_parts.extend(constraints.programming_languages)
        if constraints.skills:
            search_parts.extend(constraints.skills)
        if constraints.competencies:
            search_parts.extend(constraints.competencies)
        if constraints.domain:
            search_parts.append(constraints.domain)
            
        formulated_query = " ".join(search_parts) if search_parts else query
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["opq", "verify", "g+", "numerical", "verbal", "logical", "svar"]):
            formulated_query = f"{query} {formulated_query}"

        app_logger.info(f"Executing retrieve query: '{formulated_query}' (original: '{query}') with constraints: {constraints.model_dump_json()}")
        
        # 1. Fetch Lexical and Semantic Candidate Pools
        lexical_candidates = self.bm25_retriever.search(formulated_query, limit=100)
        semantic_candidates = self.dense_retriever.search(formulated_query, limit=100)

        bm25_scores = {item.entity_id: score for item, score in lexical_candidates}
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0

        semantic_scores = {item.entity_id: score for item, score in semantic_candidates}
        max_semantic = max(semantic_scores.values()) if semantic_scores else 0.0

        # 2. RRF-based Pre-filtering Candidates Pool
        fused_candidates = self.fusion_ranker.fuse(lexical_candidates, semantic_candidates)
        filtered_candidates = self.metadata_filter.filter(fused_candidates, constraints)

        if not filtered_candidates:
            app_logger.warning("No catalog items matched constraints filter.")
            return [], 0.0

        # 3. Calculate Weighted Scores for Candidates
        scored_candidates = []
        for item in filtered_candidates:
            # BM25 Normalized component
            bm25_raw = bm25_scores.get(item.entity_id, 0.0)
            bm25_norm = bm25_raw / max_bm25 if max_bm25 > 0 else 0.0

            # Vector Similarity Normalized component
            semantic_raw = semantic_scores.get(item.entity_id, 0.0)
            semantic_norm = semantic_raw / max_semantic if max_semantic > 0 else 0.0

            # Constraint Match Score (0.0 to 1.0)
            constraint_score = 0.0
            
            # Role matching
            if constraints.job_role:
                role_clean = constraints.job_role.lower()
                item_name_lower = item.name.lower()
                item_desc_lower = item.description.lower()
                if role_clean in item_name_lower or role_clean in item_desc_lower:
                    constraint_score += 0.4
                else:
                    role_words = [w for w in simple_tokenize(role_clean) if len(w) > 2]
                    matches = sum(1 for w in role_words if w in item_name_lower or w in item_desc_lower)
                    if role_words:
                        constraint_score += min(0.4, 0.15 * matches)
            
            # Programming languages matching
            if constraints.programming_languages:
                lang_matches = 0
                for lang in constraints.programming_languages:
                    lang_clean = lang.lower()
                    if lang_clean in item.name.lower() or lang_clean in [s.lower() for s in item.skills]:
                        lang_matches += 1
                constraint_score += min(0.2, 0.2 * (lang_matches / len(constraints.programming_languages)))

            # Technical skills matching
            if constraints.skills:
                skill_matches = 0
                for skill in constraints.skills:
                    skill_clean = skill.lower()
                    if skill_clean in item.description.lower() or skill_clean in [s.lower() for s in item.skills] or skill_clean in [k.lower() for k in item.keys]:
                        skill_matches += 1
                constraint_score += min(0.2, 0.2 * (skill_matches / len(constraints.skills)))

            # Soft skills / Competencies matching
            target_comps = constraints.competencies
            if target_comps:
                comp_matches = 0
                for comp in target_comps:
                    comp_clean = comp.lower()
                    if comp_clean in item.description.lower() or comp_clean in [c.lower() for c in item.competencies]:
                        comp_matches += 1
                constraint_score += min(0.2, 0.2 * (comp_matches / len(target_comps)))

            # Soft skills extra indicators
            extra_indicators = 0.0
            item_text = (item.name + " " + item.description + " " + " ".join(item.competencies)).lower()
            if constraints.communication:
                if any(w in item_text for w in ["verbal", "communication", "following instructions", "write", "writing", "presentation", "persuasion"]):
                    extra_indicators += 0.1
            if constraints.stakeholder_interaction:
                if any(w in item_text for w in ["stakeholder", "customer", "client", "negotiation", "persuasion", "influence", "relationship", "teamwork"]):
                    extra_indicators += 0.1
            if constraints.leadership:
                if any(w in item_text for w in ["leadership", "manager", "leader", "management", "strategic", "director"]):
                    extra_indicators += 0.1
            constraint_score = min(1.0, constraint_score + extra_indicators)

            # Metadata Match Score (0.0 to 1.0)
            metadata_score = 0.0
            if constraints.test_type_preferences:
                item_types = [t.strip() for t in item.test_type.split(",")]
                if any(preferred in item_types for preferred in constraints.test_type_preferences):
                    metadata_score += 0.5
            else:
                metadata_score += 0.5
                
            if constraints.seniority:
                sen_lower = constraints.seniority.lower()
                levels_lower = [l.lower() for l in item.job_levels]
                is_match = False
                if "entry" in sen_lower or "junior" in sen_lower:
                    is_match = any(any(k in lvl for k in ["entry", "graduate", "individual contributor"]) for lvl in levels_lower)
                elif "mid" in sen_lower or "intermediate" in sen_lower or "professional" in sen_lower:
                    is_match = any(any(k in lvl for k in ["mid", "professional", "individual contributor", "management"]) for lvl in levels_lower)
                elif "senior" in sen_lower or "lead" in sen_lower:
                    is_match = any(any(k in lvl for k in ["mid", "professional", "management", "director", "senior"]) for lvl in levels_lower)
                elif "executive" in sen_lower or "director" in sen_lower or "manager" in sen_lower:
                    is_match = any(any(k in lvl for k in ["director", "executive", "management"]) for lvl in levels_lower)
                
                if is_match or not levels_lower:
                    metadata_score += 0.5
            else:
                metadata_score += 0.5

            # Business Rules Score (0.0 to 1.0)
            business_score = 0.5
            is_irrelevant_tech = False
            item_name_lower = item.name.lower()
            item_desc_lower = item.description.lower()
            
            # Irrelevant Technology Penalty
            requested_langs = [l.lower() for l in constraints.programming_languages]
            if requested_langs:
                # Other languages in catalog we want to filter out
                all_other_langs = ["sap", "abap", "cobol", "php", "ruby", "rust", "c++", "c#", "go", "golang", "swift", "kotlin", "scala", "java", "python"]
                other_unrequested_langs = [l for l in all_other_langs if l not in requested_langs]
                for other in other_unrequested_langs:
                    if other == "go" or other == "golang":
                        if re.search(r'\bgo\s+(?:developer|engineer|programmer|code|lang|programming|test|assessment)\b', item_name_lower) or \
                           re.search(r'\b(?:using|in|with)\s+go\b', item_name_lower) or \
                           "golang" in item_name_lower:
                            is_irrelevant_tech = True
                            break
                    else:
                        if re.search(r'\b' + re.escape(other) + r'\b', item_name_lower):
                            is_irrelevant_tech = True
                            break
            
            if is_irrelevant_tech:
                business_score = 0.0
            else:
                boosts = 0.0
                if constraints.communication:
                    if any(w in item_text for w in ["verbal", "communication", "following instructions", "write", "writing"]):
                        boosts += 0.2
                if constraints.stakeholder_interaction:
                    if any(w in item_text for w in ["stakeholder", "customer", "client", "negotiation", "persuasion", "influence", "personality"]):
                        boosts += 0.2
                if constraints.leadership:
                    if any(w in item_text for w in ["leadership", "manager", "leader", "management", "strategic", "director"]):
                        boosts += 0.2
                
                # Boost requested language match
                for req_lang in requested_langs:
                    if req_lang in item_name_lower or req_lang in [s.lower() for s in item.skills]:
                        boosts += 0.3
                business_score = min(1.0, business_score + boosts)

            # Unified Scoring weights:
            # 0.35 BM25 + 0.30 Vector Similarity + 0.20 Constraint Match + 0.10 Metadata Match + 0.05 Business Rules
            final_score = (
                0.35 * bm25_norm +
                0.30 * semantic_norm +
                0.20 * constraint_score +
                0.10 * metadata_score +
                0.05 * business_score
            )
            
            if is_irrelevant_tech:
                final_score = 0.0

            scored_candidates.append((item, final_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Diversify the sorted candidates to prevent duplicate assessment families
        seen_base_names = set()
        diversified_candidates = []
        
        for item, score in scored_candidates:
            # Normalization to group identical/near-duplicate tests
            name_lower = item.name.lower()
            # Strip prefixes like "shl", qualifiers like "interactive", "online", "standard", "new", "v1", "v2", "essentials", "version", "(new)" and punctuation
            base_name = re.sub(r'\(new\)|\[new\]|\bnew\b|\binteractive\b|\bonline\b|\bstandard\b|\bv\d+\b|\bversion\b|\bessentials\b|\bshl\b', '', name_lower)
            base_name = re.sub(r'[^a-z0-9]', ' ', base_name).strip()
            base_name = " ".join(base_name.split())
            
            if base_name in seen_base_names:
                continue
            seen_base_names.add(base_name)
            diversified_candidates.append((item, score))
            
        top_candidates = [item for item, _ in diversified_candidates[:limit]]
        confidence = self.confidence_scorer.calculate_confidence(diversified_candidates, constraints)

        app_logger.info(
            f"Retrieval complete. Found {len(top_candidates)} candidates. "
            f"Retrieval Confidence: {confidence:.4f}"
        )
        return top_candidates, confidence


