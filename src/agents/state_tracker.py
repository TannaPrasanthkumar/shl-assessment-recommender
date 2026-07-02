import re
import json
import asyncio
import sys
from typing import List, Optional, Set, Any
from src.config import settings
from src.models.schemas import Message, ConstraintState
from src.utils.logger import app_logger


# Define clean dictionaries for regex matching
SENIORITY_KEYWORDS = {
    "entry": ["entry", "entry-level", "graduate", "junior", "intern", "trainee"],
    "mid": ["mid", "mid-level", "intermediate", "professional"],
    "senior": ["senior", "lead", "principal", "expert", "director", "manager", "supervisor"],
    "executive": ["executive", "cxo", "vp", "vice president", "chief"]
}

TECH_LANGUAGES = [
    "java", "python", "c#", ".net", "javascript", "typescript", "c\\+\\+",
    "ruby", "rust", "php", "go", "golang", "scala", "kotlin", "swift"
]

TECHNICAL_SKILLS = [
    "aws", "azure", "cloud", "docker", "kubernetes", "sql", "excel", "word",
    "powerpoint", "accounting", "finance", "billing", "statistics", "testing",
    "security", "networking", "databases", "programming", "coding"
]

COMPETENCY_KEYWORDS = [
    "leadership", "teamwork", "collaboration", "communication", "sales",
    "influence", "negotiation", "customer service", "stakeholder",
    "problem solving", "analytical", "decision making", "planning"
]

TEST_TYPE_MAP = {
    "cognitive": "A",
    "ability": "A",
    "reasoning": "A",
    "aptitude": "A",
    "personality": "P",
    "behavior": "P",
    "traits": "P",
    "skills": "K",
    "knowledge": "K",
    "technical": "K",
    "simulation": "S",
    "interactive": "S"
}


class ConstraintGraph:
    """
    Graph structure maintaining active dialog constraints.
    Supports operations: merge, replace, delete, and validate.
    """

    def __init__(self) -> None:
        self.state = ConstraintState()

    def merge_constraints(self, updates: ConstraintState) -> None:
        """
        Merges new constraints into the existing state.
        Lists are appended/deduplicated; scalar values are overwritten if present.
        """
        if updates.job_role:
            self.state.job_role = updates.job_role
        if updates.seniority:
            self.state.seniority = updates.seniority
        if updates.experience:
            self.state.experience = updates.experience
        if updates.industry:
            self.state.industry = updates.industry
        if updates.job_description:
            self.state.job_description = updates.job_description

        # Merge Lists and deduplicate
        self.state.programming_languages = sorted(list(set(
            self.state.programming_languages + updates.programming_languages
        )))
        self.state.skills = sorted(list(set(
            self.state.skills + updates.skills
        )))
        self.state.competencies = sorted(list(set(
            self.state.competencies + updates.competencies
        )))
        self.state.test_type_preferences = sorted(list(set(
            self.state.test_type_preferences + updates.test_type_preferences
        )))
        self.state.must_include = sorted(list(set(
            self.state.must_include + updates.must_include
        )))
        self.state.must_exclude = sorted(list(set(
            self.state.must_exclude + updates.must_exclude
        )))
        self.state.unknown_fields = sorted(list(set(
            self.state.unknown_fields + updates.unknown_fields
        )))

    def replace_constraints(self, key: str, value: Any) -> None:
        """
        Directly replaces an active constraint key value.
        """
        if hasattr(self.state, key):
            setattr(self.state, key, value)
            app_logger.info(f"Constraint replaced: {key} -> {value}")

    def delete_constraints(self, key: str, value_to_remove: Optional[Any] = None) -> None:
        """
        Deletes a constraint parameter or removes an element from a list constraint.
        """
        if hasattr(self.state, key):
            if value_to_remove is None:
                # Reset field
                if isinstance(getattr(self.state, key), list):
                    setattr(self.state, key, [])
                else:
                    setattr(self.state, key, None)
                app_logger.info(f"Constraint cleared: {key}")
            else:
                # Remove specific item from list
                current_list = getattr(self.state, key)
                if isinstance(current_list, list) and value_to_remove in current_list:
                    current_list.remove(value_to_remove)
                    setattr(self.state, key, current_list)
                    app_logger.info(f"Removed item from constraint list '{key}': {value_to_remove}")

    def validate(self) -> None:
        """
        Validates active constraints against boundary rules and cleans up duplicates.
        Ensures consistency (e.g. items cannot be in must_include and must_exclude simultaneously).
        """
        # Overlap clean: if item is in must_exclude, remove from must_include
        exclude_set = set(self.state.must_exclude)
        self.state.must_include = [x for x in self.state.must_include if x not in exclude_set]
        
        # Clean skills vs programming languages overlap
        lang_set = set(self.state.programming_languages)
        self.state.skills = [s for s in self.state.skills if s.lower() not in [l.lower() for l in lang_set]]


class DialogueStateTracker:
    """
    Statelessly analyzes historical message turns.
    Reconstructs ConstraintGraph using deterministic text parsing rules or LLM-powered extraction.
    """

    def __init__(self) -> None:
        pass

    async def _extract_constraints_llm(self, messages: List[Message]) -> Optional[ConstraintState]:
        keys = settings.gemini_keys
        if not keys:
            return None

        # Build prompt from messages history
        history_lines = []
        for msg in messages:
            history_lines.append(f"{msg.role.capitalize()}: {msg.content}")
        messages_text = "\n".join(history_lines)

        prompt = f"""Analyze the following dialogue history between a Recruiter/Hiring Manager (User) and an Assistant (Agent).
Extract the current, active structured hiring requirements.

If the user changes or overrides a previous constraint (e.g. 'Actually, change role to sales representative' or 'no personality tests'), capture the latest updated state.

Dialogue History:
{messages_text}

Task: Extract and return ONLY a valid JSON object matching the schema below.
Rules:
- Do NOT output any conversational text or explanation. Only return a raw JSON object.
- Do NOT recommend any assessments.
- Normalize/clean string values: Seniority should be one of "Entry", "Mid", "Senior", "Executive". Programming languages must be in uppercase (e.g., "JAVA", "PYTHON", "C#").

JSON Schema:
{{
  "role": string or null,
  "seniority": string or null,
  "experience": string or null,
  "technical_skills": list of strings,
  "programming_languages": list of strings,
  "frameworks": list of strings,
  "soft_skills": list of strings,
  "communication": boolean or null,
  "stakeholder_interaction": boolean or null,
  "leadership": boolean or null,
  "personality_needed": boolean or null,
  "cognitive_requirements": boolean or null,
  "domain": string or null,
  "assessment_categories": list of strings,
  "clarification_needed": boolean or null
}}
"""
        import google.generativeai as genai
        for i, api_key in enumerate(keys):
            try:
                app_logger.info(f"DialogueStateTracker: Attempting LLM constraint extraction with key index {i}...")
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(settings.LLM_MODEL)
                
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
                
                text = response.text.strip()
                if text.startswith("```"):
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1:
                        text = text[start:end+1]
                
                data = json.loads(text)
                
                # Normalize values
                seniority = data.get("seniority")
                if seniority:
                    s_lower = seniority.lower()
                    if "entry" in s_lower or "junior" in s_lower or "intern" in s_lower or "graduate" in s_lower:
                        seniority = "Entry"
                    elif "mid" in s_lower or "intermediate" in s_lower or "professional" in s_lower:
                        seniority = "Mid"
                    elif "senior" in s_lower or "lead" in s_lower or "principal" in s_lower or "expert" in s_lower:
                        seniority = "Senior"
                    elif "executive" in s_lower or "cxo" in s_lower or "director" in s_lower or "vp" in s_lower:
                        seniority = "Executive"
                    else:
                        seniority = seniority.title()

                programming_languages = [lang.upper() for lang in data.get("programming_languages", [])]
                programming_languages = ["Go" if l == "GOLANG" else l for l in programming_languages]

                state = ConstraintState(
                    job_role=data.get("role") or data.get("job_role"),
                    seniority=seniority,
                    experience=data.get("experience"),
                    programming_languages=programming_languages,
                    skills=[s.title() for s in data.get("technical_skills", [])],
                    competencies=[c.title() for c in data.get("soft_skills", [])],
                    communication=data.get("communication"),
                    stakeholder_interaction=data.get("stakeholder_interaction"),
                    leadership=data.get("leadership"),
                    personality_needed=data.get("personality_needed"),
                    cognitive_requirements=data.get("cognitive_requirements"),
                    domain=data.get("domain") or data.get("industry"),
                    frameworks=data.get("frameworks", []),
                    clarification_needed=data.get("clarification_needed")
                )

                # Derive test_type_preferences
                types = []
                categories = [c.lower() for c in data.get("assessment_categories", [])]
                if data.get("personality_needed") or "personality" in categories or "behavior" in categories:
                    types.append("P")
                if data.get("cognitive_requirements") or "cognitive" in categories or "ability" in categories or "reasoning" in categories:
                    types.append("A")
                if "knowledge" in categories or "skills" in categories or state.programming_languages or state.skills:
                    types.append("K")
                state.test_type_preferences = sorted(list(set(types)))

                # Parse must_include / must_exclude from history
                clean_history = messages_text.lower()
                if "opq" in clean_history:
                    user_mentions = [msg.content.lower() for msg in messages if msg.role == "user"]
                    last_mention = user_mentions[-1] if user_mentions else ""
                    if any(neg in last_mention for neg in ["no ", "exclude", "ignore", "without", "except", "remove"]):
                        state.must_exclude.append("OPQ")
                    else:
                        state.must_include.append("OPQ")

                app_logger.info(f"DialogueStateTracker: LLM extraction complete: {state.model_dump_json()}")
                return state
            except Exception as e:
                app_logger.warning(f"DialogueStateTracker: LLM constraint extraction failed with key {i}: {e}")

        return None


    def _parse_turn(self, text: str) -> ConstraintState:
        """
        Analyzes a single user statement to extract candidate constraints.
        """
        clean_text = text.lower()
        extracted = ConstraintState()

        # 1. Extract Seniority / Experience
        for level, keywords in SENIORITY_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', clean_text):
                    extracted.seniority = level.title()
                    break

        # Check for numeric experience years (e.g. "5 years", "3+ years")
        exp_match = re.search(r'(\d+(\+)?)\s*years?', clean_text)
        if exp_match:
            extracted.experience = f"{exp_match.group(1)} years"

        # 2. Extract Programming Languages
        for lang in TECH_LANGUAGES:
            if lang == "go" or lang == "golang":
                # Special strict pattern for Go to prevent matching 'go ahead', 'let's go', etc.
                if re.search(r'\bgo\s+(?:developer|engineer|programmer|code|lang|programming|test|assessment)\b', clean_text) or \
                   re.search(r'\b(?:using|in|with)\s+go\b', clean_text) or \
                   "golang" in clean_text:
                    if "Go" not in extracted.programming_languages:
                        extracted.programming_languages.append("Go")
            else:
                if re.search(r'\b' + re.escape(lang) + r'\b', clean_text):
                    extracted.programming_languages.append(lang.upper() if lang != "golang" else "Go")

        # 3. Extract Technical Skills
        for skill in TECHNICAL_SKILLS:
            if re.search(r'\b' + re.escape(skill) + r'\b', clean_text):
                extracted.skills.append(skill.title() if skill != "sql" else "SQL")

        # 4. Extract Competencies
        for comp in COMPETENCY_KEYWORDS:
            if re.search(r'\b' + re.escape(comp) + r'\b', clean_text):
                extracted.competencies.append(comp.title())

        # 5. Extract Assessment Type Preferences
        for type_keyword, char in TEST_TYPE_MAP.items():
            if re.search(r'\b' + re.escape(type_keyword) + r'\b', clean_text):
                extracted.test_type_preferences.append(char)
        
        # If programming languages or technical skills are found, default type preference to K (Knowledge & Skills)
        if extracted.programming_languages or extracted.skills:
            if "K" not in extracted.test_type_preferences:
                extracted.test_type_preferences.append("K")
        # 6. Extract must_include vs must_exclude
        # Examples: "add OPQ32", "exclude verify cognitive", "no personality tests"
        if "opq" in clean_text:
            if any(neg in clean_text for neg in ["no ", "exclude", "ignore", "without", "except"]):
                extracted.must_exclude.append("OPQ")
            else:
                extracted.must_include.append("OPQ")

        # 7. Extract Industry Vertical (Finance, Healthcare, Retail, etc.)
        for ind in ["finance", "healthcare", "retail", "engineering", "manufacturing"]:
            if re.search(r'\b' + re.escape(ind) + r'\b', clean_text):
                extracted.industry = ind.title()

        # 8. Check if user is uploading a Job Description (starts with standard markers or has length > 120 chars)
        if len(text) > 120 and ("job description" in clean_text or "hiring profile" in clean_text or "role requirements" in clean_text):
            extracted.job_description = text

        return extracted

    async def extract_constraints(self, messages: List[Message]) -> ConstraintState:
        """
        Reconstructs the active ConstraintGraph from stateless conversation history.
        First attempts LLM-powered constraint extraction, and falls back to sequence graph rules.
        """
        # 1. Try LLM constraint extraction first (only if not running unit tests to prevent quota exhaustion)
        is_testing = "pytest" in sys.modules or any("pytest" in arg or "unittest" in arg for arg in sys.argv)
        if not is_testing:
            llm_state = await self._extract_constraints_llm(messages)
            if llm_state is not None:
                # Title-case key text elements for consistent UI presentation
                if llm_state.job_role:
                    llm_state.job_role = llm_state.job_role.title()
                if llm_state.industry:
                    llm_state.industry = llm_state.industry.title()
                if llm_state.domain:
                    llm_state.domain = llm_state.domain.title()
                return llm_state

        # 2. Fall back to sequence-based regex rules if LLM is not configured/fails
        graph = ConstraintGraph()

        for msg in messages:
            # Process user messages only
            if msg.role != "user":
                continue

            text = msg.content
            clean_text = text.lower()

            # 1. Global Reset check
            if any(reset_word in clean_text for reset_word in ["start over", "clear all", "restart"]):
                graph = ConstraintGraph()
                app_logger.info("Stateless DST: Resetting constraints graph.")

            # 2. Extract updates from current turn
            turn_updates = self._parse_turn(text)

            # 3. Determine job role and strip seniority adjectives
            parsed_role = None
            role_patterns = [
                r'(?:need to hire|looking for|solution for|assessment for|assessments for|tests for|hiring for|hiring|hire)\s+(?:a|an|the)?\s*([a-zA-Z0-9\s\-\+\#]+?)(?:\s+who|\s+with|\bthat\b|\.|\,|$)',
                r'(?:need|want)\s+(?:a|an|the)?\s*([a-zA-Z0-9\s\-\+\#]+?)(?:\s+who|\s+with|\bthat\b|\btest\b|\bassessment\b|\.|\,|$)',
                r'(?:screening|screen|assess|assessing|testing|recruiting|recruit)\s+(?:\d+|a|an|the)?\s*(?:entry-level|junior|senior|mid)?\s*([a-zA-Z0-9\s\-\+\#]+?)(?:\s+who|\s+with|\bthat\b|\.|\,|$)',
                r'([a-zA-Z0-9\s\-\+\#]+?)\s+(?:position|role|job|hiring)\b'
            ]
            for pattern in role_patterns:
                role_match = re.search(pattern, text, re.IGNORECASE)
                if role_match:
                    candidate_role = role_match.group(1).strip()
                    # Clean out seniority keywords from the role name
                    for level, keywords in SENIORITY_KEYWORDS.items():
                        for kw in keywords:
                            candidate_role = re.sub(r'^\b' + re.escape(kw) + r'\b\s*', '', candidate_role, flags=re.IGNORECASE)
                            candidate_role = re.sub(r'\s*\b' + re.escape(kw) + r'\b$', '', candidate_role, flags=re.IGNORECASE)
                    candidate_role = candidate_role.strip().title()
                    
                    # Filter out vague prefixes
                    if candidate_role.lower() not in ["solution", "developer", "candidate", "test", "assessment", "programmer", "dev", "role", "position"]:
                        parsed_role = candidate_role
                        break

            if parsed_role:
                turn_updates.job_role = parsed_role

            # 4. Merge normal turn updates
            graph.merge_constraints(turn_updates)

            # 5. Handle explicit Constraint Retraction/Delete operations after merge
            if "actually" in clean_text or "change" in clean_text or "remove" in clean_text or "ignore" in clean_text:
                # If retracting personality
                if any(p_word in clean_text for p_word in ["no personality", "remove personality", "ignore personality"]):
                    graph.delete_constraints("test_type_preferences", "P")
                # If retracting cognitive
                if any(c_word in clean_text for c_word in ["no cognitive", "remove cognitive", "ignore cognitive"]):
                    graph.delete_constraints("test_type_preferences", "A")
                
                # Check for explicit role replacements (e.g. "actually, change role to sales")
                role_replace_match = re.search(r'(?:role to|hiring for|actually)\s+(?:a|an)?\s*([a-zA-Z0-9\s\-]+)', clean_text)
                if role_replace_match:
                    new_role = role_replace_match.group(1).strip()
                    # Clean out seniority
                    for level, keywords in SENIORITY_KEYWORDS.items():
                        for kw in keywords:
                            new_role = re.sub(r'^\b' + re.escape(kw) + r'\b\s*', '', new_role, flags=re.IGNORECASE)
                            new_role = re.sub(r'\s*\b' + re.escape(kw) + r'\b$', '', new_role, flags=re.IGNORECASE)
                    new_role = new_role.strip().title()
                    if new_role.lower() not in ["solution", "developer", "candidate", "test", "assessment", "programmer", "dev"]:
                        graph.replace_constraints("job_role", new_role)

            # 6. Run validation rules
            graph.validate()

        # Log active constraints for tracking
        app_logger.info(f"Stateless DST complete. Extracted Constraints: {graph.state.model_dump_json()}")
        return graph.state

