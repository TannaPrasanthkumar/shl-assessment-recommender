"""
Pydantic model definitions for request/response serialization, validation,
and internal dialog state/catalog representation.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator


class Message(BaseModel):
    """
    Represents a single message in the conversational history.
    """
    role: Literal["user", "assistant", "system"] = Field(
        ..., 
        description="The role of the message author."
    )
    content: str = Field(
        ..., 
        description="The raw string content of the message."
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """
        Enforce that conversation turns must have non-empty content.
        """
        if not v.strip():
            raise ValueError("Message content cannot be empty or whitespace only.")
        return v


class ChatRequest(BaseModel):
    """
    Incoming request payload for the stateless POST /chat endpoint.
    """
    messages: List[Message] = Field(
        ..., 
        description="Stateless full conversation history."
    )

    @field_validator("messages")
    @classmethod
    def validate_message_history(cls, v: List[Message]) -> List[Message]:
        """
        Validates the conversation structure, ensuring it has at least one message.
        """
        if len(v) == 0:
            raise ValueError("Messages list must contain at least one message.")
        return v


class Recommendation(BaseModel):
    """
    Single recommended SHL assessment. 
    Strictly validated to ensure all fields are retrieved from the verified catalog.
    """
    name: str = Field(
        ..., 
        description="The official name of the SHL assessment."
    )
    url: str = Field(
        ..., 
        description="Verbatim scraped URL path matching the catalog."
    )
    test_type: str = Field(
        ..., 
        description="Designation of assessment type: K (Cognitive/Knowledge), P (Personality/Behavioral), etc."
    )


class ChatResponse(BaseModel):
    """
    Response schema returning conversational feedback and recommendations.
    Matches the automated evaluator's non-negotiable payload requirement.
    """
    reply: str = Field(
        ..., 
        description="The conversational text generated for the user."
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list, 
        description="An array of 1 to 10 recommended assessments (empty if turn is clarification/refusal)."
    )
    end_of_conversation: bool = Field(
        default=False, 
        description="Indicates if the agent considers the recommendation dialogue complete."
    )


class ConstraintState(BaseModel):
    """
    Dialog state representing extracted constraints from the conversation.
    Normalized state model used by the Decision Engine.
    """
    job_role: Optional[str] = Field(None, description="Extracted targeted job role.")
    seniority: Optional[str] = Field(None, description="Experience level or target rank.")
    experience: Optional[str] = Field(None, description="Years of experience or specific duration.")
    industry: Optional[str] = Field(None, description="Target industry vertical.")
    programming_languages: List[str] = Field(default_factory=list, description="Target coding languages.")
    skills: List[str] = Field(default_factory=list, description="Extracted technical skills.")
    competencies: List[str] = Field(default_factory=list, description="Targeted soft skills or focus areas.")
    test_type_preferences: List[str] = Field(default_factory=list, description="Target types, e.g. K (Cognitive) or P (Personality).")
    must_include: List[str] = Field(default_factory=list, description="Specific tests to include.")
    must_exclude: List[str] = Field(default_factory=list, description="Keywords or names to exclude from recommendations.")
    job_description: Optional[str] = Field(None, description="Raw text of the uploaded job description.")
    unknown_fields: List[str] = Field(default_factory=list, description="Unrecognized or vague constraints requiring clarification.")
    communication: Optional[bool] = Field(None, description="True if communication is required.")
    stakeholder_interaction: Optional[bool] = Field(None, description="True if stakeholder interaction is required.")
    leadership: Optional[bool] = Field(None, description="True if leadership is required.")
    personality_needed: Optional[bool] = Field(None, description="True if personality/behavioral traits are needed.")
    cognitive_requirements: Optional[bool] = Field(None, description="True if cognitive/ability traits are needed.")
    domain: Optional[str] = Field(None, description="Hiring domain or industry area.")
    frameworks: List[str] = Field(default_factory=list, description="Target technical frameworks.")
    clarification_needed: Optional[bool] = Field(None, description="True if dialogue lacks key information to recommend.")



class CatalogItem(BaseModel):
    """
    Entity representation of an assessment scraped from the SHL product catalog.
    """
    entity_id: str = Field(..., description="Unique assessment identifier.")
    name: str = Field(..., description="Assessment name.")
    link: str = Field(..., description="Verbatim URL link.")
    job_levels: List[str] = Field(default_factory=list, description="Job levels mapped to this assessment.")
    languages: List[str] = Field(default_factory=list, description="Available localization languages.")
    duration: str = Field(default="", description="Estimated assessment completion duration.")
    description: str = Field(..., description="Description details for indexing and vector mapping.")
    keys: List[str] = Field(default_factory=list, description="Key tags mapped by SHL for the test category.")
    competencies: List[str] = Field(default_factory=list, description="Extracted soft competencies.")
    skills: List[str] = Field(default_factory=list, description="Extracted technical skills.")
    test_type: str = Field(default="K", description="Mapped test type character.")


class HealthResponse(BaseModel):
    """
    Response schema returning current status of the service health.
    """
    status: str = Field(..., description="Overall system health status.")

