"""
Unit tests for the Dialog State Tracker (DST) module.
Verifies parsing of conversation history, constraint operations (merge, replace, delete),
and schema validations.
"""

import pytest
import asyncio
from src.agents.state_tracker import DialogueStateTracker
from src.models.schemas import Message


def test_basic_constraint_extraction() -> None:
    """
    Verifies that basic attributes like role, seniority, coding languages,
    and competency traits are correctly parsed from a single turn.
    """
    tracker = DialogueStateTracker()
    
    # 1. Test target role and seniority
    msg = Message(role="user", content="I need to hire a senior Java Developer who can lead teamwork.")
    state = asyncio.run(tracker.extract_constraints([msg]))
    
    assert state.job_role == "Java Developer"
    assert state.seniority == "Senior"
    assert "JAVA" in state.programming_languages
    assert "Teamwork" in state.competencies
    assert "K" in state.test_type_preferences  # Java is technical (K)
    
    # 2. Test industry vertical and specific experience
    msg_2 = Message(role="user", content="Hiring for a sales candidate with 5 years experience in finance.")
    state_2 = asyncio.run(tracker.extract_constraints([msg_2]))
    
    assert state_2.job_role == "Sales Candidate"
    assert state_2.experience == "5 years"
    assert state_2.industry == "Finance"
    assert "Sales" in state_2.competencies


def test_constraint_merge_multi_turn() -> None:
    """
    Verifies that constraints accumulate and merge across multiple conversation turns.
    """
    tracker = DialogueStateTracker()
    
    history = [
        Message(role="user", content="We need an assessment for a junior Python programmer."),
        Message(role="assistant", content="What competencies should we prioritize?"),
        Message(role="user", content="They must be good at problem solving and communication.")
    ]
    
    state = asyncio.run(tracker.extract_constraints(history))
    
    assert state.job_role == "Python Programmer"
    assert state.seniority == "Entry"
    assert "PYTHON" in state.programming_languages
    assert "Problem Solving" in state.competencies
    assert "Communication" in state.competencies


def test_constraint_replace_override() -> None:
    """
    Verifies that changing constraints in history overrides existing values.
    """
    tracker = DialogueStateTracker()
    
    history = [
        Message(role="user", content="Hiring for a junior Python dev."),
        Message(role="user", content="Actually, change role to sales representative.")
    ]
    
    state = asyncio.run(tracker.extract_constraints(history))
    
    assert state.job_role == "Sales Representative"
    assert state.seniority == "Entry"  # Junior matches Entry, preserved unless overridden


def test_constraint_delete_retraction() -> None:
    """
    Verifies that retractions (e.g. 'remove' or 'no' command phrases)
    correctly delete constraint elements.
    """
    tracker = DialogueStateTracker()
    
    history = [
        Message(role="user", content="I want cognitive and personality tests for a manager."),
        Message(role="user", content="Actually, no personality tests.")
    ]
    
    state = asyncio.run(tracker.extract_constraints(history))
    
    assert "A" in state.test_type_preferences  # Cognitive remains
    assert "P" not in state.test_type_preferences  # Personality is deleted
    assert state.seniority == "Senior"  # Manager matches Senior, preserved


def test_global_reset() -> None:
    """
    Verifies that commands like 'start over' reset the constraint graph completely.
    """
    tracker = DialogueStateTracker()
    
    history = [
        Message(role="user", content="Hiring a senior developer with AWS skills."),
        Message(role="user", content="Let's start over, we need a junior accountant.")
    ]
    
    state = asyncio.run(tracker.extract_constraints(history))
    
    assert state.job_role == "Accountant"
    assert state.seniority == "Entry"
    assert "AWS" not in state.skills
    assert "JAVA" not in state.programming_languages
