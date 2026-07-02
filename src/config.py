"""
Configuration management system for env validation and application parameters.
Uses Pydantic settings for strict type verification of environment configuration.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Explicitly load .env into environment variables on startup
load_dotenv()



class AppSettings(BaseSettings):
    """
    Validates and stores system environment configuration.
    Falls back to safe defaults or loads from .env where applicable.
    """
    # API metadata
    PROJECT_NAME: str = "SHL Assessment Recommender API"
    API_V1_STR: str = "/api/v1"
    
    # Environment mode
    ENV: str = "development"
    DEBUG: bool = False
    
    # Paths configuration
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CATALOG_FILE_PATH: Path = DATA_DIR / "raw_catalog.json"
    
    # LLM Settings
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gemini-2.5-flash"
    
    # Database Settings
    VECTOR_DB_PATH: str = "data/vector_db"
    
    # Log settings
    LOG_LEVEL: str = "INFO"
    
    # Config configuration for Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def gemini_keys(self) -> list[str]:
        """
        Deduplicated list of available Gemini API keys loaded from config or environment.
        """
        keys = []
        if self.GEMINI_API_KEY:
            keys.append(self.GEMINI_API_KEY)
            
        # Check standard hyphenated environment variables directly
        for k in ["gemini-api-key-1", "gemini-api-key-2", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_BACKUP"]:
            val = os.environ.get(k)
            if val and val not in keys:
                keys.append(val)
        return keys



# Globally instantiated configuration instance
settings = AppSettings()
