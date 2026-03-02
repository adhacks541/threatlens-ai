from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AnyUrl, validator
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "ThreatLens AI"
    # Endee Vector DB Configuration
    ENDEE_URL: str = Field(default="http://localhost:8080", description="URL to the Endee vector database")
    ENDEE_AUTH_TOKEN: Optional[str] = Field(default=None, description="Optional auth token for Endee")
    ENDEE_COLLECTION_NAME: str = Field(default="threat_intel", description="Name of the index/collection")
    
    # OpenAI / LLM Configuration
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API Key")
    LLM_MODEL: str = Field(default="gpt-4o", description="LLM model to use")
    
    # Embeddings
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Huggingface sentence-transformers model")
    EMBEDDING_DIMENSION: int = Field(default=384, description="Dimension of embeddings vectors")

    # App Config
    DEFAULT_TOP_K: int = Field(default=3, description="Default number of vectors to retrieve")
    HYBRID_ALPHA: float = Field(default=0.7, description="Weight of the Dense (Semantic) score in Hybrid Retrieval")
    HYBRID_BETA: float = Field(default=0.3, description="Weight of the Sparse (Keyword) score in Hybrid Retrieval")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
