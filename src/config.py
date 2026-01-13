import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # LLM Settings
    DEEPSEEK_API_KEY: str = Field(..., validation_alias="API_KEY_DEEPSEEK")
    DEEPSEEK_API_BASE: str = Field("https://api.deepseek.com", validation_alias="API_URL_DEEPSEEK")
    
    ALI_API_KEY: str = Field(..., validation_alias="API_KEY_ALI")
    ALI_API_URL: str = Field("https://dashscope.aliyuncs.com/compatible-mode/v1", validation_alias="ALI_API_URL")

    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = Field(default=None) # Optional if user wants to provide full URL
    QDRANT_COLLECTION_NAME: str = "technical_docs"
    
    # Model Settings
    LLM_MODEL_NAME: str = "deepseek-v3.2"
    # Logical name (path resolution is handled in embedding.py)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-large"
    
    # Text Splitting Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def qdrant_effective_url(self) -> str:
        if self.QDRANT_URL:
            return self.QDRANT_URL
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

# Global settings instance
settings = Settings()
