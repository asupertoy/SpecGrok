import os
from typing import Optional, List
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
    QDRANT_PORT: int = 16333
    QDRANT_URL: Optional[str] = Field(default=None) # Optional if user wants to provide full URL
    QDRANT_COLLECTION_NAME: str = "specgrok_docs"
    
    # Model Settings
    LLM_MODEL_NAME: str = "deepseek-v3.2"
    VLM_MODEL_NAME: str = "qwen3-vl-plus"  # qwen-vl-max、qwen3-vl-plus、qwen3-vl-flash
    # Logical name (path resolution is handled in embedding.py)
    EMBEDDING_MODEL_NAME: str = "models/bge-m3-onnx"  # ["models/bge-m3", "models/bge-m3-onnx"]
    RERANKER_MODEL_NAME: str = "gte-rerank-v2"

    # Pix2Text model directory (relative to project root or absolute path)
    PIX2TEXT_HOME: str = Field("models/pix2text", validation_alias="PIX2TEXT_HOME")
    CACHE_DIR: str = Field("data/cache_vlm", validation_alias="CACHE_DIR")  # 用于存放图片和 VLM 结果缓存的目录。

    # Ingestion Pipeline Settings
    INGESTION_EXTENSIONS: List[str] = ['.txt', '.md', '.html'] 
    INGESTION_RECURSIVE: bool = True
    REMOVE_IMAGES: bool = True  # md、html parser setting
    REMOVE_LINKS: bool = False  # html parser setting
    PDF_VLM_ENABLED: bool = True  # pdf parser setting
    ENABLE_HYBRID: bool = True
    
    # Index Manager Settings
    # Auto reingest controls
    AUTO_REINGEST_MAX_BATCH: int = 50  # max docs to reingest in one auto call
    AUTO_REINGEST_DELAY_SECONDS: float = 0.0  # delay between reingests to avoid bursts

    # Text Splitting Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100

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
