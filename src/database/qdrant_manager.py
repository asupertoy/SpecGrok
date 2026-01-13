import logging
from typing import Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings

logger = logging.getLogger(__name__)

class QdrantManager:
    _instance: Optional["QdrantManager"] = None
    client: QdrantClient

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QdrantManager, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        try:
            self.client = QdrantClient(
                url=settings.qdrant_effective_url,
                # api_key=settings.QDRANT_API_KEY  # Uncomment if API Key is needed
            )
            logger.info(f"Connected to Qdrant at {settings.qdrant_effective_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise e

    def get_client(self) -> QdrantClient:
        return self.client

    def create_collection_if_not_exists(self):
        """
        Creates the collection with Hybrid Search support (Dense + Sparse).
        BGE-M3 Dense Dimension: 1024
        """
        collection_name = settings.QDRANT_COLLECTION_NAME
        
        if self.client.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists.")
            return

        logger.info(f"Creating collection '{collection_name}' with hybrid search config...")
        
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1024,  # BGE-M3 dense dimension
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )
            logger.info(f"Successfully created collection '{collection_name}'.")
        except UnexpectedResponse as e:
            logger.error(f"Failed to create collection: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred while creating collection: {e}")
            raise e

# Global instance
qdrant_manager = QdrantManager()