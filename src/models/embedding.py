import logging
import os
from typing import Any, Dict, List, Optional, Union
from FlagEmbedding import BGEM3FlagModel
from src.config import settings
from src.models.backend_onnx import BGEM3OnnxBackend
from llama_index.core.embeddings import BaseEmbedding

# Configure logging
logger = logging.getLogger(__name__)

class BgeM3Service:
    _instance = None
    model: Optional[Union[BGEM3FlagModel, BGEM3OnnxBackend]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BgeM3Service, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        try:
            model_config_name = settings.EMBEDDING_MODEL_NAME
            logger.info(f"Initializing embedding model with config: {model_config_name}")
            
            # Set mirror if needed for FlagEmbedding internal checks
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # 1. Resolve path (Relative to project -> Absolute -> Original Name)
            resolved_path = os.path.join(project_root, model_config_name)
            
            final_path = model_config_name
            use_onnx = False
            
            # Check if resolved path exists (user provided relative path like 'models/bge-m3-onnx/model.onnx')
            if os.path.exists(resolved_path):
                final_path = resolved_path
            elif os.path.exists(model_config_name): # Absolute path provided
                final_path = model_config_name
                
            # 2. Check for ONNX signature
            if os.path.isfile(final_path) and final_path.endswith('.onnx'):
                use_onnx = True
            elif os.path.isdir(final_path) and os.path.exists(os.path.join(final_path, 'model.onnx')):
                final_path = os.path.join(final_path, 'model.onnx')
                use_onnx = True
                
            if use_onnx:
                logger.info(f"Detected ONNX model at {final_path}. Using ONNX backend.")
                tokenizer_path = os.path.dirname(final_path)
                self.model = BGEM3OnnxBackend(
                    model_path=final_path,
                    tokenizer_path=tokenizer_path,
                    use_fp16=False
                )
                logger.info("BGEM3OnnxBackend loaded successfully.")
                return

            # 3. Fallback to PyTorch (FlagEmbedding)
            
            # Special handling for default model name to prefer local cache legacy path
            if model_config_name == "BAAI/bge-m3":
                local_default_path = os.path.join(project_root, "models", "bge-m3")
                if os.path.exists(local_default_path) and os.path.isdir(local_default_path):
                    logger.info(f"Found local cache for {model_config_name} at {local_default_path}, using it.")
                    final_path = local_default_path
            
            logger.info(f"Loading BGEM3FlagModel: {final_path}...")
            self.model = BGEM3FlagModel(
                final_path, 
                use_fp16=False 
            )
            logger.info("BGEM3FlagModel loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Embedding Model: {e}")
            raise e

    def encode(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Encode text(s) using BGE-M3 model.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Dict with 'dense_vecs' and optionally 'lexical_weights' for hybrid search
        """
        if isinstance(text, str):
            text = [text]
            
        result = self.model.encode(
            text, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        
        return {
            'dense_vecs': result['dense_vecs'],
            'lexical_weights': result['lexical_weights']
        }

    @staticmethod
    def get_sparse_embedding_adapter(texts: List[str]) -> Any:
        """
        Adapter for LlamaIndex QdrantVectorStore 'sparse_doc_fn'.
        Converts BGE-M3's lexical weights (dict) to Qdrant's sparse format (indices, values).
        
        Args:
            texts: List of strings to encode.
            
        Returns:
            Tuple(List[List[int]], List[List[float]]): Batch indices and values.
        """
        instance = BgeM3Service()
        if not texts:
            return ([], [])
            
        result = instance.encode(texts)
        
        batch_indices = []
        batch_values = []
        
        for d in result['lexical_weights']:
            # d is {str_token_id: weight}
            indices = [int(k) for k in d.keys()]
            values = [float(v) for v in d.values()]
            batch_indices.append(indices)
            batch_values.append(values)
            
        return (batch_indices, batch_values)


class BgeM3Embedding(BaseEmbedding):
    """LlamaIndex compatible embedding wrapper for BGE-M3."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, service: BgeM3Service):
        super().__init__()
        object.__setattr__(self, 'service', service)

    @classmethod
    def class_name(cls) -> str:
        return "BgeM3Embedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        result = self.service.encode([query])
        return result['dense_vecs'][0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get query embedding."""
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_query_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get text embedding."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings."""
        result = self.service.encode(texts)
        return result['dense_vecs']

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async get multiple text embeddings."""
        return self._get_text_embeddings(texts)

    # Compatibility API: provide encode(...) similar to BgeM3Service
    def encode(self, texts: Union[str, List[str]]) -> dict:
        """Compatibility wrapper: delegate to underlying service.encode and normalize single-input response.

        Returns:
            If input is a string, returns {'dense_vecs': List[float], 'lexical_weights': Dict[str, float]}
            If input is a list, returns {'dense_vecs': List[List[float]], 'lexical_weights': List[Dict[str,float]]}
        """
        result = self.service.encode(texts)
        dense = result.get('dense_vecs')
        lexical = result.get('lexical_weights')
        if isinstance(texts, str):
            # single input: unwrap the first element
            return {
                'dense_vecs': dense[0] if isinstance(dense, list) and len(dense) > 0 else dense,
                'lexical_weights': lexical[0] if isinstance(lexical, list) and len(lexical) > 0 else lexical,
            }
        return result

def get_embed_model() -> BgeM3Embedding:
    """
    Returns a LlamaIndex compatible embedding model using BGE-M3.
    """
    service = BgeM3Service()
    return BgeM3Embedding(service)
