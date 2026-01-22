import logging
import os
from typing import Any, Dict, List, Optional, Union
from FlagEmbedding import BGEM3FlagModel
from src.config import settings
from src.models.backend_onnx import BGEM3OnnxBackend

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
        Returns a dictionary with 'dense_vecs', 'lexical_weights' (sparse), etc.
        """
        if self.model is None:
            self._init_model()
        
        # return_dense=True, return_sparse=True, return_colbert_vecs=False
        return self.model.encode(
            text, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )

# Global instance getter
_service_instance: Optional[BgeM3Service] = None

def get_embed_model() -> BgeM3Service:
    """
    Returns a singleton instance of the BgeM3Service (wrapper around BGEM3FlagModel).
    NOTE: This is NOT a LlamaIndex embedding model.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = BgeM3Service()
    return _service_instance
