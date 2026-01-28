import logging
import os
from typing import Any, Dict, List, Optional, Union
from FlagEmbedding import BGEM3FlagModel
from config import settings
from .backend_onnx import BGEM3OnnxBackend

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

    def get_text_embedding(self, text: str) -> List[float]:
        """Return a single dense embedding vector."""
        result = self.encode(text)
        dense_vecs = result.get("dense_vecs")

        if dense_vecs is None:
            return []

        # normalize various return shapes: list, numpy array, list of lists
        first = dense_vecs[0] if isinstance(dense_vecs, (list, tuple)) else dense_vecs
        if isinstance(first, (list, tuple)):
            return list(first)
        return list(dense_vecs)

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch helper matching LlamaIndex expectations."""
        if not texts:
            return []

        result = self.encode(texts)
        dense_vecs = result.get("dense_vecs")

        if dense_vecs is None:
            return []

        # If a single vector is returned for batch input, wrap it.
        if isinstance(dense_vecs, (list, tuple)) and dense_vecs and not isinstance(dense_vecs[0], (list, tuple)):
            return [list(dense_vecs)]

        # Convert nested arrays/tuples to plain lists
        return [list(v) if isinstance(v, (list, tuple)) else list(v.tolist()) if hasattr(v, "tolist") else [] for v in dense_vecs]

    # Expose sparse adapters as convenience staticmethods on the service
    @staticmethod
    def sparse_doc_fn(texts: List[str]) -> tuple:
        """Convenience wrapper to expose module-level `sparse_doc_fn` via the service."""
        return sparse_doc_fn(texts)

    @staticmethod
    def sparse_query_fn(text: Union[str, List[str]]) -> tuple:
        """Convenience wrapper to expose module-level `sparse_query_fn` via the service."""
        return sparse_query_fn(text)

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


# -----------------------------
# Sparse Adapter for Qdrant
# -----------------------------

def _normalize_lexical_item(item) -> List[tuple]:
    """Normalize one lexical_weights item into list of (idx:int, value:float).

    Accepts formats like:
      - dict mapping int/str -> float
      - list of (idx, val) tuples
      - list of dicts
    Returns: list of (int, float)
    """
    pairs = []

    # dict mapping
    if isinstance(item, dict):
        for k, v in item.items():
            try:
                idx = int(k)
            except Exception:
                # if key is not numeric, skip it
                continue
            try:
                val = float(v)
            except Exception:
                continue
            if val != 0.0:
                pairs.append((idx, val))

    # list of tuples or lists
    elif isinstance(item, (list, tuple)):
        for elem in item:
            if isinstance(elem, (list, tuple)) and len(elem) >= 2:
                try:
                    idx = int(elem[0])
                    val = float(elem[1])
                except Exception:
                    continue
                if val != 0.0:
                    pairs.append((idx, val))

    return pairs


def sparse_doc_fn(texts: List[str]) -> tuple:
    """Adapter for LlamaIndex QdrantVectorStore 'sparse_doc_fn'.

    Accepts:
      - list[str]
      - list[Document/Node-like] (will attempt to extract text via .get_content or .text)

    Returns:
        (List[List[int]], List[List[float]])
    """
    if not texts:
        return ([], [])

    # Coerce inputs to strings when possible
    coerced_texts = []
    for t in texts:
        if isinstance(t, str):
            coerced_texts.append(t)
        else:
            # LlamaIndex may pass Document/Node objects
            try:
                if hasattr(t, "get_content"):
                    coerced_texts.append(t.get_content())
                    continue
            except Exception:
                pass
            try:
                if hasattr(t, "text"):
                    coerced_texts.append(t.text)
                    continue
            except Exception:
                pass
            coerced_texts.append(str(t))

    svc = get_embed_model()
    out = svc.encode(coerced_texts)

    lexical = out.get("lexical_weights") or out.get("sparse_values") or out.get("lexical")

    if lexical is None:
        # fallback: return empty lists for each doc
        return ([[] for _ in coerced_texts], [[] for _ in coerced_texts])

    # ensure list-like
    if isinstance(lexical, dict):
        lexical = [lexical]

    indices_list = []
    values_list = []

    for item in lexical:
        pairs = _normalize_lexical_item(item)
        if not pairs:
            indices_list.append([])
            values_list.append([])
            continue

        # sort by index for deterministic ordering (could also sort by weight)
        pairs.sort(key=lambda x: x[0])
        idxs, vals = zip(*pairs)
        indices_list.append(list(idxs))
        values_list.append(list(vals))

    return (indices_list, values_list)


def sparse_query_fn(text: Union[str, List[str]]) -> tuple:
    """Adapter for Qdrant sparse query.

    Accepts either a single string or a list of strings (batch). Always returns
    (List[List[int]], List[List[float]]) to match QdrantVectorStore expectations.
    """
    # If a single string is passed, wrap it to a list
    if text is None:
        return ([], [])

    if isinstance(text, str):
        return sparse_doc_fn([text])

    # If it's list-like, coerce elements then delegate
    try:
        texts = []
        for t in text:
            if isinstance(t, str):
                texts.append(t)
            else:
                try:
                    if hasattr(t, "get_content"):
                        texts.append(t.get_content())
                        continue
                except Exception:
                    pass
                try:
                    if hasattr(t, "text"):
                        texts.append(t.text)
                        continue
                except Exception:
                    pass
                texts.append(str(t))
        return sparse_doc_fn(texts)
    except Exception:
        # fallback: return empty lists for the single input
        return ([[]], [[]])

