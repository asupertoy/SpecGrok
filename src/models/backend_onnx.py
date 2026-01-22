
import os
import torch
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, List, Union, Optional
from transformers import AutoTokenizer

class BGEM3OnnxBackend:
    def __init__(self, model_path: str, tokenizer_path: str, use_fp16: bool = False, device: str = "cpu"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load ONNX Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        if device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
    def encode(self, 
               sentences: Union[str, List[str]], 
               return_dense: bool = True, 
               return_sparse: bool = False, 
               return_colbert_vecs: bool = False,
               batch_size: int = 12,
               max_length: int = 512) -> Dict[str, Any]:
        
        if isinstance(sentences, str):
            is_single = True
            sentences = [sentences]
        else:
            is_single = False
            
        all_dense = []
        all_sparse = []
        all_colbert = []
        
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts, 
                max_length=max_length, 
                padding=True, 
                truncation=True, 
                return_tensors="np"
            )
            
            input_ids = encoded_input['input_ids']
            attention_mask = encoded_input['attention_mask']
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            # Run inference
            # Outputs: dense_vecs, sparse_weights, colbert_vecs
            # Note: output names must match export script
            outputs = self.session.run(None, inputs)
            dense_vecs, sparse_weights, colbert_vecs = outputs
            
            if return_dense:
                all_dense.append(dense_vecs)
                
            if return_sparse:
                # Process sparse weights to dicts
                # sparse_weights: (batch, seq, 1)
                for j in range(len(batch_texts)):
                    # Get sequence length (ignoring padding locally if we want, but mask handles it)
                    # Actually mask is needed to filter special tokens?
                    # Using tokenizer.all_special_ids
                    seq_ids = input_ids[j] # numpy array
                    seq_weights = sparse_weights[j] # (seq, 1)
                    
                    sparse_dict = {}
                    for k, token_id in enumerate(seq_ids):
                        if token_id in self.tokenizer.all_special_ids:
                            continue
                        
                        weight = float(seq_weights[k][0])
                        if weight > 0:
                            # Use string of token ID to match FlagEmbedding default?
                            # Or decode?
                            # Existing FlagEmbedding (v1.3.5) on this machine returns IDs as strings.
                            str_token = str(token_id)
                            if str_token in sparse_dict:
                                sparse_dict[str_token] = max(sparse_dict[str_token], weight)
                            else:
                                sparse_dict[str_token] = weight
                    all_sparse.append(sparse_dict)
            
            if return_colbert_vecs:
                # FlagEmbedding returns list of arrays, usually without padding?
                # We will return the full padded array or list of arrays?
                # BGE-M3 `encode` returns list of np.arrays for colbert
                for j in range(len(batch_texts)):
                    # Extract active tokens based on mask? 
                    # Or just return expected length?
                    # Usually colbert vecs correspond to tokens.
                    # We'll just append the row for now.
                    # To be precise: we should remove padding.
                    # BGE-M3 Logic: Exclude CLS token (index 0) from ColBERT vectors
                    length = np.sum(attention_mask[j])
                    vec = colbert_vecs[j][1:length]
                    all_colbert.append(vec)

        result = {}
        if is_single:
            if return_dense:
                # Concatenate to handle possible batching logic if any (though single usually 1 batch)
                # But dense_vecs in loop is (batch, dim)
                # If single, we have one batch of size 1.
                # all_dense[0] is (1, 1024).
                result['dense_vecs'] = all_dense[0][0]
            if return_sparse:
                result['lexical_weights'] = all_sparse[0]
            if return_colbert_vecs:
                result['colbert_vecs'] = all_colbert[0]
        else:
            if return_dense:
                result['dense_vecs'] = np.concatenate(all_dense, axis=0)
                
            if return_sparse:
                result['lexical_weights'] = all_sparse
                
            if return_colbert_vecs:
                result['colbert_vecs'] = all_colbert
            
        return result
