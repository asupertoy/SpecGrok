
import os
import time
import numpy as np
import torch
import onnxruntime as ort
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

# Configure
project_root = "/home/john/project/SpecGrok"
model_path = os.path.join(project_root, "models", "bge-m3")
onnx_path = os.path.join(project_root, "models", "bge-m3-onnx", "model.onnx")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. Load Original Model
print("Loading Original Pytorch Model...")
flag_model = BGEM3FlagModel(model_path, use_fp16=False)

# 2. Load ONNX Model
print(f"Loading ONNX Model from {onnx_path}...")
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])

# Test Data
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "BGE M3 is an embedding model supporting dense, sparse and colbert retrieval.",
    "奇迹行者，还在刷野，去看一眼呀！",
    "今天天气不错，适合出去游玩。",
    "Technical documentation regarding the SpecGrok project."
]

print(f"\nRunning inference on {len(texts)} texts...")

# --- Pytorch Inference ---
start_time = time.time()
pt_output = flag_model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)
pt_time = time.time() - start_time
print(f"Pytorch Time: {pt_time:.4f}s")


# --- ONNX Inference ---
tokenizer = flag_model.tokenizer

def onnx_encode(texts):
    # Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np", max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Run
    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    # Run inference
    dense, sparse, colbert = ort_session.run(None, onnx_inputs)
    
    return dense, sparse, colbert, input_ids

start_time = time.time()
onnx_dense, onnx_sparse, onnx_colbert, input_ids = onnx_encode(texts)
onnx_time = time.time() - start_time
print(f"ONNX Time:    {onnx_time:.4f}s")
print(f"Speedup:      {pt_time / onnx_time:.2f}x")

# --- Verification ---

# 1. Dense Check
# pt_output['dense_vecs'] is numpy array
diff = np.abs(pt_output['dense_vecs'] - onnx_dense)
max_diff = np.max(diff)
print(f"\n[Dense] Max difference: {max_diff:.6f}")
if max_diff < 1e-4:
    print("✅ Dense vectors match.")
else:
    print("❌ Dense vectors mismatch!")

# 2. Sparse Check
# pt_output['lexical_weights'] is list of dicts
# onnx_sparse is (batch, seq_len, 1) weights
# We need to reconstruct the dict from onnx_sparse
print("\n[Sparse] Checking first sample...")
pt_sparse_0 = pt_output['lexical_weights'][0]

# Construct ONNX sparse dict for first sample
onnx_sparse_0 = {}
for i, token_id in enumerate(input_ids[0]):
    if token_id in iter(tokenizer.all_special_ids):
        continue
    weight = float(onnx_sparse[0][i][0])
    if weight > 0:
        # Use simple string of token ID
        str_token = str(token_id)
        if str_token in onnx_sparse_0:
            onnx_sparse_0[str_token] = max(onnx_sparse_0[str_token], weight)
        else:
            onnx_sparse_0[str_token] = weight

# Compare keys
print(f"PT keys: {list(pt_sparse_0.keys())[:5]}")
print(f"ONNX keys: {list(onnx_sparse_0.keys())[:5]}")

pt_keys = set(pt_sparse_0.keys())
onnx_keys = set(onnx_sparse_0.keys())
common = pt_keys.intersection(onnx_keys)
print(f"Keys overlap: {len(common)} / {len(pt_keys)}")

# Compare values for common keys
diffs = []
for k in common:
    diffs.append(abs(pt_sparse_0[k] - onnx_sparse_0[k]))
if diffs:
    print(f"Avg weight diff: {np.mean(diffs):.6f}")

if len(common) == len(pt_keys) and (not diffs or np.mean(diffs) < 1e-4):
    print("✅ Sparse weights match.")
else:
    print("⚠️ Sparse weights might follow different logic or simple float error.")


# 3. ColBERT Check
print("\n[ColBERT] Checking first sample...")
pt_colbert_0 = pt_output['colbert_vecs'][0]  # List of vectors for valid tokens only

# ONNX output includes padding. We need to mask it using the attention mask.
# input_ids shape: (batch, seq_len)
# onnx_colbert shape: (batch, seq_len, 1024)
attn_mask_0 = input_ids[0] != tokenizer.pad_token_id # Or use attention_mask from iterator
valid_len = np.sum(attn_mask_0)

# Slice ONNX output to valid length
# Hypothesis: BGE-M3 excludes CLS token (index 0) from ColBERT vectors
onnx_colbert_0 = onnx_colbert[0][1:valid_len]

print(f"PT shape: {pt_colbert_0.shape}")
print(f"ONNX shape (valid): {onnx_colbert_0.shape} (raw was {onnx_colbert[0].shape})")

if pt_colbert_0.shape == onnx_colbert_0.shape:
    diff_c = np.abs(pt_colbert_0 - onnx_colbert_0)
    max_diff_c = np.max(diff_c)
    print(f"Max diff: {max_diff_c:.6f}")
    if max_diff_c < 1e-3: # FP16/FP32 conversion tolerance
        print("✅ ColBERT vectors match.")
    else:
        print("❌ ColBERT vectors mismatch (value diff)!")
else:
    print("❌ ColBERT vectors mismatch (shape diff)!")

