
import os
import torch
import torch.nn as nn
from FlagEmbedding import BGEM3FlagModel

# Configure
project_root = "/home/john/project/SpecGrok"
model_path = os.path.join(project_root, "models", "bge-m3")
output_path = os.path.join(project_root, "models", "bge-m3-onnx")
onnx_file = os.path.join(output_path, "model.onnx")

os.makedirs(output_path, exist_ok=True)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print(f"Loading model from {model_path}...")
flag_model = BGEM3FlagModel(model_path, use_fp16=False)
flag_model.model.eval()

class BgeM3ExportWrapper(nn.Module):
    def __init__(self, flag_model):
        super().__init__()
        # Access the internal XLMRobertaModel
        self.transformer = flag_model.model.model
        self.sparse_linear = flag_model.model.sparse_linear
        self.colbert_linear = flag_model.model.colbert_linear
        
    def forward(self, input_ids, attention_mask):
        # Run transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # 1. Dense (CLS token)
        # Note: BGE-M3 uses CLS token for dense embedding
        dense_vecs = last_hidden_state[:, 0]
        # Normalize
        dense_vecs = torch.nn.functional.normalize(dense_vecs, p=2, dim=1)
        
        # 2. Sparse (Linear -> ReLU)
        # Based on BGE-M3 papers/code, sparse weights are ReLU(linear(hidden))
        sparse_logits = self.sparse_linear(last_hidden_state)
        sparse_weights = torch.relu(sparse_logits)
        
        # 3. ColBERT (Linear -> Normalize)
        colbert_vecs = self.colbert_linear(last_hidden_state)
        colbert_vecs = torch.nn.functional.normalize(colbert_vecs, p=2, dim=2)
        
        return dense_vecs, sparse_weights, colbert_vecs

print("Creating wrapper...")
model_wrapper = BgeM3ExportWrapper(flag_model)
model_wrapper.eval()

# Dummy input
print("Creating dummy input...")
tokenizer = flag_model.tokenizer
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Dynamic axes
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "dense_vecs": {0: "batch_size"},
    "sparse_weights": {0: "batch_size", 1: "sequence_length"},
    "colbert_vecs": {0: "batch_size", 1: "sequence_length"}
}

print(f"Exporting to {onnx_file}...")
try:
    torch.onnx.export(
        model_wrapper,
        (input_ids, attention_mask),
        onnx_file,
        input_names=["input_ids", "attention_mask"],
        output_names=["dense_vecs", "sparse_weights", "colbert_vecs"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True
    )
    print("Export successful!")
    
    # Save tokenizer as well (copy files)
    print("Copying tokenizer files...")
    import shutil
    for file in os.listdir(model_path):
        if "token" in file or "config" in file or "spiece" in file:
            src = os.path.join(model_path, file)
            dst = os.path.join(output_path, file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    print("Tokenizer files copied.")
    
except Exception as e:
    print(f"Export failed: {e}")
    import traceback
    traceback.print_exc()

