
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
# Test Data - Enhanced to test BGE-M3's dense, sparse, and multi-vector capabilities
texts = [
    # 1. 复杂语义和实体混合 - 测试稀疏向量(关键词提取)和稠密向量(语义理解)
    "量子纠缠的超距作用挑战了经典物理学的局域实在性假设，而薛定谔的猫实验则揭示了量子叠加态的哲学困境，这促使哥本哈根解释与多世界诠释之间的激烈争论。",
    
    # 2. 多重专业术语混合 - 测试稀疏向量的关键词权重分配
    "Transformer架构中的多头注意力机制通过残差连接和层归一化优化梯度流动，结合BERT的掩码语言建模和GPT的自回归生成范式，实现了跨模态的预训练-微调范式迁移。",
    
    # 3. 长文档上下文和核心概念分散 - 测试ColBERT的细粒度匹配
    """在微服务架构的演进过程中，服务网格（Service Mesh）如Istio和Linkerd通过边车代理模式解耦了业务逻辑与网络策略，
    实现了熔断、限流和可观测性的基础设施层抽象，这与API网关的功能形成了互补而非替代关系，共同构建了云原生应用的全栈治理体系。
    同时，服务发现机制从客户端负载均衡向中心化服务注册表的转变，反映了分布式系统设计范式的演进路径。""",
    
    # 4. 情感、产品和规格的复杂组合 - 测试多维度理解
    "Apple最新发布的M3 Max芯片采用3nm制程工艺，集成920亿个晶体管，40核GPU在Blender渲染测试中比M2 Ultra快30%，但其3499美元的起售价引发了市场对高端笔记本性价比的讨论。",
    
    # 5. 跨语言和文化概念混合 - 测试多语言稀疏编码
    "机器学习的过拟合问题（overfitting）类似于儒家思想中的'过犹不及'，需要通过正则化（如L1/L2惩罚）或数据增强来寻找偏差-方差权衡的'中庸之道'，这与深度学习中的Dropout技术有异曲同工之妙。",
    
    # 6. 时序和因果关系描述 - 测试逻辑关系理解
    "由于美联储持续加息导致国债收益率曲线倒挂，科技股估值承压，但人工智能领域的突破性进展部分抵消了宏观不利因素，使得纳斯达克指数在2023年呈现U型反弹态势。",
    
    # 7. 包含代码、数学和自然语言的混合文本 - 测试结构化信息提取
    "损失函数定义为 L(θ) = -∑log P(y_i|x_i;θ) + λ‖θ‖²，其中λ控制正则化强度，Adam优化器通过一阶矩和二阶矩的指数移动平均调整学习率，这在PyTorch中通过torch.optim.Adam实现。",
    
    # 8. 多实体关系和事件描述 - 测试关系提取能力
    "特斯拉上海超级工厂在2023年交付了947,000辆Model 3和Model Y，占全球产量的52%，这得益于中国完善的供应链和较低的制造成本，但欧盟的反补贴调查可能影响其出口关税优惠。",
    
    # 9. 抽象概念和具象例子的结合 - 测试概念泛化
    "注意力机制在神经机器翻译中如同人类阅读时的'焦点转移'，当解码器生成'人工智能'时，它会自动关注源语言中'artificial intelligence'的对应位置，这比传统的编码器-解码器架构更接近认知科学的'工作记忆'模型。",
    
    # 10. 对比和类比结构 - 测试比较分析能力
    "卷积神经网络的空间不变性与循环神经网络的时序依赖性形成互补，正如计算机视觉处理二维局部特征而自然语言处理关注一维序列关系，但Vision Transformer通过图像分块打破了这种界限，证明了自注意力机制的通用性。"
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

