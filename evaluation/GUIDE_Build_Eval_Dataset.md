# LlamaIndex 评估数据集构建指南 (Evaluation Dataset Construction Guide)

本指南旨在指导如何基于现有文档数据，构建一套标准化的“黄金数据集”（Golden Dataset）。该数据集将包含“问题-参考文档ID”对（Question-Context Pairs），用于后续的检索评估（Retrieval Eval）和生成评估（Response Eval）。

核心原则
评估数据集永远以「Chunk / Node」为最小事实单元
不直接评估“原始文档”，而评估 Parser + Chunker + Retriever + Generator 的整体效果

---

## 📋 任务列表 (Task List)

### Phase 1: 环境与数据准备
- [ ] **确定 LLM**: 选择一个高质量的模型（如 GPT-4 或 DeepSeek-V3）用于生成问题。*注意：生成数据集的模型最好优于或等同于 RAG 运行时使用的模型。*
- [ ] **加载文档 (Nodes)**: 准备好经过清洗、切分后的 `List[BaseNode]`。
    - *提示*: 可以复用您项目中 `ingestion` 管道生成的 Nodes。

### Phase 2: 数据集生成 (核心)
- [ ] **生成 QA 对**: 使用 `generate_question_context_pairs` 函数自动合成问题。
- [ ] **配置参数**: 
    - `num_questions_per_chunk`: 建议设置为 1 或 2（每个切片生成的问题数）。
    - `llm`: 传入配置好的 LLM 实例。
- [ ] **过滤与清洗 (可选)**: 人工或通过脚本检查生成的问题，剔除含糊不清或过于简单的问题。

### Phase 3: 持久化与加载
- [ ] **保存数据集**: 将生成的对象保存为 JSON 文件（如 `eval_dataset_v1.json`）。
- [ ] **验证加载**: 编写测试代码确保能通过 `EmbeddingQAFinetuneDataset` 类正确读取。

### Phase 4: 集成评估
- [ ] **对接检索评估**: 使用 `RetrieverEvaluator` 测试 Hit Rate 和 MRR。
- [ ] **对接生成评估**: 提取数据集中的 Query，使用 `BatchEvalRunner` 测试 Faithfulness 和 Relevancy。

---

## 🛠️ 详细实施代码参考

### 1. 生成并保存数据集 (Generate & Save)

此脚本用于从 Nodes 自动生成 QA 数据集。

```python
import os
import asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.openai import OpenAI
# from models.llm import get_llm  # 如果你有自定义的 LLM 获取方式

async def build_golden_dataset(nodes, output_path="data/golden_dataset.json"):
    """
    输入: nodes (List[BaseNode])
    输出: 保存 JSON 文件
    """
    # 1. 配置生成用 LLM (建议使用能力较强的模型以保证问题质量)
    # llm = get_llm() 
    llm = OpenAI(model="gpt-4", temperature=0.0)

    print(f"正在基于 {len(nodes)} 个节点生成 QA 对...这可能需要一些时间。")

    # 2. 生成核心逻辑
    # generate_question_context_pairs 会返回一个 EmbeddingQAFinetuneDataset 对象
    # 它包含: queries (问题), relevant_docs (问题对应的 node_id), corpus (所有节点的文本)
    dataset = generate_question_context_pairs(
        nodes,
        llm=llm,
        num_questions_per_chunk=1,  # 每个 chunk 生成 1 个问题，避免问题重复
    )

    # 3. 持久化保存
    # 这一步非常重要，确保评估基准固定，方便后续对比不同 Retriever 的效果
    dataset.save_json(output_path)
    print(f"✅ 数据集已保存至: {output_path}")
    
    # 打印示例
    first_query_id = list(dataset.queries.keys())[0]
    print(f"示例 Question: {dataset.queries[first_query_id]}")
    print(f"关联 Node ID: {dataset.relevant_docs[first_query_id]}")

# 运行示例
if __name__ == "__main__":
    # 假设你已经有了 nodes，如果没有，临时加载：
    # reader = SimpleDirectoryReader("./data/raw")
    # documents = reader.load_data()
    # nodes = ... (执行你的 Chunking 逻辑)
    
    # asyncio.run(build_golden_dataset(nodes))
    pass
```

### 2.加载数据集进行检索评估 (Load & Eval Retrieval)
此脚本展示如何读取刚才保存的 JSON，并对当前的 Retriever 进行打分。

```python
import asyncio
import pandas as pd
from llama_index.core.evaluation import RetrieverEvaluator, EmbeddingQAFinetuneDataset
from llama_index.core import VectorStoreIndex

# 假设你已经构建好了 index
# from src.database.client import get_index 

async def run_retrieval_eval(dataset_path="data/golden_dataset.json"):
    # 1. 加载黄金数据集
    print(f"正在加载数据集: {dataset_path}")
    dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)

    # 2. 准备 Retriever (待评估对象)
    # index = get_index()
    # retriever = index.as_retriever(similarity_top_k=5)
    retriever = ... # 初始化你的 retriever

    # 3. 定义评估器
    # hit_rate: 正确答案是否在 top_k 中
    # mrr: 正确答案排名的倒数 (Mean Reciprocal Rank)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )

    # 4. 批量运行评估 (aevaluate_dataset 是专门针对 EmbeddingQAFinetuneDataset 的优化方法)
    print("开始运行批量评估...")
    eval_results = await retriever_evaluator.aevaluate_dataset(dataset)

    # 5. 展示结果
    metric_dicts = []
    for eval_result in eval_results:
        metric_dicts.append(eval_result.metric_vals_dict)

    df = pd.DataFrame(metric_dicts)
    print("\n------------------ 评估报告 ------------------")
    print(f"平均 Hit Rate: {df['hit_rate'].mean():.4f}")
    print(f"平均 MRR:      {df['mrr'].mean():.4f}")
    print("---------------------------------------------")

if __name__ == "__main__":
    # asyncio.run(run_retrieval_eval())
    pass
```

### 3.用于生成评估 (Response Eval)
虽然 generate_question_context_pairs 主要用于检索评估，但生成的 Questions 列表同样可以直接用于生成评估。

```python
from llama_index.core.evaluation import BatchEvalRunner, FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

async def run_response_eval(dataset_path="data/golden_dataset.json", query_engine=None):
    # 1. 加载数据集
    dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)
    questions = list(dataset.queries.values()) # 提取所有问题列表

    # 2. 定义评估器 (Faithfulness & Relevancy)
    # 这里的 llm 充当“裁判”
    # evaluator_llm = OpenAI(model="gpt-4")
    # faithfulness = FaithfulnessEvaluator(llm=evaluator_llm)
    # relevancy = RelevancyEvaluator(llm=evaluator_llm)

    # 3. 批量运行
    # runner = BatchEvalRunner(
    #    {"faithfulness": faithfulness, "relevancy": relevancy},
    #    workers=8
    # )
    
    # 4. 执行评估
    # eval_results = await runner.aevaluate_queries(
    #    query_engine, queries=questions
    # )
    pass
```

#### 注意事项
1.数据隔离: 确保用于评估的文档已经包含在你的 Vector Store (Index) 中，否则 Retrieval Eval 的 Hit Rate 将永远为 0。
2.成本控制: generate_question_context_pairs 会对每个 chunk 调用一次 LLM。如果你有 4000 个文档，全部生成可能成本较高。建议先采样 50-100 个代表性文档构建一个小型的 v0.1 数据集。
3.Human-in-the-loop: 自动生成的 QA 对可能偶尔包含幻觉或指代不明（例如问“本文的作者是谁？”但上下文中没有名字）。建议在保存 json 后，人工快速浏览一遍 queries，删除低质量问题。