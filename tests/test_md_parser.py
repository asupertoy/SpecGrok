
import os
import sys
from pathlib import Path

# Add src to path
# Assumes the script is located at project/SpecGrok/tests/test_md_parser.py
current_dir = Path(__file__).resolve().parent
# Go up to SpecGrok root then down to src
sys.path.append(str(current_dir.parent / "src"))

from src.ingestion.parsers.parser_md import MarkdownParser
from src.ingestion.loaders import Blob

def test_markdown_parser(save_results=True):
    """测试复杂的Markdown解析，包括多种元素和嵌套结构"""
    # Construct a complex markdown content
    md_content = """
---
title: "复杂Markdown文档测试"
author: "测试作者"
date: "2024-01-22"
tags: ["测试", "Markdown", "解析"]
---

# 引言

这是文档的**引言部分**，包含*斜体*和`内联代码`。

## 背景介绍

### 历史发展

Markdown于2004年由John Gruber创建，旨在提供一种易读易写的纯文本格式。

#### 版本演进

- **Markdown 1.0**: 最初版本
- **CommonMark**: 标准化规范
- **GitHub Flavored Markdown**: 扩展版本

### 主要特性

Markdown支持：

1. **标题层级** (H1-H6)
2. **文本格式化**：
   - *斜体*
   - **粗体**
   - `代码`
   - ~~删除线~~
3. **列表**：
   - 无序列表
   - 有序列表
   - 嵌套列表

## 代码示例

### Python代码块

```python
# 这是一个复杂的Python函数
def complex_function(data: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
    \"\"\"复杂的函数处理数据\"\"\"
    # 过滤数据
    filtered = [item for item in data if item.get('score', 0) > threshold]

    # 统计分析
    stats = {
        'total': len(filtered),
        'average_score': sum(item['score'] for item in filtered) / len(filtered) if filtered else 0,
        'categories': {}
    }

    # 分类统计
    for item in filtered:
        cat = item.get('category', 'unknown')
        stats['categories'][cat] = stats['categories'].get(cat, 0) + 1

    # ## 这不是标题，只是在代码注释中
    return stats

# 调用函数
result = complex_function(sample_data)
print(f"处理结果: {result}")
```

### JavaScript代码块

```javascript
// 复杂的JavaScript函数
class DataProcessor {
    constructor(config) {
        this.config = config;
        // ## 这也不是标题
    }

    async process(data) {
        try {
            // 数据验证
            if (!Array.isArray(data)) {
                throw new Error('数据必须是数组');
            }

            // 并行处理
            const promises = data.map(async (item) => {
                const processed = await this.transform(item);
                return this.validate(processed);
            });

            return await Promise.all(promises);
        } catch (error) {
            console.error('处理失败:', error);
            throw error;
        }
    }

    // ## 私有方法
    transform(item) {
        return {
            ...item,
            processed_at: new Date().toISOString(),
            hash: this.generateHash(item)
        };
    }
}
```

## 数学公式

### 基本公式

内联公式：$E = mc^2$ 是爱因斯坦的质能方程。

### 复杂公式块

$$
\\frac{d}{dx} \\int_a^x f(t) \\, dt = f(x)
$$

$$
\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1
$$

### 矩阵和方程组

$$
\\begin{pmatrix}
a & b \\\\
c & d
\\end{pmatrix}
\\begin{pmatrix}
x \\\\
y
\\end{pmatrix}
=
\\begin{pmatrix}
ax + by \\\\
cx + dy
\\end{pmatrix}
$$

### 多行方程

$$
\\begin{align}
\\nabla \\cdot \\mathbf{E} &= \\frac{\\rho}{\\epsilon_0} \\\\
\\nabla \\cdot \\mathbf{B} &= 0 \\\\
\\nabla \\times \\mathbf{E} &= -\\frac{\\partial \\mathbf{B}}{\\partial t} \\\\
\\nabla \\times \\mathbf{B} &= \\mu_0 \\mathbf{J} + \\mu_0 \\epsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t}
\\end{align}
$$

<!-- 这是一个HTML注释，应该被忽略 -->

## 数据表格

### 基本表格

| 姓名 | 年龄 | 职业 | 薪资 |
|------|------|------|------|
| 张三 | 28 | 工程师 | ¥12000 |
| 李四 | 32 | 设计师 | ¥10000 |
| 王五 | 25 | 产品经理 | ¥15000 |

### 复杂表格（包含Markdown语法）

| 功能 | 描述 | 示例 | 状态 |
|------|------|------|------|
| **标题** | 支持多级标题 | # H1<br>## H2<br>### H3 | ✅ |
| *格式化* | 文本样式 | **粗体**<br>*斜体*<br>`代码` | ✅ |
| 链接 | 外部链接 | [Google](https://google.com)<br>[内部](#section) | ✅ |
| 图片 | 图片显示 | ![Logo](https://example.com/logo.png) | ✅ |
| 列表 | 嵌套列表 | - 项目1<br>  - 子项目<br>- 项目2 | ✅ |
| # 标题标记 | 表格中的标题 | # 这不是标题 | ✅ |

### 跨行表格

| 项目 | 说明 | 状态 | 备注 |
|------|------|------|------|
| 数据处理 | 实现复杂的数据处理逻辑 | 完成 | 支持多格式输入 |
| 用户界面 | 设计直观的用户界面 | 进行中 | 使用现代UI框架 |
| API集成 | 与第三方API集成 | 待开始 | 需要API密钥 |
| 测试覆盖 | 编写全面的单元测试 | 完成 | 覆盖率95% |

## 图片和媒体

### 图片示例

![系统架构图](https://example.com/architecture.png "系统架构图")

![流程图](https://example.com/flowchart.svg "业务流程图")

### 图片引用

参考上方的[系统架构图](#系统架构图)和[流程图](#流程图)。

## 引用和注释

### 引用块

> 这是一个简单的引用块。

> **嵌套引用**
>
> > 这是一个嵌套的引用块。
> >
> > 包含多行内容。

### 文献引用

根据Smith等人的研究[^1]，Markdown具有以下优势：

- 易读性强
- 转换灵活
- 版本控制友好

## 特殊内容

### HTML内嵌

<div class="warning">
<strong>警告：</strong>以下内容包含HTML标记。
</div>

### 转义字符

特殊字符：\\* \\* \\` \\` \\[ \\[ \\] \\] \\( \\( \\) \\)

### 水平分割线

---

## 结论

### 总结

本文档演示了Markdown的各种复杂特性：

1. **多级标题嵌套**
2. **多种代码块**
3. **复杂数学公式**
4. **丰富的表格格式**
5. **图片和链接**
6. **引用和注释**

### 未来展望

未来将继续扩展Markdown的功能，支持更多现代文档需求。

---

[^1]: Smith, J. (2020). Markdown: A Writing Tool for the Digital Age. Journal of Digital Writing, 15(2), 45-67.

<!-- 文档结束 -->
    """.strip()

    # Create a Blob
    blob = Blob(
        data=md_content.encode("utf-8"),
        source="complex_test_doc.md"
    )

    parser = MarkdownParser(remove_images=True)
    nodes = parser.parse(blob)

    print(f"Total Nodes: {len(nodes)}")

    # 验证基本结构
    assert len(nodes) >= 8, f"期望至少8个节点，实际{len(nodes)}"

    # 检查标题层级
    headers = [node.metadata.get('section_header') for node in nodes]
    expected_headers = [
        "引言",
        "背景介绍",
        "历史发展",
        "版本演进",
        "主要特性",
        "代码示例",
        "Python代码块",
        "JavaScript代码块",
        "数学公式",
        "基本公式",
        "复杂公式块",
        "矩阵和方程组",
        "多行方程",
        "数据表格",
        "基本表格",
        "复杂表格（包含Markdown语法）",
        "跨行表格",
        "图片和媒体",
        "图片示例",
        "图片引用",
        "引用和注释",
        "引用块",
        "文献引用",
        "特殊内容",
        "HTML内嵌",
        "转义字符",
        "水平分割线",
        "结论",
        "总结",
        "未来展望"
    ]

    # 检查是否包含主要标题
    for expected in ["引言", "代码示例", "数学公式", "数据表格", "结论"]:
        assert expected in headers, f"缺少标题: {expected}"

    # 检查section_path层级
    paths = [node.metadata.get('section_path') for node in nodes]
    print(f"Section Paths: {paths[:5]}...")  # 只打印前5个

    # 检查内容包含
    all_content = "\n".join([node.text for node in nodes])

    # 检查代码块是否保留
    assert "def complex_function" in all_content, "Python代码块应该保留"
    assert "class DataProcessor" in all_content, "JavaScript代码块应该保留"

    # 检查数学公式是否保留
    assert "E = mc^2" in all_content, "数学公式应该保留"
    assert "\\frac{d}{dx}" in all_content, "复杂公式应该保留"

    # 检查表格是否保留
    assert "|" in all_content, "表格应该保留"

    # 检查图片是否被移除（因为remove_images=True）
    assert "![" not in all_content, "图片标记应该被移除"

    # 检查图片是否提取到metadata
    all_images = []
    for node in nodes:
        all_images.extend(node.metadata.get('images', []))
    assert len(all_images) >= 3, f"应该提取到至少3个图片，实际{len(all_images)}"

    # 检查YAML frontmatter是否被移除（frontmatter被移除，但水平分割线保留）
    # assert "---" not in all_content, "YAML frontmatter应该被移除"

    # 检查HTML注释是否被移除
    assert "<!--" not in all_content, "HTML注释应该被移除"

    print("✓ 复杂Markdown解析测试通过")

    # 可选保存结果到 txt 文件
    if save_results:
        output_dir = current_dir / "test_output_files"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "md_parser_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Markdown 解析测试结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"总节点数: {len(nodes)}\n\n")
            for i, node in enumerate(nodes):
                f.write(f"节点 {i}:\n")
                f.write(f"  文本: {node.text}\n")
                f.write(f"  元数据: {node.metadata}\n")
                f.write("\n")
        print(f"测试结果已保存到: {output_file}")

    # 详细输出前几个节点
    for i, node in enumerate(nodes[:5]):
        print(f"\n--- Node {i} ---")
        print(f"Header: {node.metadata.get('section_header')}")
        print(f"Path: {node.metadata.get('section_path')}")
        print(f"Level: {node.metadata.get('level')}")
        print(f"Images: {node.metadata.get('images')}")
        print(f"Content Preview:\n{node.text[:150]}...")

if __name__ == "__main__":
    test_markdown_parser(save_results=True)
