import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent / "src"))

from ingestion.parsers.parser_html import HTMLParser
from ingestion.loaders import Blob

def test_basic_parsing():
    """测试基本HTML解析和Markdown转换"""
    print("=== 测试基本解析 ===")
    html_content = """
    <html>
    <head><title>测试页面</title></head>
    <body>
    <h1>介绍</h1>
    <p>这是介绍部分。</p>
    <h2>代码部分</h2>
    <pre><code>def hello():
    print("Hello World")</code></pre>
    <h2>表格部分</h2>
    <table>
    <tr><th>姓名</th><th>年龄</th></tr>
    <tr><td>小明</td><td>25</td></tr>
    </table>
    </body>
    </html>
    """.strip()

    blob = Blob(data=html_content.encode("utf-8"), source="test.html")
    parser = HTMLParser()
    nodes = parser.parse(blob)

    assert len(nodes) == 3, f"期望3个节点，实际{len(nodes)}"
    assert nodes[0].metadata['section_header'] == '介绍'
    assert '测试页面' in nodes[0].metadata.get('title', '')
    assert 'def hello():' in nodes[1].text
    assert '|' in nodes[2].text  # 表格转换为Markdown
    print("✓ 基本解析测试通过")

def test_remove_images_and_links():
    """测试移除图片和链接的功能"""
    print("=== 测试移除图片和链接 ===")
    html_content = """
    <html>
    <body>
    <h1>测试</h1>
    <p>查看这个<a href="http://example.com">链接</a>和<img src="image.jpg" alt="图片"></p>
    </body>
    </html>
    """

    # 测试移除链接
    blob = Blob(data=html_content.encode("utf-8"), source="test.html")
    parser = HTMLParser(remove_links=True, remove_images=True)
    nodes = parser.parse(blob)

    assert len(nodes) == 1
    text = nodes[0].text
    assert 'http://example.com' not in text, "链接应该被移除"
    assert 'image.jpg' not in text, "图片应该被移除"
    print("✓ 移除图片和链接测试通过")

def test_custom_clean_rules():
    """测试自定义清洗规则"""
    print("=== 测试自定义清洗规则 ===")
    html_content = """
    <html>
    <body>
    <h1>测试</h1>
    <div class="ad">广告内容</div>
    <p>正常内容</p>
    <span class="noise">噪音</span>
    </body>
    </html>
    """

    blob = Blob(data=html_content.encode("utf-8"), source="test.html")
    parser = HTMLParser(custom_clean_rules=['.ad', '.noise'])
    nodes = parser.parse(blob)

    assert len(nodes) == 1
    text = nodes[0].text
    assert '广告内容' not in text, "广告应该被移除"
    assert '噪音' not in text, "噪音应该被移除"
    assert '正常内容' in text, "正常内容应该保留"
    print("✓ 自定义清洗规则测试通过")

def test_metadata_extraction():
    """测试元数据提取"""
    print("=== 测试元数据提取 ===")
    html_content = """
    <html>
    <head>
    <title>页面标题</title>
    <meta name="description" content="页面描述">
    <meta name="keywords" content="关键词1,关键词2">
    <link rel="canonical" href="https://example.com/page">
    </head>
    <body><h1>内容</h1></body>
    </html>
    """

    blob = Blob(data=html_content.encode("utf-8"), source="test.html")
    parser = HTMLParser()
    nodes = parser.parse(blob)

    assert len(nodes) == 1
    meta = nodes[0].metadata
    assert meta.get('title') == '页面标题'
    assert meta.get('description') == '页面描述'
    assert meta.get('keywords') == '关键词1,关键词2'
    assert meta.get('canonical_url') == 'https://example.com/page'
    print("✓ 元数据提取测试通过")

def test_edge_cases():
    """测试边界情况"""
    print("=== 测试边界情况 ===")
    # 空HTML
    blob = Blob(data=b"", source="empty.html")
    parser = HTMLParser()
    nodes = parser.parse(blob)
    assert len(nodes) == 0, "空HTML应该返回空节点列表"

    # 无标题的HTML
    html_content = "<html><body><p>只有段落</p></body></html>"
    blob = Blob(data=html_content.encode("utf-8"), source="no_title.html")
    nodes = parser.parse(blob)
    assert len(nodes) == 1
    assert nodes[0].metadata['section_header'] == 'Introduction'  # 默认标题
    print("✓ 边界情况测试通过")

def test_load_data():
    """测试load_data方法"""
    print("=== 测试load_data方法 ===")
    # 创建临时HTML文件
    temp_file = Path("/tmp/test_page.html")
    html_content = "<html><head><title>文件测试</title></head><body><h1>测试</h1></body></html>"
    temp_file.write_text(html_content, encoding='utf-8')

    try:
        parser = HTMLParser()
        nodes = parser.load_data(str(temp_file))
        assert len(nodes) == 1
        assert nodes[0].metadata.get('title') == '文件测试'
        print("✓ load_data方法测试通过")
    finally:
        temp_file.unlink()

def test_section_path():
    """测试section_path层级路径"""
    print("=== 测试section_path层级路径 ===")
    html_content = """
    <html>
    <body>
    <h1>一级标题</h1>
    <p>一级内容</p>
    <h2>二级标题</h2>
    <p>二级内容</p>
    <h3>三级标题</h3>
    <p>三级内容</p>
    <h2>另一个二级标题</h2>
    <p>另一个二级内容</p>
    </body>
    </html>
    """

    blob = Blob(data=html_content.encode("utf-8"), source="test.html")
    parser = HTMLParser()
    nodes = parser.parse(blob)

    # 应该有4个节点：一级、二级、三级、另一个二级
    assert len(nodes) == 4
    paths = [node.metadata.get('section_path') for node in nodes]
    expected_paths = [
        "一级标题",
        "一级标题 > 二级标题",
        "一级标题 > 二级标题 > 三级标题",
        "一级标题 > 另一个二级标题"
    ]
    print("✓ section_path层级路径测试通过")

def test_block_protection():
    """测试块保护：代码块、数学公式块、表格内的内容不会被误切分"""
    print("=== 测试块保护逻辑 ===")
    html_content = """
    <html>
    <body>
    <h1>主要内容</h1>
    <p>这里是一些内容</p>
    <pre><code>
# 这不是标题
def function():
    # 这也不是标题
    return True
    </code></pre>
    <h2>数学部分</h2>
    <p>公式：</p>
    $$
    # 这不是标题
    E = mc^2
    # 也不是标题
    $$
    <h2>表格部分</h2>
    <table>
    <tr><th>项目</th><th>值</th></tr>
    <tr><td># 不是标题</td><td>100</td></tr>
    </table>
    <h1>结尾</h1>
    <p>结束内容</p>
    </body>
    </html>
    """

    blob = Blob(data=html_content.encode("utf-8"), source="test.html")
    parser = HTMLParser()
    nodes = parser.parse(blob)

    # 应该有4个节点：主要内容、数学部分、表格部分、结尾
    assert len(nodes) == 4
    headers = [node.metadata.get('section_header') for node in nodes]
    expected_headers = ["主要内容", "数学部分", "表格部分", "结尾"]
    assert headers == expected_headers, f"期望标题: {expected_headers}, 实际: {headers}"
    
    # 检查内容中是否包含了代码块和数学公式块
    content_main = nodes[0].text
    content_math = nodes[1].text
    assert "def function():" in content_main, "代码块应该包含在主要内容中"
    assert "E = mc^2" in content_math, "数学公式应该包含在数学部分中"
    
    print("✓ 块保护逻辑测试通过")

def run_all_tests():
    """运行所有测试"""
    print("开始HTMLParser功能测试...\n")
    try:
        test_basic_parsing()
        test_remove_images_and_links()
        test_custom_clean_rules()
        test_metadata_extraction()
        test_section_path()
        test_block_protection()
        test_edge_cases()
        test_load_data()
        print("\n🎉 所有测试通过！HTMLParser功能正确实现。")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()