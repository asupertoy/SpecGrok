import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.qdrant_manager import qdrant_manager

try:
    # 1. 初始化并获取客户端
    cli = qdrant_manager.get_client()
    
    # 2. 获取集合列表
    collections = cli.get_collections()
    
    print("✅ 成功连接到 Qdrant!")
    # print(f"📍 地址: {cli.rest_uri}")
    print(f"📚 当前集合列表: {collections}")

    # 3. 尝试创建我们在配置中定义的集合
    qdrant_manager.create_collection_if_not_exists()
    
    print("✅ 集合检查/创建完成")

except Exception as e:
    print(f"❌ 连接失败: {e}")
