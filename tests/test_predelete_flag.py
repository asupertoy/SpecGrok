import sys, os
# 确保在独立运行单测时能找到项目的 `src` 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.ingestion.indexmannager import IndexManager


class DummyIndex:
    def __init__(self):
        self.deleted = None
        self.inserted = None

    def delete_nodes(self, node_ids):
        self.deleted = list(node_ids)

    def insert_nodes(self, nodes):
        self.inserted = nodes


class DummyNode:
    def __init__(self, text, node_id=None, source=None):
        self.text = text
        self.metadata = {}
        if source:
            self.metadata['source'] = source
        if node_id:
            self.node_id = node_id


def test_predelete_flag_default_false(monkeypatch):
    idx_mgr = IndexManager(storage_context=None)  # storage_context not used in this test
    # 避免实际调用嵌入逻辑
    monkeypatch.setattr(idx_mgr, "_ensure_embeddings", lambda nodes: None)
    dummy_index = DummyIndex()
    monkeypatch.setattr(idx_mgr, "_get_or_load_index", lambda: dummy_index)

    nodes = [DummyNode("a", node_id="id1"), DummyNode("b", node_id="id2")]
    idx_mgr.upsert_nodes(nodes)

    assert dummy_index.inserted is not None
    assert dummy_index.deleted is None


def test_predelete_flag_true_triggers_delete(monkeypatch):
    idx_mgr = IndexManager(storage_context=None, predelete_before_upsert=True)
    monkeypatch.setattr(idx_mgr, "_ensure_embeddings", lambda nodes: None)
    dummy_index = DummyIndex()
    monkeypatch.setattr(idx_mgr, "_get_or_load_index", lambda: dummy_index)

    nodes = [DummyNode("a", node_id="id1"), DummyNode("b", node_id="id2")]
    idx_mgr.upsert_nodes(nodes)

    assert dummy_index.inserted is not None
    # IndexManager 会覆盖 node_id 为确定性 md5 值，基于 source + 文本内容计算
    import hashlib
    expected_deleted = set()
    for n in nodes:
        source = n.metadata.get('source', '') or n.metadata.get('file_name', '')
        content_hash = hashlib.md5((n.text or '').encode('utf-8')).hexdigest()
        node_key = f"{source}|{content_hash}"
        expected_deleted.add(hashlib.md5(node_key.encode('utf-8')).hexdigest())

    assert set(dummy_index.deleted) == expected_deleted