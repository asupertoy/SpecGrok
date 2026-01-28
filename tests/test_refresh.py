#!/usr/bin/env python3
"""Tests for IndexManager refresh_index automatic reingest behavior."""

import os
import sys
from pathlib import Path
import hashlib

# Set testing mode
os.environ['IS_TESTING'] = '1'

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from llama_index.core import StorageContext, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from src.database.qdrant_manager import qdrant_manager
from src.ingestion.indexmannager import IndexManager
from src.ingestion.pipeline import IngestionPipelineWrapper
from src.config import settings


def test_refresh_auto_reingest(tmp_path):
    # prepare test file
    test_file = tmp_path / "refresh_sample.txt"
    test_file.write_text("first version\n")
    file_path = str(test_file)

    # setup storage and manager
    client = qdrant_manager.get_client()
    vector_store = QdrantVectorStore(client=client, collection_name=settings.QDRANT_COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # simple handler recording calls and invoking pipeline.run to actually reingest
    calls = []

    index_manager = None
    pipeline = None

    def handler(doc_ids):
        # 记录调用；在单元测试中不要实际触发 reingest 的 pipeline.run（避免依赖外部 embedding）
        calls.extend(doc_ids)

    # init manager with auto_reingest and handler
    index_manager = IndexManager(storage_context=storage_context, auto_reingest=True, reingest_handler=handler)

    pipeline = IngestionPipelineWrapper(index_manager=index_manager)

    # initial ingest
    pipeline.run(file_path)

    # modify file to trigger change
    test_file.write_text("second version\nmore content\n")

    # create a Document with updated hash
    with open(file_path, 'rb') as f:
        data = f.read()
    new_hash = hashlib.md5(data).hexdigest()

    doc = Document(doc_id=file_path, extra_info={'doc_id': new_hash, 'source': file_path})

    # call refresh_index, expect it to schedule reingest and automatically call handler
    reingest = index_manager.refresh_index([doc])

    assert file_path in reingest
    assert file_path in calls

    # cleanup - remove all points with this collection is left to test teardown
    # (Qdrant test collection may persist across tests; production tests should use isolated collection)
