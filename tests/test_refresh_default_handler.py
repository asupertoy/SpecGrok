#!/usr/bin/env python3
"""Test default reingest handler uses IngestionPipelineWrapper.run for each doc."""

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
from src.config import settings


def test_default_reingest_handler_monkeypatch(monkeypatch, tmp_path):
    # prepare test file
    test_file = tmp_path / "refresh_default.txt"
    test_file.write_text("initial\n")
    file_path = str(test_file)

    client = qdrant_manager.get_client()
    vector_store = QdrantVectorStore(client=client, collection_name=settings.QDRANT_COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # fake pipeline to capture runs
    class FakePipeline:
        created = []
        def __init__(self, index_manager, config=None):
            FakePipeline.created.append(self)
            self.calls = []
        def run(self, doc_id):
            self.calls.append(doc_id)
            return []

    # monkeypatch the real IngestionPipelineWrapper to our fake
    import src.ingestion.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, 'IngestionPipelineWrapper', FakePipeline)

    # init manager without handler, auto_reingest True
    index_manager = IndexManager(storage_context=storage_context, auto_reingest=True)

    # computation: new hash
    with open(file_path, 'rb') as f:
        new_hash = hashlib.md5(f.read()).hexdigest()

    doc = Document(doc_id=file_path, extra_info={'doc_id': new_hash, 'source': file_path})

    reingest = index_manager.refresh_index([doc])

    # ensure reingest requested
    assert file_path in reingest

    # ensure FakePipeline was created and run called
    assert FakePipeline.created, "FakePipeline not created"
    created = FakePipeline.created[0]
    assert file_path in created.calls
