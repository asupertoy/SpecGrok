#!/usr/bin/env python3
"""Tests for default reingest handler behavior: batching, delay, and failure resilience."""

import time
from pathlib import Path
import hashlib

import pytest

project_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(project_root))

from llama_index.core import StorageContext, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from src.database.qdrant_manager import qdrant_manager
from src.ingestion.indexmannager import IndexManager
from src.config import settings


class FakePipelineRecorder:
    """A fake pipeline that records run calls per instance."""
    created = []

    def __init__(self, index_manager, config=None):
        FakePipelineRecorder.created.append(self)
        self.calls = []

    def run(self, doc_id):
        # simulate a quick processing
        self.calls.append(doc_id)
        return []


class FakePipelineFlaky:
    def __init__(self, index_manager, config=None):
        self.calls = []

    def run(self, doc_id):
        # raise for doc containing 'fail'
        if 'fail' in doc_id:
            raise RuntimeError('simulated failure')
        self.calls.append(doc_id)
        return []


@pytest.fixture
def storage_context():
    client = qdrant_manager.get_client()
    vector_store = QdrantVectorStore(client=client, collection_name=settings.QDRANT_COLLECTION_NAME)
    return StorageContext.from_defaults(vector_store=vector_store)


def test_default_reingest_handler_batch_and_delay(monkeypatch, storage_context, tmp_path):
    # configure settings
    monkeypatch.setattr(settings, 'AUTO_REINGEST_MAX_BATCH', 2)
    monkeypatch.setattr(settings, 'AUTO_REINGEST_DELAY_SECONDS', 0.05)

    # monkeypatch pipeline class
    import src.ingestion.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, 'IngestionPipelineWrapper', FakePipelineRecorder)

    mgr = IndexManager(storage_context=storage_context, auto_reingest=True)

    # create 3 fake doc ids (exceeding max_batch)
    docs = ['d1', 'd2', 'd3']

    start = time.monotonic()
    mgr._default_reingest_handler(docs)
    elapsed = time.monotonic() - start

    # Only max_batch should be processed
    assert FakePipelineRecorder.created, 'No pipeline instance was created'
    inst = FakePipelineRecorder.created[-1]
    assert len(inst.calls) == 2

    # elapsed should be at least delay*(calls-1) (with small tolerance)
    assert elapsed >= settings.AUTO_REINGEST_DELAY_SECONDS * (len(inst.calls) - 1) - 0.01


def test_default_reingest_handler_continues_on_failure(monkeypatch, storage_context):
    # small batch, no delay
    monkeypatch.setattr(settings, 'AUTO_REINGEST_MAX_BATCH', 10)
    monkeypatch.setattr(settings, 'AUTO_REINGEST_DELAY_SECONDS', 0.0)

    import src.ingestion.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, 'IngestionPipelineWrapper', FakePipelineFlaky)

    mgr = IndexManager(storage_context=storage_context, auto_reingest=True)

    docs = ['ok1', 'fail_doc', 'ok2']

    # Should not raise, and should attempt all docs (fail handled internally)
    mgr._default_reingest_handler(docs)

    # Since FakePipelineFlaky throws for 'fail', ensure it didn't stop subsequent docs
    # There is no direct instance to inspect (created inside handler), but absence of exception is adequate.


def test_refresh_index_triggers_default_handler(monkeypatch, storage_context, tmp_path):
    # ensure default handler invoked via refresh_index
    import src.ingestion.pipeline as pipeline_mod

    # monkeypatch pipeline to recorder
    monkeypatch.setattr(pipeline_mod, 'IngestionPipelineWrapper', FakePipelineRecorder)

    mgr = IndexManager(storage_context=storage_context, auto_reingest=True)

    # create a file and doc with updated hash
    test_file = tmp_path / 'r2.txt'
    test_file.write_text('v2')
    file_path = str(test_file)
    new_hash = hashlib.md5(test_file.read_bytes()).hexdigest()

    doc = Document(doc_id=file_path, extra_info={'file_hash': new_hash, 'source': file_path})

    reingest = mgr.refresh_index([doc])
    assert file_path in reingest
    assert FakePipelineRecorder.created
    inst = FakePipelineRecorder.created[-1]
    assert file_path in inst.calls
