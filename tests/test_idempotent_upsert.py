#!/usr/bin/env python3
"""Test idempotent upsert: running pipeline twice shouldn't increase Qdrant points."""

import sys
from pathlib import Path
import hashlib

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from src.database.qdrant_manager import qdrant_manager
from src.ingestion.indexmannager import IndexManager
from src.ingestion.pipeline import IngestionPipelineWrapper
from src.config import settings


def test_idempotent_upsert(tmp_path):
    # create test file
    test_file = tmp_path / 'idempotent.txt'
    test_file.write_text('some stable content')
    file_path = str(test_file)

    client = qdrant_manager.get_client()
    vector_store = QdrantVectorStore(client=client, collection_name=settings.QDRANT_COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index_manager = IndexManager(storage_context=storage_context)
    pipeline = IngestionPipelineWrapper(index_manager=index_manager)

    # Run pipeline twice
    pipeline.run(file_path)
    before_info = client.get_collection(settings.QDRANT_COLLECTION_NAME) if client.collection_exists(settings.QDRANT_COLLECTION_NAME) else None
    before = before_info.points_count if before_info else 0

    pipeline.run(file_path)
    after_info = client.get_collection(settings.QDRANT_COLLECTION_NAME) if client.collection_exists(settings.QDRANT_COLLECTION_NAME) else None
    after = after_info.points_count if after_info else 0

    assert after == before, f"Points increased after idempotent upsert: before={before} after={after}"
