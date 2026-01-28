import sys
from pathlib import Path
import pytest

# ensure src is on sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from src.models.embedding import sparse_doc_fn, sparse_query_fn


def test_sparse_adapter_simple():
    texts = ["hello world", "the quick brown fox"]
    indices_list, values_list = sparse_doc_fn(texts)

    assert isinstance(indices_list, list)
    assert isinstance(values_list, list)
    assert len(indices_list) == len(texts)
    assert len(values_list) == len(texts)

    # elements should be lists (can be empty)
    for idxs, vals in zip(indices_list, values_list):
        assert isinstance(idxs, list)
        assert isinstance(vals, list)
        assert len(idxs) == len(vals)


def test_sparse_query_fn_simple():
    idxs, vals = sparse_query_fn("some query text")
    assert isinstance(idxs, list)
    assert isinstance(vals, list)
    assert len(idxs) == len(vals)


def test_sparse_doc_fn_with_node_like():
    class DummyNode:
        def __init__(self, text):
            self.text = text

    idxs_list, vals_list = sparse_doc_fn([DummyNode("node text")])
    assert isinstance(idxs_list, list)
    assert isinstance(vals_list, list)
    assert len(idxs_list) == 1
    assert len(vals_list) == 1
    assert isinstance(idxs_list[0], list)
    assert isinstance(vals_list[0], list)


def test_service_exposes_adapter_methods():
    from src.models.embedding import BgeM3Service

    # call class staticmethods
    ids_list, vals_list = BgeM3Service.sparse_doc_fn(["quick brown fox"])
    assert isinstance(ids_list, list) and isinstance(vals_list, list)

    ids_q, vals_q = BgeM3Service.sparse_query_fn("hello")
    assert isinstance(ids_q, list) and isinstance(vals_q, list)
