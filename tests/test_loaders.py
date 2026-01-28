import sys
from pathlib import Path
import mimetypes
import pytest

# Ensure we can import the `src` package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.ingestion.loaders import Loader, Blob


def test_load_file(tmp_path):
    p = tmp_path / "file.txt"
    data = b"hello"
    p.write_bytes(data)

    loader = Loader()
    blobs = loader.load(p)

    assert len(blobs) == 1
    b = blobs[0]
    assert isinstance(b, Blob)
    assert b.as_bytes() == data
    assert b.source == str(p)
    assert b.metadata["size"] == len(data)
    # [Added] Verify file_hash computation
    import hashlib
    expected_hash = hashlib.md5(data).hexdigest()
    assert b.metadata.get("file_hash") == expected_hash, f"Hash mismatch: {b.metadata.get('file_hash')} != {expected_hash}"

    assert b.metadata.get("mime_type") == mimetypes.guess_type(str(p))[0]


def test_extension_filter(tmp_path):
    p = tmp_path / "note.md"
    p.write_bytes(b"md")

    loader = Loader(extensions=[".md"])
    blobs = loader.load(p)
    assert len(blobs) == 1

    loader2 = Loader(extensions=[".txt"])
    with pytest.raises(ValueError):
        loader2.load(p)


def test_docs_enabled_filter(tmp_path):
    d = tmp_path / "d2"
    d.mkdir()
    (d / "a.md").write_bytes(b"md")
    (d / "b.pdf").write_bytes(b"pdf")
    (d / "c.exe").write_bytes(b"exe")

    # 默认 Loader 应该使用 docs_enabled 默认集，包含 .md 和 .pdf
    loader_default = Loader()
    blobs = loader_default.load(d)
    sources = {b.source for b in blobs}
    assert str(d / "a.md") in sources
    assert str(d / "b.pdf") in sources
    assert not any(str(d / "c.exe") == s for s in sources)

    # 自定义 docs_enabled
    loader_custom = Loader(docs_enabled=[".md"])
    blobs2 = loader_custom.load(d)
    sources2 = {b.source for b in blobs2}
    assert str(d / "a.md") in sources2
    assert not any(str(d / "b.pdf") == s for s in sources2)

    # 显式传入 extensions 应优先于 docs_enabled
    loader_ext = Loader(extensions=[".md"])
    blobs3 = loader_ext.load(d)
    sources3 = {b.source for b in blobs3}
    assert str(d / "a.md") in sources3
    assert not any(str(d / "b.pdf") == s for s in sources3)


def test_load_dir_recursive_and_nonrecursive(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_bytes(b"a")
    sub = d / "sub"
    sub.mkdir()
    (sub / "b.txt").write_bytes(b"b")

    loader = Loader(recursive=False)
    blobs = loader.load(d)
    assert any(str((d / "a.txt")) == b.source for b in blobs)
    assert not any(str((sub / "b.txt")) == b.source for b in blobs)

    loader_rec = Loader(recursive=True)
    blobs_rec = loader_rec.load(d)
    assert any(str((sub / "b.txt")) == b.source for b in blobs_rec)


def test_from_bytes():
    blob = Loader.from_bytes(b"xyz", source="inmem", metadata={"k": "v"})
    assert isinstance(blob, Blob)
    assert blob.source == "inmem"
    assert blob.metadata["k"] == "v"


def test_symlink_follow(tmp_path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"t")

    linkdir = tmp_path / "linkdir"
    linkdir.mkdir()
    link = linkdir / "link.txt"
    link.symlink_to(target)

    loader_no_follow = Loader(follow_symlinks=False)
    blobs = loader_no_follow.load(linkdir)
    assert not any(str(link) == b.source for b in blobs)

    loader_follow = Loader(follow_symlinks=True)
    blobs2 = loader_follow.load(linkdir)
    assert any(str(link) == b.source for b in blobs2)
