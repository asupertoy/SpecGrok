from typing import Optional, Dict, List, Union
from pathlib import Path
import mimetypes
import os
import hashlib

class Blob:
    """即插即用的简单数据容器"""
    def __init__(self, data: bytes, source: str, metadata: Optional[Dict] = None):
        self.data = data
        self.source = source
        self.metadata = metadata or {}

    def as_bytes(self) -> bytes:
        """Return raw bytes of the blob"""
        return self.data

    @classmethod
    def from_path(cls, file_path: str):
        """智能读取文件并提取元数据"""
        with open(file_path, 'rb') as f:
            data = f.read()
            
        # --- 自动生成 Metadata ---
        stats = os.stat(file_path)
        metadata = {
            "size": stats.st_size,
            "file_size": stats.st_size,
            "file_name": os.path.basename(file_path),
            "doc_id": hashlib.md5(data).hexdigest(),  # Early computation of hash
            "extension": os.path.splitext(file_path)[1].lower(),
            # 兼容 mimetypes
            "mime_type": mimetypes.guess_type(file_path)[0]
        }

        # --- 目录结构提取逻辑 ---
        domain_map = {
            'rfc': 'RFC',
            'k8s': 'Kubernetes',
            'docker': 'Docker',
            'dsp': 'DSP',
            'llamaindex': 'LlamaIndex',
            'langchain': 'LangChain',
            'langgraph': 'LangGraph'
        }
        parts = os.path.normpath(file_path).split(os.sep)
        for part in parts:
            part_lower = part.lower()
            if part_lower in domain_map:
                metadata['domain'] = domain_map[part_lower]
                break 

        return cls(data=data, source=file_path, metadata=metadata)


class Loader:
    """通用 Loader，用于批量加载 Blob 对象"""

    def __init__(self, extensions: Optional[List[str]] = None, docs_enabled: Optional[List[str]] = None, recursive: bool = False, follow_symlinks: bool = False):
        # 统一转小写
        self.extensions = [ext.lower() for ext in extensions] if extensions else None
        # 可选择加载的文档类型（仅在没有显式传入 extensions 时生效）
        default_docs = ['.md', '.html', '.pdf', '.txt']
        self.docs_enabled = [ext.lower() for ext in docs_enabled] if docs_enabled else default_docs
        self.recursive = recursive
        self.follow_symlinks = follow_symlinks
        # 默认忽略的目录和文件
        self.exclude_dirs = {'_images', '.git', '__pycache__'}
        self.exclude_files = {'.DS_Store'}

    @staticmethod
    def from_bytes(data: bytes, source: str, metadata: Optional[Dict] = None) -> Blob:
        """Construct a Blob directly from bytes"""
        return Blob(data=data, source=source, metadata=metadata or {})

    def _match_extension(self, path: Path) -> bool:
        ext = path.suffix.lower()
        if self.extensions is not None:
            return ext in self.extensions
        # 使用 docs_enabled 作为默认的过滤规则
        return ext in self.docs_enabled

    def load(self, path: Union[str, Path]) -> List[Blob]:
        p = Path(path)
        if p.is_file():
            if not self._match_extension(p):
                raise ValueError(f"File extension not allowed: {p}")
            return [self._load_single_file(p)]
        elif p.is_dir():
            return self._load_dir(p)
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    def _load_single_file(self, file_path: Path) -> Blob:
        # !!! 关键修改：直接调用 Blob.from_path 复用逻辑 !!!
        return Blob.from_path(str(file_path))

    def _load_dir(self, dir_path: Path) -> List[Blob]:
        blobs: List[Blob] = []
        # 使用 os.walk 来获得更好的控制力（过滤目录）
        if self.recursive:
            for root, dirs, files in os.walk(dir_path, followlinks=self.follow_symlinks):
                # 过滤掉不需要的目录
                dirs[:] = [d for d in dirs if d not in self.exclude_dirs and not d.startswith('.')]
                
                for file in files:
                    if file in self.exclude_files or file.startswith('.'):
                        continue
                        
                    file_p = Path(root) / file
                    # respect follow_symlinks: skip symlink files if follow_symlinks is False
                    if file_p.is_symlink() and not self.follow_symlinks:
                        continue
                    if self._match_extension(file_p):
                        blobs.append(self._load_single_file(file_p))
        else:
            # 非递归只看当前层
            for p in dir_path.iterdir():
                if p.is_file():
                     # 简单过滤
                    if p.name in self.exclude_files or p.name.startswith('.'):
                        continue
                    # respect follow_symlinks for files
                    if p.is_symlink() and not self.follow_symlinks:
                        continue
                    if self._match_extension(p):
                         blobs.append(self._load_single_file(p))
                         
        return blobs

