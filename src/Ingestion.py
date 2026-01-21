import os
# from langchain_community_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from typing import List, Callable, Iterable, Optional

SUPPORTED_EXTENSIONS = {'.pdf', '.xls', '.xlsx', '.docx', '.ppt', '.pptx', '.csv'}

class data_ingestor:
    """
    Clean ingestor:
    - validate files using file_validator
    - dispatch to the appropriate loader lazily
    - add consistent metadata to returned Documents
    """
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path is not None else None

    def file_validator(self, file_path: Optional[Path] = None) -> str:
        """Return the lower-case suffix if supported, else raise ValueError."""
        p = file_path if file_path is not None else self.path
        suffix = p.suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            return suffix
        raise ValueError(f"Unsupported file type: {suffix}")

    def _loader_for(self, suffix: str) -> Callable[[Path], Iterable]:
        """Return a callable that accepts a Path and returns an iterable of Documents."""
        suffix = suffix.lower()
        if suffix == '.pdf':
            from langchain_community.document_loaders import PyMuPDFLoader
            return lambda p: PyMuPDFLoader(str(p)).load()

        if suffix == '.csv':
            from langchain_community.document_loaders import CSVLoader
            return lambda p: CSVLoader(str(p)).load()

        if suffix == '.docx':
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                return lambda p: UnstructuredWordDocumentLoader(str(p)).load()
            except Exception:
                from langchain_community.document_loaders import Docx2txtLoader
                return lambda p: Docx2txtLoader(str(p)).load()

        if suffix in ('.ppt', '.pptx'):
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            return lambda p: UnstructuredPowerPointLoader(str(p)).load()

        if suffix in ('.xls', '.xlsx'):
            try:
                from langchain_community.document_loaders import UnstructuredExcelLoader
                return lambda p: UnstructuredExcelLoader(str(p)).load()
            except Exception as e:
                print(f"Error importing UnstructuredExcelLoader: {e}")
                

        raise ValueError(f"No loader configured for {suffix}")

    def ingest(self) -> List:
        """Ingest a file or directory and return list of Documents."""

        all_documents = []
        files = []

        if self.path.is_file():
            try:
                self.file_validator()
                files = [self.path]
            except ValueError:
                print(f"Skipping unsupported file: {self.path.name}")
                return []
        elif self.path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(list(self.path.glob(f"**/*{ext}")))
        else:
            print(f"Path does not exist: {self.path}")
            return []

        print(f"Found {len(files)} file(s) to process")

        for file_path in files:
            print(f"\nProcessing file {file_path.name}")
            try:
                suffix = self.file_validator(file_path)
                loader = self._loader_for(suffix)
                documents = list(loader(file_path))

                for doc in documents:
                    # ensure metadata exists and update
                    meta = getattr(doc, "metadata", {}) or {}
                    meta['source_file'] = file_path.name
                    meta['file_type'] = suffix.lstrip('.')
                    doc.metadata = meta

                all_documents.extend(documents)
                print(f"Loaded {len(documents)} pages/documents")
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents