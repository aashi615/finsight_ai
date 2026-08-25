import io
import re
from uuid import UUID
from urllib.parse import urlsplit
import faiss
import numpy as np
from pypdf import PdfReader
from sqlalchemy.orm import Session
from app.core.exceptions import api_error
from app.llm.base import LLMProvider, LLMProviderError
from app.models.document import Document, DocumentChunk
from app.repositories.document_repository import DocumentRepository


class RagService:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.documents = DocumentRepository()

    def ingest_text(self, db: Session, organization_id: UUID, title: str, content: str, company_id: UUID | None = None, source_url: str | None = None, mime_type: str = "text/plain") -> Document:
        if not content.strip():
            raise api_error(422, "INVALID_DOCUMENT", "Document content cannot be empty.")
        self._validate_source_url(source_url)
        document = Document(organization_id=organization_id, company_id=company_id, title=title.strip(), content=content, source_url=source_url, mime_type=mime_type)
        return self._persist_document(db, document, [(part, None, None) for part in self._chunk(content)])

    def _persist_document(self, db: Session, document: Document, parts: list[tuple[str, int | None, str | None]]) -> Document:
        self.documents.add(db, document)
        db.flush()
        try:
            embeddings = self.llm.embed([part[0] for part in parts])
        except LLMProviderError:
            db.rollback()
            raise api_error(503, "LLM_UNAVAILABLE", "Embedding provider is unavailable.")
        if len(embeddings) != len(parts):
            db.rollback()
            raise api_error(502, "MALFORMED_LLM_OUTPUT", "Embedding provider returned invalid data.")
        self.documents.add_chunks(db, [DocumentChunk(organization_id=document.organization_id, document_id=document.id, chunk_index=index, page_number=page_number, section=section, source_url=document.source_url, content=content, embedding=embedding) for index, ((content, page_number, section), embedding) in enumerate(zip(parts, embeddings, strict=True))])
        db.commit()
        db.refresh(document)
        return document

    def ingest_file(self, db: Session, organization_id: UUID, title: str, raw: bytes, mime_type: str, company_id: UUID | None = None, source_url: str | None = None) -> Document:
        self._validate_source_url(source_url)
        if mime_type == "application/pdf":
            try:
                page_text = [(index + 1, page.extract_text() or "") for index, page in enumerate(PdfReader(io.BytesIO(raw)).pages)]
            except Exception as exc:
                raise api_error(422, "INVALID_DOCUMENT", "PDF could not be read.") from exc
            parts = [(chunk, page_number, f"page {page_number}") for page_number, text in page_text for chunk in self._chunk(text)]
            if not parts:
                raise api_error(422, "INVALID_DOCUMENT", "Document content cannot be empty.")
            document = Document(organization_id=organization_id, company_id=company_id, title=title.strip(), content="\n".join(text for _, text in page_text), source_url=source_url, mime_type=mime_type)
            return self._persist_document(db, document, parts)
        elif mime_type.startswith("text/"):
            content = raw.decode("utf-8")
        else:
            raise api_error(422, "UNSUPPORTED_DOCUMENT", "Only text and PDF documents are supported.")
        return self.ingest_text(db, organization_id, title, content, company_id, source_url, mime_type)

    def retrieve(self, db: Session, organization_id: UUID, query: str, company_id: UUID | None, limit: int = 5) -> list[DocumentChunk]:
        chunks = self.documents.chunks_for_organization(db, organization_id, company_id)
        if not chunks:
            return []
        try:
            vector = self.llm.embed([query])[0]
        except (LLMProviderError, IndexError):
            raise api_error(503, "LLM_UNAVAILABLE", "Embedding provider is unavailable.")
        matrix = np.asarray([chunk.embedding for chunk in chunks], dtype="float32")
        query_matrix = np.asarray([vector], dtype="float32")
        if matrix.ndim != 2 or query_matrix.shape[1] != matrix.shape[1]:
            # Existing documents may have been embedded by a retired external
            # provider. Keep those documents usable after provider migration.
            terms = set(re.findall(r"[a-z0-9]{2,}", query.lower()))
            return sorted(chunks, key=lambda chunk: len(terms & set(re.findall(r"[a-z0-9]{2,}", chunk.content.lower()))), reverse=True)[:limit]
        faiss.normalize_L2(matrix)
        faiss.normalize_L2(query_matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        _, positions = index.search(query_matrix, min(limit, len(chunks)))
        return [chunks[position] for position in positions[0] if position >= 0]

    @staticmethod
    def _chunk(content: str, size: int = 1000, overlap: int = 150) -> list[str]:
        return [content[start:start + size] for start in range(0, len(content), size - overlap) if content[start:start + size].strip()]

    @staticmethod
    def _validate_source_url(source_url: str | None) -> None:
        if source_url:
            parts = urlsplit(source_url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                raise api_error(422, "INVALID_SOURCE_URL", "Document source URL must be an absolute HTTP(S) URL.")
