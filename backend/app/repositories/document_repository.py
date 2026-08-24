from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.document import Document, DocumentChunk


class DocumentRepository:
    def add(self, db: Session, document: Document) -> None:
        db.add(document)

    def chunks_for_organization(self, db: Session, organization_id: UUID, company_id: UUID | None) -> list[DocumentChunk]:
        statement = select(DocumentChunk).join(Document, Document.id == DocumentChunk.document_id).where(DocumentChunk.organization_id == organization_id)
        if company_id is not None:
            statement = statement.where((Document.company_id == company_id) | (Document.company_id.is_(None)))
        return list(db.scalars(statement.order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)))

    def add_chunks(self, db: Session, chunks: list[DocumentChunk]) -> None:
        db.add_all(chunks)
