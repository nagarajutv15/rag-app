from typing import List
from src.models.document_schema import DocumentMetadata
from src.rag.ingestion.document_upload import save_document
from fastapi import APIRouter, UploadFile, File, Header, Depends, Form
from sqlalchemy.orm import Session
from src.models.database import get_db


router = APIRouter()


@router.post("/documents/upload")
async def upload_document(
    source: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    file_path = save_document(file)

    existing_document = (

        db.query(DocumentMetadata)
        .filter(
            DocumentMetadata.file_name == file.filename
        )
        .order_by(
            DocumentMetadata.version.desc()
        )
        .first()
    )

    version = 1

    if existing_document:

        existing_document.is_active = False

        version = existing_document.version + 1


    document = DocumentMetadata(

        file_name=file.filename,
        source=source,
        version=version,
        file_path=file_path,
        is_active=True
    )

    db.add(document)

    db.commit()

    db.refresh(document)

    return {

        "message": "Document uploaded successfully",

        "document_id": document.document_id,

        "file_name": document.file_name,

        "version": document.version
    }