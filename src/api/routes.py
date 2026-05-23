from typing import List
from src.rag.ingestion.document_upload import save_document
from fastapi import APIRouter, UploadFile, File, Header
from src.rag.ingestion.document_upload import save_document 

router = APIRouter()

@router.post("/rag/documents/upload")
async def upload_file(
    files: List[UploadFile] = File(...),
    description: str = Header(..., alias="X-Description")
):

    status_upload = await save_document(
        files,
        description
    )

    return {
        "status": status_upload
    }


