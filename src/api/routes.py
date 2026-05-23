from src.rag.document_upload import save_document
from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    await save_document(file)

    return {"message": f"File '{file.filename}' uploaded successfully"}
