from pathlib import Path
from fastapi import UploadFile
from typing import List
import shutil


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


async def save_document(
    files: List[UploadFile],
    description: str
):

    uploaded_files = []

    for upload in files:

        file_path = UPLOAD_DIR / upload.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)

        uploaded_files.append(
            {
                "filename": upload.filename,
                "description": description,
                "path": str(file_path)
            }
        )

    return uploaded_files