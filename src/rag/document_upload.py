from pathlib import Path
from fastapi import UploadFile
import shutil 

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

async def save_document(file: UploadFile):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return {
        "filename": file.filename,
        "path": str(file_path)
    }