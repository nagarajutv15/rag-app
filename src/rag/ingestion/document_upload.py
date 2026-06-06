import pathlib
import os
import shutil
from fastapi import UploadFile

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

def save_document(
    file: UploadFile,
    department_id: str
):
    
    department_folder = os.path.join(
        UPLOAD_FOLDER,
        department_id
    )

    os.makedirs(
        department_folder,
        exist_ok=True
    )

    file_path = os.path.join(
        department_folder, 
        file.filename
    )

    with open(file_path, "wb") as buffer:
        
        shutil.copyfileobj(
            file.file,
            buffer
        )

    return file_path
