import os
import tempfile
from src.rag.retriever_setup import retriever_chain
import pytesseract
from PIL import Image

from fastapi import UploadFile, File, HTTPException
from langchain_core.documents import Document
from langchain_text_splitters  import  RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader


SUPPORTED_TYPES = {
    ".pdf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg"
}


def document_loader(description: str, file: UploadFile = File(...)):

    filename = file.filename
    extension = os.path.splitext(filename)[1].lower()

    if extension not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Only {SUPPORTED_TYPES} are allowed.")

    file_bytes = file.file.read()  

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as tmp_file:
        
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name
    

    try:
        """Load document based on file type"""

        #PDF
        if extension == ".pdf":

            pymupdf_loader = PyMuPDFLoader(tmp_file_path)
            docs = pymupdf_loader.load()

        #TEXT
        elif extension == ".txt":

            text_loader = TextLoader(tmp_file_path, encoding="utf-8")
            docs = text_loader.load()
        
        #IMAGES 
        else:

            image = Image.open(tmp_file_path)
            text = pytesseract.image_to_string(image)

            docs = [Document(page_content=text, metadata={})]

    except Exception as e:

        raise HTTPException(
            status_code=500, 
            detail=f"Error processing file: {str(e)}")
    
    finally:
        os.unlink(tmp_file_path)
    

    # Add metadata 
    for doc in docs:

        doc.metadata.update(
            {
                "source": filename,
                "description": description,
                "file_type": extension
            }
        )

   
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    
    chunks = splitter.split_documents(docs)

    return retriever_chain(chunks)

