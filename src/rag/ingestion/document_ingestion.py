import uuid
import os
import shutil
from fastapi import UploadFile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from fastapi import HTTPException
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings



load_dotenv()


UPLOAD_FOLDER = "uploads"

# Create the uploads directory if it doesn't exist
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# chunk size
CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000"
    )
)

# chunk overlap
CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200"
    )
)

# embedding model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

# embedding batch size
EMBEDDING_BATCH_SIZE = int(os.getenv(
    "EMBEDDING_BATCH_SIZE",
    "100",
))


#----------------------------------------------------------------------------------------------------------#
# This function loads a document based on its file type (PDF, TXT, DOCX) and returns the content as a list of documents.

def load_document(file_path: str):

    try:
        if file_path.endswith(".pdf"):

            loader = PyPDFLoader(file_path)

        elif file_path.endswith(".txt"):

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

        elif file_path.endswith(".docx"):

            loader = Docx2txtLoader(file_path)

        else:

            raise HTTPException(
                status_code=400,
                detail="Only PDF, TXT, and DOCX files are supported"
            )

        documents = loader.load()

        if not documents:

            raise HTTPException(
                status_code=400,
                detail="No content found in document"
            )

        return documents

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document loading failed: {str(e)}"
        )
    


#----------------------------------------------------------------------------------------------------------#
# This function saves an uploaded document to a specific department folder and returns the file path.   

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


#----------------------------------------------------------------------------------------------------------#
# This function chunks a list of documents into smaller pieces based on the specified chunk size and overlap.


def chunk_documents(
    documents: List[Document]
) -> List[Document]:

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:

        chunk.metadata["chunk_id"] = str(
            uuid.uuid4()
        )

    return chunks



#----------------------------------------------------------------------------------------------------------#
# This Gives us the embedding model instance configured with the specified model name and batch size for generating embeddings.

def get_embedding_model():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        chunk_size=EMBEDDING_BATCH_SIZE,
    )



#----------------------------------------------------------------------------------------------------------#
# This function generates embeddings for a list of document chunks using the specified embedding model.

def generate_embeddings(chunks: List[Document]):

    if not chunks:
        return []
    
    embeddings_model = get_embedding_model()

    texts = [chunk.page_content for chunk in chunks]

    vectors = embeddings_model.embed_documents(texts=texts)

    return vectors

