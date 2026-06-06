from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from fastapi import HTTPException

# This function loads a document based on its file type (PDF, TXT, DOCX) and returns the content as a list of documents.

def load_document(file_path: str):

    try:

        if file_path.endswith(".pdf"):

            loader = PyPDFLoader(
                file_path
            )

        elif file_path.endswith(".txt"):

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

        elif file_path.endswith(".docx"):

            loader = Docx2txtLoader(
                file_path
            )

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
    