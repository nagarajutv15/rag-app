from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from fastapi import HTTPException


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

        else:

            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported"
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