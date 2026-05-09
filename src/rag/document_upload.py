import os
import tempfile
from src.tools.common_tools import enhance_description_with_llm
from src.rag.retriever_setup import retriever_chain

from fastapi import UploadFile, File, HTTPException
from langchain_text_splitters  import  RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader


def document_loader(description: str, file: UploadFile = File(...)):

    filename = file.filename
    if not filename.endswith(".pdf") and not filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and TXT are allowed.")

    file_bytes = file.file.read()  

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(filename)[1]
    ) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name
    
    
    if filename.endswith(".pdf"):
        loader = PyMuPDFLoader(tmp_file_path)
    else:
        loader = TextLoader(tmp_file_path, encoding="utf-8")


    try:
        docs = loader.load()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading file: {e}"
        )
    finally:
        os.unlink(tmp_file_path)


    # Add metadata 
    for doc in docs:
        doc.metadata["description"] = description
        doc.metadata["source"] = filename

    # # Enhance description using LLM
    # description_llm = enhance_description_with_llm(description)

    # # Save enhanced description
    # with open("description.txt", "w", encoding="utf-8") as f:
    #     f.write(description_llm)

    # with open("description.txt", "r", encoding="utf-8") as f:
    #     print("Document description from storage:")
    #     print(f.read())

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    return retriever_chain(chunks)

