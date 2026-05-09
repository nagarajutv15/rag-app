"""
API routes for RAG operations.
"""


from fastapi import APIRouter, UploadFile, File, Header
from src.rag.document_upload import document_loader

from src.rag.retriever_setup import get_retriever
from src.llms.llm import llm
from src.models.query_request import QueryRequest

from src.models.query_request import QueryRequest


router = APIRouter()


@router.post("/rag/query")
async def rag_query(req: QueryRequest):

    retriever = get_retriever()

    if retriever is None:
        return {"answer": "No documents uploaded yet"}

    docs = retriever.invoke(req.query)

    if not docs:
        return {"answer": "No relevant information found in documents"}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful assistant.

    Answer ONLY using the context below.
    If the answer is not in the context, say:
    "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {req.query}
    """

    response = llm.invoke(prompt)

    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "chunks_used": len(docs)
    }


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """
    Upload a document for RAG processing.

    Args:
        file: The file to upload (PDF or TXT).
        description: Document description provided via header.

    Returns:
        Upload status.
    """
    status_upload = document_loader(description, file)
    return {"status": status_upload}
