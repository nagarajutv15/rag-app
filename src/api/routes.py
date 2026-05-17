from fastapi import APIRouter, UploadFile, File, Header
from src.rag.document_upload import document_loader

from src.llms.llm import llm
from src.models.query_request import QueryRequest
from src.rag.retriever_setup import get_combined_retriever

router = APIRouter()


@router.post("/rag/query")
async def rag_query(req: QueryRequest):

    retriever = get_combined_retriever()

    if retriever is None:
        return {"answer": "No documents uploaded yet"}

    docs = retriever.get_documents(req.query)

    if not docs:
        return {"answer": "No relevant information found"}

    context = "\n\n".join([doc.page_content[:1000] for doc in docs])

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

