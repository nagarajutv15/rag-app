from src.rag.retrieval.hybrid_search import hybrid_search
from src.rag.prompts.templates import build_prompt
from src.llm.llm_service import generate_response

# This service handles the core logic for processing user queries, retrieving relevant information from the documents, 
# and generating responses using the LLM. It acts as the main orchestrator for the RAG workflow when a user asks a question.

def ask_rag_question(
    query: str
):

    retrieved_docs = hybrid_search(
        query=query,
        top_k=3
    )

    if not retrieved_docs:

        return {

            "question": query,

            "answer":
                "I couldn't find information in the documents."
        }

    context_chunks = [

        doc.get(
            "text",
            ""
        )

        for doc in retrieved_docs
    ]

    prompt = build_prompt(

        question=query,

        context_chunks=context_chunks
    )

    answer = generate_response(
        prompt
    )

    return {

        "question":
            query,

        "answer":
            answer,

        "sources": [

            doc.get(
                "chunk_id"
            )

            for doc in retrieved_docs
        ]
    }