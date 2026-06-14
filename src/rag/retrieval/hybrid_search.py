from src.rag.retrieval.retriever import (
    retrieve_documents
)

from src.rag.vectorstore.bm25_store import (
    bm25_search
)

from src.rag.retrieval.reranker import (
    rerank
)


# Hybrid Search:
# BM25 + Vector Search + CrossEncoder Reranking

def hybrid_search(
    query: str,
    department_id: str | None = None,
    top_k: int = 10,
    min_score: float = 0.0
):

    bm25_results = bm25_search(
        query=query,
        department_id=department_id,
        top_k=top_k
    )

    vector_results = retrieve_documents(
        query=query,
        department_id=department_id,
        top_k=top_k,
        min_score=min_score
    )

    merged_results = {}

    # VECTOR RESULTS

    for doc in vector_results:

        chunk_id = doc.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        merged_results[chunk_id] = {

            **doc,

            "hybrid_score":

                doc.get(
                    "score",
                    0.0
                ) * 0.7
        }

    # BM25 RESULTS

    for doc in bm25_results:

        chunk_id = doc.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        if chunk_id in merged_results:

            merged_results[chunk_id][
                "hybrid_score"
            ] += (

                doc.get(
                    "bm25_score",
                    0.0
                ) * 0.3
            )

        else:

            merged_results[chunk_id] = {

                **doc,

                "hybrid_score":

                    doc.get(
                        "bm25_score",
                        0.0
                    ) * 0.3
            }

    final_results = list(
        merged_results.values()
    )

    final_results.sort(

        key=lambda x: x.get(
            "hybrid_score",
            0.0
        ),

        reverse=True
    )

    reranked_results = rerank(

        query=query,

        documents=final_results,

        top_k=top_k
    )

    return reranked_results