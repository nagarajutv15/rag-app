from src.rag.retrieval.retriever import retrieve_documents
from src.rag.vectorstore.bm25_store import bm25_search
from src.rag.retrieval.reranker import rerank


def hybrid_search(
    query: str,
    top_k: int = 10,
    min_score: float = 0.0
):

    bm25_results = bm25_search(
        query=query,
        top_k=top_k
    )

    vectore_results = retrieve_documents(
        query=query,
        top_k=top_k,
        min_score=min_score
    )

    merged_results = {}

    # ADD VECTOR RESULTS
    for doc in vectore_results:

        chunk_id = doc.get("chunk_id")

        merged_results[chunk_id] = {

            **doc,
            "hubrid_score": doc.get("score", 0.0)*0.7
        }
    

    # ADD BM25 RESULTS
    for doc in bm25_results:

        chunk_id = doc["chunk_id"]

        if chunk_id in merged_results:

            merged_results[chunk_id][
                "hybrid_score"
            ] += (
                doc.get("bm25_score", 0) * 0.3
            )

        else:

            merged_results[chunk_id] = {

                **doc,

                "hybrid_score": (
                    doc.get("bm25_score", 0) * 0.3
                )
            }
    
    # CONVERT TO LIST
    final_results = list(
        merged_results.values()
    )

    # SORT
    final_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    # FINAL RERANK
    reranked_results = rerank(
        query=query,
        documents=final_results,
        top_k=top_k
    )

    return reranked_results

