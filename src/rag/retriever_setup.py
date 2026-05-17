import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from src.core.logger import logger as logging


embeddings = OpenAIEmbeddings()

_faiss_vectorstore = None
_bm25_retriever = None


"""
This module sets up a combined retriever that uses both FAISS (vector search) and BM25 (keyword search).
"""

class CombinedRetriever:

    def __init__(self, vectorstore, bm25):
        self.vectorstore = vectorstore
        self.bm25 = bm25

    def _normalize(self, scores):
        """
        Normalize scores between 0 and 1 (Min-Max normalization)
        """
        if not scores:
            return scores

        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return [1.0 for _ in scores]

        return [(s - min_s) / (max_s - min_s) for s in scores]

    def get_documents(
        self,
        query: str,
        k_vector=5,
        k_bm25=5,
        weight_vector=0.7,
        top_k=4
    ) -> List[Document]:

        vector_results = self.vectorstore.similarity_search_with_score(
            query, k=k_vector
        )

        vector_scores = [score for _, score in vector_results]
        vector_scores = [(1 / (1 + s)) for s in vector_scores]
        vector_scores = self._normalize(vector_scores)

        bm25_docs = self.bm25.get_relevant_documents(query)[:k_bm25]
        bm25_scores = list(range(len(bm25_docs), 0, -1))
        bm25_scores = self._normalize(bm25_scores)

        combined = {}

        for doc, score in vector_results:
            combined[doc.page_content] = {"doc": doc, "score": score * weight_vector}

        for i, doc in enumerate(bm25_docs):
            if doc.page_content in combined:
                combined[doc.page_content]["score"] += (
                    bm25_scores[i] * (1 - weight_vector)
                )
            else:
                combined[doc.page_content] = {
                    "doc": doc,
                    "score": (1 - weight_vector) * bm25_scores[i],
                }

        sorted_docs = sorted(
            combined.items(), key=lambda x: x[1]["score"], reverse=True
        )

        return [item["doc"] for _, item in sorted_docs[:top_k]]
            


"""
Index Creation 
"""
def retriever_chain(chunks: list[Document]):

    global _faiss_vectorstore, _bm25_retriever

    try:
        if os.path.exists("faiss_index"):
            _faiss_vectorstore = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info("Loaded FAISS index from disk")

        else:
            _faiss_vectorstore = FAISS.from_documents(chunks, embeddings)
            _faiss_vectorstore.save_local("faiss_index")
            logging.info("Saved FAISS index to disk")

        _bm25_retriever = BM25Retriever.from_documents(chunks)
        _bm25_retriever.k = 5

        return True

    except Exception as e:
        logging.error(f"Error: {e}")
        return False


""" 
Combined Retriever Getter 
"""

def get_combined_retriever():

    global _faiss_vectorstore, _bm25_retriever

    if _faiss_vectorstore is None or _bm25_retriever is None:
        return None

    return CombinedRetriever(_faiss_vectorstore, _bm25_retriever)
