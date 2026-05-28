from bm25_store import BM25Okapi

BM25_INDEX = None
BM25_DOCUMENTS = []

def build_bm25_index(chunks):

    global BM25_INDEX
    global BM25_DOCUMENTS

    BM25_DOCUMENTS = chunks

    tokenized_docs = [

        chunk.page_content.split()

        for chunk in chunks
    ]

    BM25_INDEX = BM25Okapi(
            tokenized_docs
    )

    print("BM25 Index Created")



def bm25_search(query: str, top_k: 5):

    global BM25_INDEX
    global BM25_DOCUMENTS


    toeknized_query  = query.split()

    scores = BM25_INDEX.get_scores(
        toeknized_query
    )

    scored_docs = list(zip(BM25_DOCUMENTS, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    results = []

    for chunk, score in scored_docs[:top_k]:

        results.append(
            {
                "text": chunk.page_content,

                "chunk_id": chunk.metadata.get("chunk_id"),

                "bm25_score": float(score)
            }
        )

    return results


