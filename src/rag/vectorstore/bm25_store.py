from rank_bm25 import BM25Okapi

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


def bm25_search(
    query: str,
    department_id: str,
    top_k: int = 5
):
    
    global BM25_INDEX
    global BM25_DOCUMENTS

    if BM25_INDEX is None:
        return []

    tokenized_query = query.split()

    scores = BM25_INDEX.get_scores(
        tokenized_query
    )

    scored_docs = list(
        zip(BM25_DOCUMENTS, scores)
    )

    scored_docs = []

    for chunk, score in zip(BM25_DOCUMENTS, scores):

        if (chunk.metadata.get("department_id")!= department_id):
            continue

        scored_docs.append(
            (chunk, score)
        )

    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for chunk, score in scored_docs[:top_k]:

        results.append({

            "chunk_id":
                chunk.metadata.get(
                    "chunk_id"
                ),

            "document_id":
                chunk.metadata.get(
                    "document_id"
                ),

            "department_id":
                chunk.metadata.get(
                    "department_id"
                ),

            "file_name":
                chunk.metadata.get(
                    "file_name"
                ),

            "version":
                chunk.metadata.get(
                    "version"
                ),

            "is_active":
                chunk.metadata.get(
                    "is_active"
                ),

            "text":
                chunk.page_content,

            "bm25_score":
                float(score)
        })

    return results