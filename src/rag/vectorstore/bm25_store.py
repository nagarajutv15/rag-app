from rank_bm25 import BM25Okapi

BM25_INDEX = None
BM25_DOCUMENTS = []

# BM25 Index Building Function. Takes a list of chunks (each chunk is a document with text and metadata) and 
# builds a BM25 index for efficient retrieval. 
# The function tokenizes the text of each chunk and stores the tokenized documents in the BM25 index.]

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



# BM25 Search Function. Returns the top_k most relevant chunks based on the BM25 score. 
# Each result includes the chunk's text, metadata, and BM25 score.

def bm25_search(
    query: str,
    top_k: int = 5
):
    
    global BM25_INDEX
    global BM25_DOCUMENTS

    if BM25_INDEX is None:
            return []
        

    tokenized_query  = query.split()

    scores = BM25_INDEX.get_scores(
        tokenized_query
    )

    scored_docs = list(zip(BM25_DOCUMENTS, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    results = []

    for chunk, score in scored_docs[:top_k]:

        results.append({

            "text": chunk.page_content,

            "chunk_id": chunk.metadata.get(
                "chunk_id"
            ),

            "document_id": chunk.metadata.get(
                "document_id"
            ),

            "department_id": chunk.metadata.get(
                "department_id"
            ),

            "file_name": chunk.metadata.get(
                "file_name"
            ),

            "version": chunk.metadata.get(
                "version"
            ),

            "is_active": chunk.metadata.get(
                "is_active"
            ),
            
            "uploaded_by": chunk.metadata.get(
                "uploaded_by"
            ),

            "uploaded_at": chunk.metadata.get(
                "uploaded_at"
            ),

            "bm25_score": float(score)
        })

    return results


