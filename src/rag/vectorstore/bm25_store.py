from rank_bm25 import BM25Okapi
from src.models.document_schema import DocumentMetadata
from src.rag.ingestion.document_ingestion import (
    load_document,
    chunk_documents
)



BM25_INDEX = None
BM25_DOCUMENTS = []


#----------------------------------------------------------------------------------------------------------#
# This function builds the BM25 index for the given document chunks.

def build_bm25_index(chunks):

    global BM25_INDEX
    global BM25_DOCUMENTS

    BM25_DOCUMENTS.extend(chunks)

    tokenized_docs = [

        chunk.page_content.split()

        for chunk in BM25_DOCUMENTS
    ]

    BM25_INDEX = BM25Okapi(tokenized_docs)

    print(f"BM25 Index Created. Total Chunks={len(BM25_DOCUMENTS)}")



#----------------------------------------------------------------------------------------------------------#
# This function performs a BM25 search for the given query and department ID, returning the top k results.


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

    scores = BM25_INDEX.get_scores(tokenized_query)
    
    scored_docs = []

    for chunk, score in zip(BM25_DOCUMENTS, scores):

        if (chunk.metadata.get("department_id")!= department_id):
            continue

        scored_docs.append((chunk, score))


    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    results = []

    for chunk, score in scored_docs[:top_k]:

        results.append({
            "chunk_id": chunk.metadata.get("chunk_id"),
            "document_id": chunk.metadata.get("document_id"),
            "department_id": chunk.metadata.get("department_id"),
            "file_name": chunk.metadata.get("file_name"),
            "version": chunk.metadata.get("version"),
            "is_active": chunk.metadata.get("is_active"),
            "text": chunk.page_content,
            "bm25_score": float(score)
        })

    return results



#----------------------------------------------------------------------------------------------------------#
# This function rebuilds the BM25 index by loading all active documents from the database, chunking them, and then building the index with the chunks.


def rebuild_bm25_index(db):

    active_documents = (
        db.query(DocumentMetadata)
        .filter(
            DocumentMetadata.is_active == True
        )
        .all()
    )

    all_chunks = []

    for document in active_documents:

        try:
            documents = load_document(document.file_path)
            chunks = chunk_documents(documents)

            for chunk in chunks:

                chunk.metadata.update({
                    "document_id":document.document_id,
                    "department_id":document.department_id,
                    "file_name":document.file_name,
                    "version":document.version,
                    "is_active":document.is_active
                })

            all_chunks.extend(chunks)

        except Exception as e:

            print(f"Failed loading {document.file_name}: {e}")

    if all_chunks:

        build_bm25_index(all_chunks)

        print(f"BM25 rebuilt with {len(all_chunks)} chunks")
        



#----------------------------------------------------------------------------------------------------------#
# This function removes all BM25 chunks associated with a specific document ID and rebuilds the index accordingly.



def remove_document_chunks(
    document_id: int
):

    global BM25_DOCUMENTS
    global BM25_INDEX

    BM25_DOCUMENTS = [

        chunk

        for chunk in BM25_DOCUMENTS

        if chunk.metadata.get("document_id") != document_id
    ]

    if BM25_DOCUMENTS:

        tokenized_docs = [

            chunk.page_content.split()

            for chunk in BM25_DOCUMENTS
        ]

        BM25_INDEX = BM25Okapi(tokenized_docs)

    else:

        BM25_INDEX = None

    print(f"Removed BM25 chunks for document {document_id}")
    