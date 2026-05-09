
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Global variable to store the FAISS vectorstore instance
# This ensures get_retriever() can access documents stored by retriever_chain()
_faiss_vectorstore = None


def retriever_chain(chunks: list[Document]):
    """
    Initialize and store documents in FAISS vector database.

    Args:
        chunks: List of document chunks to store.

    Returns:
        Boolean indicating success of the operation.
    """

    global _faiss_vectorstore

    try:
        vectorstore = FAISS.from_documents(
            documents=chunks, 
            embedding=embeddings
        )
        _faiss_vectorstore = vectorstore

        print("FAISS vector store initialized with documents")
        print(f"Vectorstore contains {len(chunks)} document chunks")
        return True

    except Exception as e:
        print(f"Error storing documents in FAISS: {e}")
        return False
    


def get_retriever():
    global _faiss_vectorstore

    try:
        if _faiss_vectorstore is not None:
            print("Using existing FAISS vectorstore with uploaded documents")
            return _faiss_vectorstore.as_retriever(search_kwargs={"k": 3})

        else:
            print("No documents uploaded yet")
            return None

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise Exception(e)




     