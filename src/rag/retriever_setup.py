import os

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

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
    


# def get_retriever():
#     """
#     Get a retriever instance from the FAISS vectorstore.

#     Returns the retriever tool that can search documents stored by retriever_chain().
#     If no documents have been uploaded yet, creates a retriever with a dummy document.

#     Returns:
#         A retriever instance if the vectorstore is initialized, otherwise None.

#     Raises:
#         Exception: If vector store initialization fails.
#     """

#     global _faiss_vectorstore

#     try:
#         if _faiss_vectorstore is not None:
#             retriever = _faiss_vectorstore.as_retriever()
#             print("Using existing FAISS vectorstore with uploaded documents")
#         else:
#              # No documents uploaded yet, create dummy for initialization
#             print("No documents uploaded yet, creating dummy vectorstore")
#             from langchain_core.documents import Document as LangChainDocument

#             dummy_doc = LangChainDocument(
#                 page_content="This is a dummy document. Please upload your documents to enable retrieval.", 
#                 metadata={"source": "dummy"}
#             )
#             _faiss_vectorstore = FAISS.from_documents(
#                 documents=[dummy_doc], 
#                 embedding=embeddings
#             )
#             retriever = _faiss_vectorstore.as_retriever()

#         description = None

#         retriever_tool = create_retriever_tool(
#             retriever,
#             "retriever_customer_uploaded_documents",
#             f"Use this tool **only** to answer questions about: {description}\n"
#             "Don't use this tool to answer anything else."
#         )

#         return retriever_tool
#     except Exception as e:
#         print(f"Error initializing retriever: {e}")
#         raise Exception(e)

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




     