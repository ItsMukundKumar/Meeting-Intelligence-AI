from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from core.summarize import split_transcript

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )


def build_vector_store(transcript: str):

    print("Building FAISS Vector Store...")

    chunks = split_transcript(transcript=transcript, chunk_size=500, chunk_overlap=50)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store


def get_retriever(vectorstore, k: int = 4):

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
