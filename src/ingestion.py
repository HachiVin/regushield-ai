import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_single_file(file_path: str) -> int:
    """
    Membaca, memotong (chunking), dan memasukkan satu file PDF ke ChromaDB.
    Fungsi ini dipanggil oleh app.py saat tombol Upload digunakan.
    Mengembalikan jumlah potongan teks (chunks) yang berhasil diindeks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=400
    )
    chunks = text_splitter.split_documents(documents)

    filename = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata["source"] = filename

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    vector_db_path = os.path.join("data", "vector_db")
    db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    
    db.add_documents(chunks)

    return len(chunks)