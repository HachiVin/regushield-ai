import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def run_ingestion():
    print("Membaca dokumen PDF...")
    pdf_path = os.path.join("data", "raw", "a. Kode Etik (Indonesia)_2025.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"File tidak ditemukan di: {pdf_path}")
        return

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Memotong teks menjadi paragraf kecil (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Memastikan metadata 'source' sama persis dengan nama file untuk filter JSON kita
    for chunk in chunks:
        chunk.metadata["source"] = "a. Kode Etik (Indonesia)_2025.pdf"

    print("Menghasilkan embeddings dan menyimpan ke ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    vector_db_path = os.path.join("data", "vector_db")
    db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    
    # Memasukkan teks ke database
    db.add_documents(chunks)
    
    print(f"BERHASIL! {len(chunks)} potongan teks telah dimasukkan ke Vector DB.")

if __name__ == "__main__":
    run_ingestion()