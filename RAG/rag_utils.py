# import sentence_transformers
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# import py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

DOCS_PATH = os.path.abspath("docs")
FAISS_INDEX_PATH = os.path.abspath(os.path.join("faiss_index"))


def load_chunk_pdfs(doc_dir:DOCS_PATH, embedding_model):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for file in os.listdir(doc_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(doc_dir, file)
            loader = PyPDFLoader(pdf_path)
            chunks.extend(loader.load_and_split(text_splitter))

    index = FAISS.from_documents(chunks, embedding_model)
    index.save_local(FAISS_INDEX_PATH)
    return index

if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = load_chunk_pdfs(DOCS_PATH, embedding_model)
    print(index)


            
    