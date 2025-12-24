# import sentence_transformers
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# import py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss
import os

DOCS_PATH = os.path.abspath("docs")
# FAISS_INDEX_PATH = os.path.abspath(os.path.join("faiss_index"))


def load_chunk_pdfs(embedding_model, doc_dir = DOCS_PATH):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for file in os.listdir(doc_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(doc_dir, file)
            loader = PyPDFLoader(pdf_path)
            chunks.extend(loader.load_and_split(text_splitter))

    index = faiss.IndexFlatL2(len(embedding_model.embed_query("test")))
    vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={})

    vector_store.add_documents(chunks)
    return vector_store

def clear_index_path():
    for file in os.listdir(FAISS_INDEX_PATH):
        os.remove(os.path.join(FAISS_INDEX_PATH, file))


def retrieve_from_index(index, query, k=5):
    retriever = index.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return docs

def retrieve_all_from_index(db):
    num_docs = len(db.index_to_docstore_id)

    docs = []
    for i in range(num_docs):
        doc_id = db.index_to_docstore_id[i]
        document = db.docstore.search(doc_id)
        if document:
            docs.append(document)
    return docs
    

if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = load_chunk_pdfs(embedding_model)
    print(index)
    docs = retrieve_from_index(index, "Explain linear regression", k=4)
    print(docs)
    # clear_index_path()


            
    