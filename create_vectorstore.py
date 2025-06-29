from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import tempfile
import os

VECTORDATABASE_PATH = "vectorDB/faiss_index"

# save files locally ----
def save_vectorDatabase(vectorstore):
    os.makedirs(VECTORDATABASE_PATH,exist_ok=True)
    vectorstore.save_local(VECTORDATABASE_PATH)
def load_vectorDatabase():
    if os.path.exists(VECTORDATABASE_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# for single pdf file-------
# @st.cache_resource
# def process_pdf_and_create_vectorstore(uploaded_file):
#     # Save uploaded PDF temporarily
#     temp_path = "temp_uploaded_file.pdf"
#     with open(temp_path, "wb") as f:
#         f.write(uploaded_file.read())

#     # Load and split
#     loader = PyPDFLoader(temp_path)
#     pages = loader.load_and_split()
    
#     # Text splitter
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = splitter.split_documents(pages)

#     # Embedding model
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # FAISS vector store
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     return vectorstore

@st.cache_resource
def process_add_and_pdfs_to_vectorDB(files,_existing_vectorstore = None):
    all_docs =[]
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # load and split
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()

        # text splitter 
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
        docs = splitter.split_documents(pages)
        all_docs.extend(docs)

        os.unlink(temp_file_path)
    
    vector_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if _existing_vectorstore:
        _existing_vectorstore.add_documents(all_docs)
        return _existing_vectorstore
    else:
        return FAISS.from_documents(all_docs,vector_embeddings)