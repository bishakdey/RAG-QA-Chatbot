import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.title("RAG QA with Groq and LangChain")

st.sidebar.markdown("## Upload your PDF files")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader"
)

# Save uploaded files to './pds' directory
if uploaded_files:
    os.makedirs("./pds", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join("./pds", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")

st.markdown("## Ask questions about your PDF documents!")

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant. Use the following pieces of context to answer the question at the end with the most accurate response.
    <context>
    {context}
    <context>
    Question: {query}
    """
)


def vector_embeddings():
    if "vectors" not in st.session_state:
        # Initialize Hugging Face Embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load PDF documents from './pds' directory
        st.session_state.loader = PyPDFDirectoryLoader('./pds')
        st.session_state.docs = st.session_state.loader.load()

        # Split documents into text chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


# Input from user
prompt1 = st.text_input("Enter your question here")

if st.sidebar.button("Process PDFs"):
    with st.spinner("Processing documents and creating vector store..."):
        vector_embeddings()
    st.success("Vector Store created with FAISS")


def create_retrieval_chain(retriever, llm):
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please upload and process PDFs first using the 'Process PDFs' button.")
    else:
        retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        start = time.process_time()
        response = retrieval_chain.run(prompt1)  # ✅ Use .run(prompt1) directly
        end = time.process_time()

        st.write("Answer:")
        st.write(response)
        st.write(f"Response Time: {end - start:.2f} seconds")
