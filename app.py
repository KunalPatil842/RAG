import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="US Census 2021-22 Chatbot", layout="wide")

# Title and description
st.title("US Census 2021-22 Chatbot")
st.write("""
This chatbot answers questions based on the US Census 2021-22 data. 
Enter your question below and click the button to get the most accurate response based on the provided context.
""")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")


# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner('Loading and processing documents...'):
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:20])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
                                                            st.session_state.embeddings)
            st.success('Vector Store DB is ready!')


# Input field for user question
prompt1 = st.text_input("Enter Your Question from Documents")

# Button to create document embeddings
if st.button("Create Document Embeddings"):
    vector_embedding()

# Process the user question
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please create the document embeddings first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start_time

        st.write(f"Response time: {response_time:.2f} seconds")
        st.write(response['answer'])

        # Expander for document similarity search results
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

# Add some styling
st.markdown("""
<style>
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    .css-1d391kg {
        padding: 0 1rem;
    }
    .stButton button {
        width: 100%;
        height: 3rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 1rem;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)
