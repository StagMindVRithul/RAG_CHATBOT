import streamlit as st
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

# Load API keys
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit Config
st.set_page_config(page_title="🤖 AI PDF Assistant", page_icon="📚", layout="wide")

# Sidebar: API Key & PDF Upload
st.sidebar.title("🔑 API Key & Docs")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
session_id = st.sidebar.text_input("Session ID", value="default")

def clear_chat():
    if "store" in st.session_state and session_id in st.session_state.store:
        st.session_state.store[session_id].clear()

st.sidebar.button("🧹 Clear Chat Memory", on_click=clear_chat)

# PDF Upload
st.sidebar.markdown("## 📂 Upload your PDFs")
upload_files = st.sidebar.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True)

# Session Storage
if "store" not in st.session_state:
    st.session_state.store = {}

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = None

# Main Title
st.title("🤖 Conversational RAG Chatbot")
st.markdown("Ask me anything about your uploaded PDFs! 📝")

# API Key Check
if not api_key:
    st.warning("🔐 Please provide your Groq API Key in the sidebar.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It")

# PDF Processing
if upload_files and not st.session_state.retriever:
    with st.spinner("📥 Processing uploaded PDFs..."):
        documents = []
        for uploaded_file in upload_files:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        st.session_state.retriever = retriever

    st.success("✅ PDFs processed and embedded!")

# Setup Chain Only After Retriever
if st.session_state.retriever and not st.session_state.conversational_chain:
    # Prompts
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and the latest user question, rewrite it as a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an AI assistant answering user questions."
        "Use the retrieved context to answer concisely (max 3 sentences)."
        "If unsure, say 'I don’t know'.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    st.session_state.conversational_chain = conversational_chain

# Chat Interface
if st.session_state.conversational_chain:
    session_history = st.session_state.store.get(session_id, ChatMessageHistory())

    # Display Chat History as Bubbles
    for msg in session_history.messages:
        with st.chat_message("user" if msg.type == "human" else "assistant"):
            st.markdown(msg.content)

    # User Input
    user_prompt = st.chat_input("Type your question about the PDFs…")

    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking… 🤖"):
                response = st.session_state.conversational_chain.invoke(
                    {"input": user_prompt}, config={"configurable": {"session_id": session_id}}
                )
                st.markdown(response["answer"])

else:
    st.info("⬅️ Please upload PDFs from the sidebar to start chatting.")
