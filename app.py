import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=groq_api_key
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=1000,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    if text_chunks == []:
        st.error("The above file couln't be processed", icon="⚠️")
        return
    vectorstore = FAISS.from_texts(text_chunks, HuggingFaceEmbeddings())
    return vectorstore

# setting up the RAG chain
def get_conversation_chain(vectorstore):
    if vectorstore:
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If the question is not related to the context provided,say that you "
            "don't know the answer. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )
        st.toast("You are ready to chat", icon="🎉")
        return rag_chain
    else:
        return


def main():
    # setting up page title
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:",
        layout="wide"
    )
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
      width: 20%;
    }
    .chat-message .avatar img {
      max-width: 78px;
      max-height: 78px;
      border-radius: 50%;
      object-fit: cover;
    }
    .chat-message .message {
      width: 80%;
      padding: 0 1.5rem;
      color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Chat with multiple PDFs :books:")

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "display_chat_history" not in st.session_state:
        st.session_state.display_chat_history = []
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            key="pdf_docs"
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    st.session_state.raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(st.session_state.raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.rag_chain = get_conversation_chain(vectorstore)
                st.success("Documents processed successfully!")
            else:
                st.warning("Please upload PDFs before processing.", icon="⚠️")

    # Main chat interface
    if st.session_state.raw_text and st.session_state.rag_chain:
        # Display chat messages
        for chat in st.session_state.display_chat_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

        # Chat input
        user_input = st.chat_input("Ask a question about your documents:")
        if user_input:
            st.session_state.display_chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            try:
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.invoke(
                        {"input": user_input, "chat_history": st.session_state.chat_history}
                    )
                st.session_state.display_chat_history.append({"role": "assistant", "content": response["answer"]})
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])

                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=response["answer"]),
                ])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload and process PDFs to start chatting.")

if __name__ == "__main__":
    main()