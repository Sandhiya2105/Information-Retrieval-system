import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write("Reply:", response['answer'])

def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF 💬")

    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                texts = text_splitter.split_text(raw_text)

                embeddings = HuggingFaceEmbeddings()
                document_search = FAISS.from_texts(texts, embeddings)

                # ✅ Use a supported HuggingFace model
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",  # You can try flan-t5-xl too
                    model_kwargs={"temperature": 0.5, "max_length": 500}
                )

                memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )

                conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=document_search.as_retriever(),
                    memory=memory
                )

                st.session_state.conversation = conversation_chain

if __name__ == '__main__':
    main()
