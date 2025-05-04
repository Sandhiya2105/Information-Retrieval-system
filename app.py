import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain

def user_input(user_question):
    if "conversation" in st.session_state:
        response = st.session_state.conversation({'question': user_question})
        st.write("Reply:", response['answer'])
    else:
        st.warning("Please upload and process a PDF before asking questions.")

def main():
    st.set_page_config("Information-Retrieval-System")
    st.header("Chat with PDF 💬")

    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                # Extract text
                raw_text = get_pdf_text(pdf_docs)
                
                # Split text
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # Get conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                st.success("PDF processed. You can now ask questions!")

if __name__ == '__main__':
    main()


