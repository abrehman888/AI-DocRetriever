import streamlit as st
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_key = st.secrets["QDRANT_API_KEY"]
collection_name = st.secrets["Collection_Name"]
llm_name = "gpt-4o-mini"

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Create a QdrantClient instance
client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

# Initialize QdrantVectorStore
qdrant = QdrantVectorStore(client=client, embedding=embed_model, collection_name=collection_name)

# Streamlit UI
st.image("https://raw.githubusercontent.com/abrehman888/RAG/refs/heads/main/xevensolutions_logo.jpeg", width=100)
st.markdown("<h1 style='text-align: center; font-weight: bold;'> 💬 Chat with Xeven Solution</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: grey;'>Developed by <span style='color: #D83A3A;'>Abdul Rehman</span></p>", unsafe_allow_html=True)

# Prompt user to enter OpenAI API key
api_key = st.text_input("ENTER YOUR OpenAI API KEY", type="password")

# Only show the query input if the API key is provided
if api_key:
    openai.api_key = api_key
    query = st.text_input("🔍 Ask a question about Xeven:")

    # Add a button to submit the query
    if st.button("Get Response"):
        if query:
            try:
                # Set up the prompt template
                prompt_str = """
                Answer the user question based only on the following context:
                {context}
                Question: {question}
                """
                prompt = ChatPromptTemplate.from_template(prompt_str)
                
                # Initialize the retriever
                num_chunks = 3
                retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
                
                # Retrieve relevant documents based on query
                retrieved_docs = retriever.retrieve({"question": query})
                
                # Combine context from retrieved documents
                context = "\n".join([doc.page_content for doc in retrieved_docs])

                # Use ChatOpenAI to get the response
                chat_llm = ChatOpenAI(model_name=llm_name, openai_api_key=api_key, temperature=0)
                response = chat_llm({"context": context, "question": query})

                st.write("Response:", response["content"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter your OpenAI API key to proceed.")

# Option to clear the chat history
if st.button("🧹 Clear History", key="clear_history", help="Clear the chat history to start fresh"):
    st.session_state['chat_history'] = []
    st.success("Chat history cleared!")

# Additional footer styling and separator
st.markdown("<hr style='border: 1px solid #DDD;'/>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'> 🚀 Developed by Abdul Rehman. Powered by Xeven Solutions.</p>", unsafe_allow_html=True)
