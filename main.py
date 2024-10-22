Install required libraries
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
import re
import openai
import os

# Load and split HTML content
file_path = "path_to_your_file.html"  # Replace with your actual file path
loader = BSHTMLLoader(file_path)
data = loader.load_and_split()

# Set up headers for splitting
headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"), ("h4", "Header 4")]
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text_from_file(file_path)

# Configure text splitter
chunk_size = 500
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Split documents
splits = text_splitter.split_documents(html_header_splits)

# Process each document chunk and remove unnecessary whitespace
doc_list = []
chunk_count = 1
for doc in splits:
    pg_split = text_splitter.split_text(doc.page_content)
    original_metadata = doc.metadata
    
    for i, text_chunk in enumerate(pg_split):
        metadata = {
            "source": original_metadata.get("source", ""),
            "author": ".........", #Give metadata value according to your document and data
            "description": "................",
            "keywords": "..................",
            "chunk_no": chunk_count
        }
        
        doc_string = Document(page_content=re.sub(r'\s+', ' ', text_chunk).strip(), metadata=metadata)
        doc_list.append(doc_string)
        chunk_count += 1

# Initialize Qdrant Vector Store (Sensitive data removed, use environment variables or config files for security)
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
collection_name = "your_collection_name"

# Replace with your actual embedding model and API keys
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Create a QdrantClient instance
client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
qdrant = QdrantVectorStore(client=client, embedding=embed_model, collection_name=collection_name)

# Set up LLM for answering questions (Sensitive data removed)
openai.api_key = os.getenv("OPENAI_API_KEY")
llm_name = "gpt-4o-mini"
llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai.api_key, temperature=0)

# Define chat chain for conversational context
def format_docs(docs):
    # This function formats the retrieved documents for the language model's prompt.
    # It takes a list of documents and returns a single string containing their content,
    # separated by double newlines.
    return "\n\n".join(doc.page_content for doc in docs)
history = []
prompt_str="""
Answer the user question based only on the following context:

conversation_history: {chat_history}

Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)
num_chunks=3
retriever = qdrant.as_retriever(search_type="similarity",
                                        search_kwargs={"k": num_chunks})
chat_llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai.api_key, temperature=0)
query_fetcher= itemgetter("question")
history_fetcher=itemgetter("chat_history")
setup={"question":query_fetcher,"chat_history":history_fetcher | retriever | format_docs}
_chain = setup |_prompt | chat_llm | StrOutputParser()
query="Write your query"
response=_chain.invoke({"question":query, "chat_history":"\n".join(str(history))})
print(response)
query="user_question:"+query
response="ai_response:"+response
history.append((query, response))
