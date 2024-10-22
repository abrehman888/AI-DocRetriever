# AI-DocRetriever

This project is an implementation of a Retrieval-Augmented Generation (RAG) System that processes HTML documents by splitting them based on headers, chunking the text, and storing the chunks in a Qdrant Vector Store. It also utilizes OpenAI’s language model for conversational responses based on the context from the retrieved documents. The project uses the LangChain framework and various components like embedding models and text splitters.

Features  
Load and split HTML files based on header tags.  
Chunk the content into smaller segments for efficient document retrieval.  
Store the document chunks in a Qdrant Vector Store.  
Perform similarity-based searches to retrieve relevant chunks.  
Use OpenAI’s GPT model to answer user queries based on retrieved content.  
  
Installation
To run this project locally, follow the steps below.  
  
Requirements:  
Python 3.8+  
Qdrant  
OpenAI API Key  
Required Python packages (installed in the next step)  

Step 1: Clone the repository  
git clone https:https://github.com/abrehman888/AI-DocRetriever.git  
cd AI-DocRetriever  

Step 2: Install dependencies  
Install the required Python packages using pip:  
pip install qdrant_client langchain_huggingface langchain-community langchain-qdrant pypdf openai langchain transformers langchain_text_splitters  

Step 3: Set Up Environment Variables  
For security reasons, sensitive information like API keys and service URLs should not be hardcoded. Store your credentials in environment variables.  
OPENAI_API_KEY=your_openai_api_key  
QDRANT_URL=your_qdrant_url  
QDRANT_API_KEY=your_qdrant_api_key  

Make sure to replace your_openai_api_key, your_qdrant_url, and your_qdrant_api_key with the actual values.  
 
You can load these environment variables in Python using the os library  
 import os  
openai.api_key = os.getenv("OPENAI_API_KEY")  
qdrant_url = os.getenv("QDRANT_URL")  
qdrant_key = os.getenv("QDRANT_API_KEY")  


Step 4: Run the Script  
Once everything is set up, you can run the script:   
python main.py  # Adjust the filename if necessary  

Usage  
The script processes an HTML file, chunks it into smaller segments, stores the segments in a vector store, and allows you to query the content via OpenAI’s GPT model.  

Here’s a brief summary of the steps in the code:  

HTML Loading & Splitting: The script loads an HTML file and splits it based on header tags (h1, h2, h3, etc.).  
Chunking: The split content is chunked into smaller pieces for easy retrieval.  
Storing in Qdrant: The document chunks are stored in a Qdrant Vector Store for similarity search.  
Querying: A user can query the system, and the relevant document chunks are retrieved and passed to the GPT model to generate a response.  

Example Query   
query = "Who is Elon Musk?"
response = chain.invoke({"question": query})
print(response)
