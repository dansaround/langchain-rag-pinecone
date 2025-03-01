import os
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Set your API keys for OpenAI
openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize OpenAI Embeddings using LangChain
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Specify which embedding model

# Load all text files from a directory
directory_path = "data/"  # directory path with all the national weather service documents
loader = DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader)  # Load only .txt files
documents = loader.load()

# Use a TextSplitter to split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

# Connect to the Pinecone index using LangChain's Pinecone wrapper
# Add all the split documents into the Pinecone vector database
pinecone_index_name = os.environ['PINECONE_INDEX_NAME']
vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
vectorstore.add_documents(documents=split_documents )

print("Embeddings from text files residing in the directory, created, and inserted in Pinecone Vector Database successfully!")