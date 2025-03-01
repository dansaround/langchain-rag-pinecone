import os
import openai
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize OpenAI and Pinecone
openai.api_key = os.environ['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
pinecone_index_name = os.environ['PINECONE_INDEX_NAME']
vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)

# Initialize LLM and prompt template
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai.api_key, temperature=0.7)
prompt_template = PromptTemplate(
    template="""
    Use the following context to answer the question as accurately as possible:
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)
llm_chain = prompt_template | llm | StrOutputParser()

def process_and_store_knowledge(knowledge_base: str):
    """Process and store knowledge base text in Pinecone"""
    logger.info("Processing knowledge base text")
    if not knowledge_base.strip():
        raise ValueError("Knowledge base text cannot be empty")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    split_documents = text_splitter.create_documents([knowledge_base])
    
    # Store in Pinecone
    vector_store.add_documents(documents=split_documents)
    logger.info(f"Processed and stored {len(split_documents)} documents in Pinecone")

def query_knowledge_base(question: str):
    """Query the knowledge base with a question"""
    logger.info(f"Processing query: {question}")
    if not question.strip():
        raise ValueError("Question cannot be empty")

    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate answer
    answer = llm_chain.invoke({"context": context, "question": question})
    return answer

def main(knowledge_base: str, question: str):
    """Main function to process knowledge base and query"""
    try:
        # First, process and store the knowledge base
        process_and_store_knowledge(knowledge_base)
        
        # Then, query with the question
        answer = query_knowledge_base(question)
        print("Answer:", answer)
        return answer
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python app.py 'knowledge_base_text' 'question'")
        sys.exit(1)
    
    knowledge_base = sys.argv[1]
    question = sys.argv[2]
    main(knowledge_base, question)