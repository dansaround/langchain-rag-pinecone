import os
import json
import openai
import logging
import traceback

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configurar API Keys y variables
openai.api_key = os.environ['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
pinecone_index_name = os.environ['PINECONE_INDEX_NAME']
vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings,   pool_threads=1)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Buscar los 3 más relevantes
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai.api_key, temperature=0.7)

# Definir Prompt Template
prompt_template = PromptTemplate(
    template="""
    Use the following context to answer the question as accurately as possible:
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)

# Crear LLM Chain
llm_chain = prompt_template | llm | StrOutputParser()

# Función para procesar el texto y almacenarlo en Pinecone
def process_text_and_store(knowledge_base: str):
    logger.info("Iniciando procesamiento de texto para almacenamiento")
    if not knowledge_base.strip():
        logger.error("Se recibió un texto vacío")
        raise ValueError("El texto proporcionado está vacío.")

    # Dividir el texto en fragmentos
    chunk_size = min(1000, len(knowledge_base) // 3)
    chunk_overlap = chunk_size // 3
    logger.info(f"Dividiendo texto en fragmentos: chunk_size={chunk_size}, overlap={chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.create_documents([knowledge_base])  
    logger.info(f"Texto dividido en {len(split_documents)} fragmentos")

    # Almacenar los embeddings usando ThreadPoolExecutor
    logger.info("Iniciando almacenamiento de embeddings en Pinecone")
    with ThreadPoolExecutor() as executor:
        executor.submit(vector_store.add_documents, documents=split_documents)
    logger.info("Embeddings almacenados exitosamente")

    return {"message": "Embeddings creados e insertados en Pinecone con éxito."}
# Función para hacer una consulta en Pinecone
def query_pinecone(question: str):
    logger.info(f"Iniciando consulta con pregunta: {question}")
    if not question.strip():
        logger.error("Se recibió una pregunta vacía")
        return {"error": "La pregunta no puede estar vacía."}

    try:
        logger.info("Generando embedding para la consulta")
        query_embedding = embeddings.embed_query(question)
        logger.info("Realizando búsqueda de similitud en Pinecone")
        search_results = vector_store.similarity_search_by_vector(query_embedding, k=3)
        logger.info(f"Encontrados {len(search_results)} resultados relevantes")

        # Extraer los textos relevantes
        relevant_texts = [doc.page_content for doc in search_results]
        context = "\n\n".join(relevant_texts)

        # Generar respuesta con LLM
        logger.info("Generando respuesta con LLM")
        answer = llm_chain.invoke({"context": context, "question": question})
        logger.info("Respuesta generada exitosamente")

        return {"answer": answer, "relevant_texts": relevant_texts}

    except Exception as e:
        logger.error(f"Error en query_pinecone: {str(e)}")
        return {"error": "Error al consultar Pinecone: " + str(e)}

# Handler de AWS Lambda
def lambda_handler(event, _):
    """ Maneja la solicitud: permite almacenar texto y consultar en la misma petición. """
    logger.info("Iniciando procesamiento de Lambda")
    try:
        body = json.loads(event["body"])
        response = {}

        # Si recibe "text", primero almacena en Pinecone
        if "text" in body:
            logger.info("Procesando solicitud de almacenamiento de texto")
            response["store_result"] = process_text_and_store(body["text"])

        # Si recibe "question", consulta en Pinecone
        if "question" in body:
            logger.info("Procesando solicitud de consulta")
            response["query_result"] = query_pinecone(body["question"])

        # Si no recibió ni "text" ni "question", devuelve error
        if not response:
            logger.error("Solicitud sin parámetros válidos")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Debes enviar 'text' para almacenar o 'question' para consultar."})
            }

        logger.info("Procesamiento completado exitosamente")
        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }

    except Exception as e:
        logger.error(f"Error en lambda_handler: {str(e)}")
        print("Error en Lambda:", traceback.format_exc())  # Para logs en AWS
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
