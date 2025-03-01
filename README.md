# Document RAG System

This project implements a Retrieval-Augmented Generation (RAG) system using Langchain and Pinecone.

## Prerequisites

Before starting, make sure you have Python installed on your system.

## Installation

1. Install the required dependencies:

```
bash
pip install -r requirements.txt
```

## Setup and Usage

Follow these steps in order:

### 1. Create Pinecone Index

First, you need to create a Pinecone index to store your document embeddings:

```bash
python create_pinecone_index.py
```

### 2. Generate Document Embeddings

Next, process your documents and create embeddings using Langchain:

```bash
python generate_embeddings.py
```

### 3. Run RAG System

Finally, run the RAG system to query your documents:

```bash
python run_rag.py
```

## Important Notes

- Make sure you have set up your environment variables correctly before running the scripts
- Ensure you have sufficient API credits for Pinecone and any other services you're using
- Check that your documents are in the correct format and location before processing

## Troubleshooting

If you encounter any issues:

1. Verify all dependencies are correctly installed
2. Check your API keys and environment variables
3. Ensure your Pinecone index is properly initialized
4. Confirm your document format is supported

## Additional Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Langchain Documentation](https://python.langchain.com/docs/get_started/introduction)
