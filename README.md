# RAG Pipeline

A Jupyter notebook that demonstrates a basic Retrieval-Augmented Generation (RAG) workflow with LangChain, Sentence Transformers, ChromaDB, and an optional OpenAI chat model. This is a learning project, not a production-ready application.

## How it works

```text
Text/PDF files → Documents → Chunks → Embeddings → ChromaDB
                                                    ↓
Question → Query embedding → Retrieved context → Optional LLM answer
```

The notebook walks through:

1. Loading a text file and PDFs with LangChain document loaders.
2. Splitting PDF documents into chunks (500 characters, with 50 characters of overlap).
3. Creating embeddings with `all-MiniLM-L6-v2`.
4. Persisting document chunks in a ChromaDB collection.
5. Retrieving relevant chunks for a question.
6. Optionally sending the retrieved context and question to an OpenAI chat model.

## Repository contents

```text
.
├── README.md
├── RAG_pipeline.ipynb
└── Data/
    ├── python.txt
    ├── NIPS-2017-attention-is-all-you-need-Paper.pdf
    ├── pdfs/
    │   ├── research2.pdf
    │   └── Deep_Learning_PyTorch_CNN_RNN_Project_Report.pdf
    └── vector_store/       # Existing persisted ChromaDB data
```

The notebook loads the standalone paper directly from `Data/` and loads every PDF in `Data/pdfs/`. It creates `Data/vector_store/` if the directory does not already exist. Run notebook cells with the repository root as the working directory so these relative paths resolve.

## Setup

Use Python and Jupyter in a virtual environment. From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install the notebook dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install langchain langchain-core langchain-community \
  langchain-text-splitters sentence-transformers chromadb pypdf \
  scikit-learn langchain-openai ipykernel
```

Open `RAG_pipeline.ipynb` in VS Code or Jupyter and select the virtual environment as the notebook kernel. The embedding model may be downloaded the first time it is used.

## Running the pipeline

- The loading, chunking, embedding, storage, and retrieval steps run locally and do not require an OpenAI key.
- The final answer-generation step calls an OpenAI model and requires a valid API key.
- The notebook's current LLM cell does not read `OPENAI_API_KEY`. Update it to load the key from the environment before using that step; never paste a key into a notebook or commit it to source control.

> **Security:** An API credential is currently embedded in the notebook. Revoke or rotate it with the provider before using the LLM step. The notebook was intentionally left unchanged.

The retriever currently uses Chroma's returned distance and calculates `1 - distance` as its threshold score. Treat that value as the notebook's distance-derived score; it is not necessarily a cosine-similarity score.

## Notes

- Changing chunk size or overlap can affect retrieval results.
- Embedding and retrieval quality depend on the source documents and model.
- The existing vector store contains persisted local data. Re-running ingestion can add another copy of the chunks to the collection.
- No automated test suite or dependency lockfile is included.
