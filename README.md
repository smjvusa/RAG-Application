# Simple RAG Application (Website Q&A Chatbot)

A Retrieval-Augmented Generation (RAG) application that answers questions about 
the content of a website by retrieving relevant context from a vector database 
and generating grounded responses using Google's Gemini model.

## Problem Statement

Large language models can't answer questions about content they weren't trained 
on, such as a specific organization's website. This project builds a RAG 
pipeline that loads a website's content, indexes it for semantic search, and 
uses retrieved context to generate accurate, grounded answers to user questions 
about that website.

## Tech Stack

- **LLM**: Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai`
- **Embeddings**: Gemini Embeddings (`models/gemini-embedding-001`)
- **Vector Store**: ChromaDB (`langchain-chroma`)
- **Orchestration**: LangChain (document loaders, text splitters, prompt 
  templates, LCEL runnables)
- **Environment**: Google Colab

## How It Works (RAG Pipeline)

### 1. Indexing — Loading
The target website's content is loaded using LangChain's `WebBaseLoader`, which 
fetches and parses the page into a single text document.

### 2. Indexing — Text Splitting
The loaded text is split into smaller chunks to fit within embedding and context 
limits, using `CharacterTextSplitter` (chunk size: 1000, overlap: 200). This 
overlap helps preserve context across chunk boundaries.

### 3. Indexing — Embedding & Vector Storage
Each text chunk is converted into a vector embedding using Gemini's embedding 
model and stored in a **Chroma** vector database for efficient similarity search.

### 4. Retrieval
When a user submits a question, the vector store retrieves the most semantically 
similar chunks to the query using cosine similarity search, returning the top-k 
most relevant pieces of context.

### 5. Generation
The retrieved context and the user's question are combined into a structured 
prompt (system + human message templates) and passed to the Gemini LLM, which 
generates a final answer grounded in the retrieved content.

## Architecture
Website URL → WebBaseLoader → Text Splitter → Gemini Embeddings → Chroma Vector Store

↓

User Question → Retriever (similarity search) → Context + Question → Prompt Template

↓

Gemini LLM → Answer

## Example Usage

```python
question = "What are the courses offered by Fullstack Academy?"
print(chain.invoke(question))
```

The chain retrieves the most relevant chunks from the indexed website and 
generates a concise, context-grounded answer.

## Project Structure
simple-rag-application/

├── rag_application.ipynb

├── README.md

└── requirements.txt

## How to Run

1. Clone this repository:
```bash
git clone https://github.com/smjvusa/simple-rag-application.git
cd simple-rag-application
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

4. Open and run the notebook:
```bash
jupyter notebook rag_application.ipynb
```

**Note**: This notebook was originally built in Google Colab. If running 
locally, replace `userdata.get('gemini_class_key')` with a standard environment 
variable lookup (e.g., `os.getenv('GEMINI_API_KEY')`), and update the Chroma 
`persist_directory` path to a local folder instead of a Colab-specific path.

## Key Concepts Demonstrated

- End-to-end RAG pipeline: load → split → embed → store → retrieve → generate
- LangChain Expression Language (LCEL) for composing retrieval and generation 
  into a single chain using the pipe (`|`) operator
- Semantic similarity search using vector embeddings rather than keyword matching
- Prompt engineering with separate system and human message templates for 
  controlled, context-grounded responses

## Future Improvements

- Support multiple URLs or document types (PDFs, multiple web pages) as 
  knowledge sources
- Add a simple Streamlit/Gradio interface for interactive Q&A instead of 
  notebook-only usage
- Experiment with different chunk sizes/overlaps and compare retrieval quality
- Add source citation in generated answers (showing which chunk/URL the answer 
  was derived from)
- Replace `WebBaseLoader` with a more robust scraper for JavaScript-heavy sites

## Author
Jyothi Vusa
