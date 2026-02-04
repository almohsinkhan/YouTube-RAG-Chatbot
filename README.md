
# 🎥 YouTube Transcript RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) application that allows users to chat with any YouTube video. It fetches the video transcript, indexes it into a local vector database, and uses a quantized LLM (TinyLlama) to answer questions based strictly on the video content.
🚀 Features

    Automated Transcript Fetching: Uses youtube-transcript-api to pull data directly from YouTube.

    Efficient Text Splitting: Implements RecursiveCharacterTextSplitter to maintain context across chunks.

    Local Vector Store: Uses FAISS with all-MiniLM-L6-v2 embeddings for fast, semantic search.

    Edge-Device Optimized: Utilizes TinyLlama-1.1B, a small yet capable LLM designed to run on modest hardware.

    Structured Prompting: Uses custom Chat Templates to ensure concise, hallucination-free answers.

## 🛠️ Tech Stack

    Orchestration: LangChain

    LLM: TinyLlama-1.1B-Chat

    Vector Database: FAISS

    Embeddings: HuggingFace sentence-transformers

    Language: Python 3.10+

## 📋 Prerequisites

Before running the script, ensure you have the following installed:
Bash

pip install youtube-transcript-api langchain langchain-community langchain-huggingface faiss-cpu transformers torch

📖 How It Works

    Processing: The YouTubeProcessor class downloads the transcript and splits it into 1000-character chunks with a 200-character overlap.

    Embedding: Chunks are converted into vectors and stored in a FAISS index.

    Retrieval: When a question is asked, the system finds the top k most relevant chunks.

    Generation: The context and question are formatted into a system/user prompt for the LLM to generate a concise answer.

## 💻 Usage

Run the main script and follow the CLI prompts:
Bash

python youtube.py

    Enter the YouTube Video ID (e.g., DNCn1BpCAUY).

    Enter your Question.

⚠️ Known Issues & Fixes

    CUDA Out of Memory: If your GPU has less than 4GB VRAM, the model is configured to use low_cpu_mem_usage=True. For extremely low memory, consider forcing device='cpu' in the LLM pipeline.

    Hallucinations: The prompt is engineered to force the model to say "I don't know" if the answer isn't in the transcript.