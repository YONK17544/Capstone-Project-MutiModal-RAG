Capstone Project: Multimodal RAG for Financial Data

This project demonstrates a Multimodal Retrieval-Augmented Generation (RAG) system that ingests audio and PDF documents, embeds them into a shared vector space, and retrieves relevant content to generate context-aware summaries or answers using OpenAI GPT-4o-mini.

The system is designed for financial data analysis, but the pipeline is generalizable to other domains.

🧩 Features

Audio Transcription: Uses Whisper to convert audio files (e.g., earnings calls) into text.

PDF Processing: Converts PDF pages into images for visual embedding.

Multimodal Embeddings: Uses Sentence-Transformers CLIP-ViT-B-32 for text and image embeddings.

Similarity-Based Retrieval: Retrieves top-k relevant audio chunks and images based on cosine similarity with a query.

Generative Answering: Combines retrieved text and image context to generate coherent answers with GPT-4o-mini.

Visualization: Supports embedding and retrieval verification via console outputs.

📂 Project Structure
Capstone-Multimodal-RAG/
│
├── audio/                     # Input audio files (e.g., starbucks-q3.mp3)
├── transcript/                # Whisper-generated transcriptions
├── images/                     # PDF pages converted to PNG
├── notebooks/                 # Colab notebooks for pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

🔧 Setup

Install dependencies

pip install -q openai langchain langchain-openai langchain-community openai-whisper sentence-transformers pdf2image
apt-get install -q poppler-utils
pip install --upgrade Pillow


Set your API key

from google.colab import userdata
api_key = userdata.get('genai_course')


Change working directory (example in Colab)

%cd /content/drive/MyDrive/GenAI/RAG/CAPSTONE PROJECT - MultiModal Data

⚙️ Pipeline Overview
1️⃣ Audio Transcription

Load the audio file and transcribe using Whisper.

Save transcription to transcript/transcription.txt.

Split text into manageable chunks for embedding.

2️⃣ PDF to Images

Convert PDF pages into PNG images using pdf2image.

Store images in the images/ folder.

3️⃣ Multimodal Embeddings

Text Embeddings: Sentence-Transformer (clip-ViT-B-32) embeds audio transcription chunks.

Image Embeddings: The same model embeds PDF page images.

4️⃣ Similarity Retrieval

Define a query and embed it in the same vector space.

Use cosine similarity to retrieve:

Top-k relevant audio chunks

Top-k relevant images

5️⃣ Generative Answering

Combine retrieved text and images.

Convert images to base64 for multimodal input to GPT-4o-mini.

Define a system prompt (e.g., financial adviser).

Generate a context-aware answer with OpenAI API.

🔎 Example Query
query = "How is the company doing financially?"


Retrieves the most relevant audio chunks from an earnings call transcript.

Finds relevant pages/images from the PDF earnings release.

Generates a concise, informed summary integrating both modalities.

🛠️ Potential Extensions

Extend to multiple audio files and PDFs for a company’s full quarterly reports.

Integrate document metadata for better retrieval filtering.

Add interactive dashboard for querying and visualizing results.

Combine with LangChain RAG pipelines for multi-step reasoning.

Fine-tune embeddings with domain-specific data for higher accuracy.

⚡ Notes

The pipeline supports GPU acceleration if available.

Audio chunks are split into 100-character segments for embedding efficiency.

The system can be generalized for legal, medical, or educational multimodal documents.

GPT-4o-mini provides summaries grounded in retrieved content only.
