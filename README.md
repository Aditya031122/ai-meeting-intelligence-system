# AI Meeting Intelligence System

An end-to-end **AI-powered meeting analysis platform** that converts meeting audio into structured insights.

The system automatically processes meeting recordings to generate **transcripts, summaries, action items, topics, sentiment insights, and semantic search capabilities**. It also allows users to **ask questions about past meetings using Retrieval Augmented Generation (RAG).**

This project demonstrates how modern AI systems combine **speech recognition, NLP, vector databases, and LLMs** to build intelligent productivity tools.

---

# Overview

The **AI Meeting Intelligence System** transforms raw meeting audio into actionable knowledge by performing:

* Speech-to-text transcription
* Meeting summarization
* Action item extraction
* Topic detection
* Sentiment analysis
* Vector search across past meetings
* Question answering over meeting knowledge

The project implements a **complete AI pipeline**, integrating speech AI, NLP analytics, embeddings, vector search, and a web dashboard.

---

# Features

* 🎤 **Audio Transcription** – Convert meeting recordings to text using Whisper
* 🧠 **LLM Summarization** – Generate concise meeting summaries
* ✅ **Action Item Extraction** – Identify tasks, owners, and deadlines
* 🗣 **Speaker Detection** – Detect participants from transcripts
* 🧩 **Topic Detection** – Identify key discussion themes
* 😊 **Sentiment Analysis** – Analyze the tone of meeting discussions
* 🔎 **Semantic Search** – Query past meetings using vector embeddings
* 🤖 **Meeting Q&A** – Ask questions about previous meetings
* 📊 **Meeting Dashboard** – Explore and filter meeting insights

---

# System Architecture

```
Meeting Audio
      │
      ▼
Speech-to-Text (Whisper)
      │
      ▼
Transcript
      │
      ├── Speaker Detection
      ├── Topic Detection
      └── Sentiment Analysis
      │
      ▼
Meeting Summary + Action Item Extraction
      │
      ▼
Vector Embeddings (sentence-transformers)
      │
      ▼
FAISS Vector Database
      │
      ▼
Retrieval Augmented Generation (RAG)
      │
      ▼
Meeting Q&A System
      │
      ▼
Streamlit Dashboard
```

---

# Tech Stack

| Layer                | Technology            |
| -------------------- | --------------------- |
| Frontend             | Streamlit             |
| Backend              | FastAPI               |
| Speech Recognition   | Whisper               |
| NLP Processing       | Transformers / spaCy  |
| Embeddings           | sentence-transformers |
| Vector Database      | FAISS                 |
| LLM Integration      | OpenAI / Local Models |
| Storage              | JSON                  |
| Programming Language | Python                |

---

# Project Structure

```
.
├── backend/
│   ├── analytics/
│   │   ├── speakers.py
│   │   ├── topics.py
│   │   └── sentiment.py
│   ├── action_items.py
│   ├── meeting_qa.py
│   ├── pipeline.py
│   ├── speech_to_text.py
│   ├── storage.py
│   └── summarizer.py
│
├── data/
│   ├── audio/
│   ├── transcripts/
│   ├── summaries/
│   └── actions/
│
├── rag/
│   ├── embeddings.py
│   └── retriever.py
│
├── ui/
│   ├── app.py
│   └── dashboard.py
│
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
│
├── requirements.txt
└── README.md
```

---

# Quick Start

## 1. Create Virtual Environment

```
python -m venv .venv
.\.venv\Scripts\activate
```

## 2. Install Dependencies

```
pip install -r requirements.txt
```

## 3. Run Backend API

```
uvicorn backend.app.main:app --reload
```

Backend will run at:

```
http://127.0.0.1:8000
```

---

## 4. Run Frontend UI

```
streamlit run ui/app.py
```

Open:

```
http://localhost:8501
```

---

# Using the System

### 1️⃣ Upload Meeting Audio

Upload an audio file (mp3, wav, m4a, etc).

### 2️⃣ Transcribe Audio

Whisper converts the recording into text.

### 3️⃣ Generate Meeting Insights

The system produces:

* Meeting summary
* Key discussion points
* Action items
* Topics
* Sentiment

### 4️⃣ Ask Questions

Example questions:

```
Who is responsible for backend deployment?
What decisions were made in the meeting?
What tasks were assigned to Alex?
```

The system retrieves relevant meeting content using **vector search + RAG**.

---

# Running the Pipeline Directly

You can process a meeting from the backend:

```
python backend/pipeline.py
```

---

# Notes on Models

Some models download automatically on first run:

* Whisper speech model
* sentence-transformers embeddings
* Transformers NLP models

First execution may take **1–2 minutes**.

---

# Optional: Improve NLP Accuracy

Install spaCy model:

```
python -m spacy download en_core_web_sm
```

This improves action item extraction and entity detection.

---

# Example Use Cases

* Automated meeting minutes
* Engineering standup summaries
* Product planning documentation
* Team task tracking
* Knowledge retrieval across meetings

---

# Future Improvements

* Real-time meeting transcription
* Speaker diarization using PyAnnote
* Zoom / Google Meet integration
* Cloud deployment (AWS / GCP)
* Meeting knowledge graph
* Multi-language support

---

# Author

Developed as an **AI engineering project demonstrating speech AI, NLP pipelines, vector search, and LLM applications.**

---

# License

MIT License
