# 🍽️ Connoisseur Companion — IBM AI Capstone Project

> **AI-powered restaurant recommendation system** for California restaurants, featuring multi-agent workflows, multimodal RAG, and an MCP-based chat interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Modules](#modules)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Recipe Images (Download)](#recipe-images-download)
- [Usage](#usage)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Data Files](#data-files)

---

## Overview

This capstone project builds an end-to-end AI restaurant recommendation pipeline:

1. **Extract** structured data from 100+ restaurant descriptions using LLM (Gemini)
2. **Index** restaurant articles and recipe images into a vector database (ChromaDB)
3. **Retrieve** relevant restaurants via multimodal similarity search (text + image)
4. **Recommend** restaurants through a multi-agent system with specialized AI agents
5. **Serve** recommendations via an MCP server + Gradio chat UI with a ReAct agent loop

The project was originally designed for IBM Watson but has been **adapted to use Google Gemini API** (`google-genai`).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONNOISSEUR COMPANION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Module 1: Data Pipeline                                            │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │ California        │───▶│ Gemini API       │                      │
│  │ Culinary Map     │    │ (Structured      │                      │
│  │ (.txt, 100+      │    │  Output)         │                      │
│  │  restaurants)    │    └────────┬─────────┘                      │
│  └──────────────────┘             │                                 │
│                                   ▼                                 │
│                    ┌──────────────────────────┐                    │
│                    │ structured_restaurant    │                    │
│                    │ _data.json (100+ recs)   │                    │
│                    └──────────┬───────────────┘                    │
│                               │                                     │
│  Module 2-3: Multimodal RAG  │                                     │
│  ┌────────────────┐          │    ┌────────────────┐               │
│  │ SentenceTransf. │◀─────────┘    │ CLIP           │               │
│  │ (text embeds)  │               │ (image embeds) │               │
│  └───────┬────────┘               └───────┬────────┘               │
│          │                                │                         │
│          └─────────┐    ┌─────────────────┘                        │
│                    ▼    ▼                                           │
│              ┌──────────────┐                                      │
│              │   ChromaDB   │                                      │
│              │ (vector DB)  │                                      │
│              └──────┬───────┘                                      │
│                     │                                               │
│  Module 3: Multi-Agent System                                       │
│  ┌──────────────────┤                                              │
│  │ 6 Specialized    │                                              │
│  │ AI Agents:       │                                              │
│  │ • User Profiler  │                                              │
│  │ • RAG Retriever  │                                              │
│  │ • Trend Analyst  │                                              │
│  │ • Style Expert   │                                              │
│  │ • Nutrition Exp. │                                              │
│  │ • Recommendation │                                              │
│  └──────────────────┤                                              │
│                     │                                               │
│  Module 4: MCP + Chat UI                                            │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │ MCP Server       │◀──▶│ Gradio Chat UI   │                      │
│  │ (FastMCP)       │    │ (ReAct Agent     │                      │
│  │ 3 tools +       │    │  Loop)           │                      │
│  │ 1 resource      │    └──────────────────┘                      │
│  └──────────────────┘                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Modules

### Module 1 — Data Extraction & Management (`src/data_pipeline/`)

| File | Description |
|------|-------------|
| `extract_restaurants.py` | Batch processor: reads `California-Culinary-Map.txt`, sends restaurant paragraphs to Gemini in batches of 5, validates via Pydantic, saves with checkpointing |
| `manage_restaurants.py` | CRUD CLI app for managing restaurant records. Supports add (via LLM parsing), edit, delete, and browse operations with auto JSON repair |

### Module 2 — Multimodal Vector Indexing & Retrieval (`src/retrieval/`)

| File | Description |
|------|-------------|
| `build_vector_index.py` | Builds ChromaDB vector indices for restaurant articles (SentenceTransformer) and recipe images (CLIP) |
| `similarity_search.py` | Similarity search functions: text-to-text, image-to-image, with optional metadata filtering |
| `multimodal_fusion.py` | Weighted multimodal fusion ranking — combines text and image similarity scores with tunable weights |

### Module 3 — Multi-Agent System (`src/agents/`)

| File | Description |
|------|-------------|
| `agent_definitions.py` | Defines 6 specialized AI agents (profiler, retriever, trend analyst, food style expert, nutrition expert, recommendation expert) |
| `multi_agent_workflow.py` | Orchestrates agents in a 4-phase workflow (profile → retrieve → analyze in parallel → synthesize) using `ThreadPoolExecutor` |
| `chatbot_ui.py` | Gradio-based chatbot with intent classification, preference extraction, and tabbed UI (chat + add restaurant + add recipe) |

### Module 4 — MCP Server & LLM Host (`src/mcp_chat/`)

| File | Description |
|------|-------------|
| `server.py` | FastMCP server exposing 3 tools (`get_restaurant_info`, `recommend_by_vibe`, `get_review`) and 1 resource (`culinary-map://california`) |
| `client.py` | MCP client with tool discovery, verification, sampling callback, and demo functions |
| `app.py` | Gradio chat UI with LangChain + Gemini + MCP integration, implementing a ReAct agent loop (max 5 tool-call iterations) |

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **LLM** | Google Gemini API (`google-genai`, `gemini-3.6-flash`) |
| **LLM Framework** | LangChain (`langchain-google-genai`, `langchain-core`) |
| **MCP** | FastMCP, MCP SDK |
| **Vector DB** | ChromaDB (`langchain-chroma`) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`), OpenAI CLIP (`clip-vit-base-patch32`) |
| **Data Validation** | Pydantic v2 |
| **UI** | Gradio |
| **ML** | PyTorch, Transformers (Hugging Face) |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### Steps

1. **Clone / download** the project

2. **Create virtual environment**
   ```bash
   cd capstone-project
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key** — create a `.env` file in the project root (`capstone-project/`):
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

5. **Download recipe images** — folder `data/recipe_images/` tidak ikut di repo, lihat [Recipe Images (Download)](#recipe-images-download). Langkah ini hanya wajib untuk Module 2 (vector index & retrieval).

---

## Recipe Images (Download)

Gambar resep (PNG) tidak disertakan di repo karena ukurannya besar. Download dataset-nya dari link berikut:

```
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip
```

Lalu extract isinya ke `data/recipe_images/`:

```bash
# macOS/Linux
curl -L -o synthetic-recipe-images.zip "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"
mkdir -p data/recipe_images
unzip synthetic-recipe-images.zip -d data/recipe_images

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip" -OutFile synthetic-recipe-images.zip
Expand-Archive synthetic-recipe-images.zip -DestinationPath data/recipe_images
```

> `build_vector_index.py` mencari semua file `*.png` di dalam `data/recipe_images/` (termasuk subfolder), jadi struktur isi zip-nya tidak masalah. Folder ini sudah di-ignore di `.gitignore`.

---

## Usage

> Jalankan semua perintah di bawah dari **root project** (`capstone-project/`).

### 1. Extract restaurant data (Module 1)
```bash
python src/data_pipeline/extract_restaurants.py
```
Processes `data/raw/California-Culinary-Map.txt` → `data/processed/structured_restaurant_data.json`

Untuk mengelola data (add / edit / delete) lewat CLI:
```bash
python src/data_pipeline/manage_restaurants.py
```

### 2. Build vector indices (Module 2)
```bash
python src/retrieval/build_vector_index.py
```
Creates ChromaDB indices for text and image retrieval (butuh `data/recipe_images/`, lihat bagian download di atas).

> Index disimpan di `~/chroma_multimodal` (folder home user), bukan di dalam project. Jalankan ulang script ini setelah memindahkan gambar, karena path gambar ikut tersimpan di metadata index.

### 3. Run similarity search demos (Module 2)
```bash
python src/retrieval/similarity_search.py
python src/retrieval/multimodal_fusion.py
```

### 4. Run multi-agent workflow (Module 3)
```bash
python src/agents/agent_definitions.py
python src/agents/multi_agent_workflow.py
```

Chatbot Gradio (Module 3):
```bash
python src/agents/chatbot_ui.py
```
> Di akhir file ini `demo.launch()` masih di-comment. Uncomment dulu kalau mau UI-nya benar-benar terbuka di browser.

### 5. Launch MCP chat UI (Module 4)
```bash
python src/mcp_chat/app.py
```
Opens a Gradio chat interface at `http://localhost:7860` with MCP tool calling.

### 6. Test MCP server standalone
```bash
# Terminal 1: start server
python src/mcp_chat/server.py

# Terminal 2: run client
python src/mcp_chat/client.py
```

Atau pakai smoke test singkat (otomatis menjalankan server sendiri lewat stdio):
```bash
python tests/smoke_test_server.py
```

---

## Testing

Unit tests cover data integrity, MCP tools, CRUD operations, Pydantic schemas, fusion math, and chatbot helpers. **All tests are fully offline** — no API calls.

```bash
# Run all tests
python -m pytest tests/ -v

# Or with unittest
python -m unittest tests.test_project -v
```

### Test Coverage

| Test Group | # Tests | What It Covers |
|-----------|---------|----------------|
| `TestDataIntegrity` | 8 | JSON validity, required fields, ratings range, duplicates |
| `TestMCPServerTools` | 8 | `get_restaurant_info`, `recommend_by_vibe`, `get_review` |
| `TestDataManagement` | 6 | File I/O, backup creation, add/delete roundtrip |
| `TestPydanticSchema` | 4 | Model validation, defaults, JSON roundtrip |
| `TestMultimodalFusionLogic` | 6 | min-max normalization, similarity conversion, ranking |
| `TestChatbotHelpers` | 3 | Recommendation formatting |
| `TestAgentConfigs` | 3 | Agent config structure validation |

---

## Project Structure

```
capstone-project/
│
├── .gitignore                          # Git ignore rules
├── .env                                # API keys (not tracked)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── data/                               # Semua data (non-kode)
│   ├── raw/
│   │   └── California-Culinary-Map.txt         # Source: 100+ restaurant descriptions
│   ├── processed/
│   │   ├── structured_restaurant_data.json     # Extracted structured data
│   │   ├── augmented_user_review.json          # Augmented review data
│   │   └── augmented_food_recipe.json          # Food recipe data
│   ├── source/
│   │   ├── Recipes.json                        # Recipe metadata
│   │   └── Synthetic-User-Reviews.json         # Synthetic user reviews
│   ├── backups/
│   │   └── structured_restaurant_data.json.bak # Auto-backup dari manage_restaurants.py
│   ├── archive/
│   │   └── augmented_user_review_alt.json      # Versi lama review (key `restaurant_name`)
│   └── recipe_images/                  # (tidak ikut repo) download, lihat bagian Recipe Images
│
├── src/                                # Semua kode Python
│   ├── data_pipeline/                  # Module 1
│   │   ├── extract_restaurants.py          # Batch data extraction (Gemini)
│   │   └── manage_restaurants.py           # CRUD management app
│   ├── retrieval/                      # Module 2
│   │   ├── build_vector_index.py           # Vector index builder
│   │   ├── similarity_search.py            # Similarity search
│   │   └── multimodal_fusion.py            # Multimodal fusion ranking
│   ├── agents/                         # Module 3
│   │   ├── agent_definitions.py            # Agent definitions
│   │   ├── multi_agent_workflow.py         # Multi-agent workflow
│   │   └── chatbot_ui.py                   # Chatbot UI (Gradio)
│   └── mcp_chat/                       # Module 4
│       ├── server.py                       # MCP server (FastMCP)
│       ├── client.py                       # MCP client + verification
│       └── app.py                          # Gradio chat UI (ReAct)
│
└── tests/
    ├── test_project.py                 # Unit test suite (offline)
    └── smoke_test_server.py            # Smoke test MCP server
```

---

## Data Files

| File | Records | Description |
|------|---------|-------------|
| `data/raw/California-Culinary-Map.txt` | 100+ | Raw text descriptions of California restaurants |
| `data/processed/structured_restaurant_data.json` | 80 | LLM-extracted structured records (name, location, type, food_style, rating, price_range, signatures, vibe, environment) |
| `data/processed/augmented_user_review.json` | 10 | Reviews with reviewer info, ratings, text, and image descriptions |
| `data/processed/augmented_food_recipe.json` | 109 | Recipe data yang di-augment, dipakai untuk vector index |
| `data/source/Synthetic-User-Reviews.json` | 10 | Synthetic user reviews with user IDs and image URLs |
| `data/source/Recipes.json` | 109 | Recipe metadata |
| `data/recipe_images/` | 109 | PNG images of recipes for multimodal retrieval — [download di sini](#recipe-images-download) (tidak ada di repo) |

---

## Credits

- **Course**: IBM AI Engineering Professional Certificate — Capstone Project
- **Adapted by**: Ahmad Lesmana
- **LLM Provider**: Google Gemini API (adapted from original IBM Watson design)

