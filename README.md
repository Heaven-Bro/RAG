# 🚀 Advanced RAG Toolkit: From Ingestion to CRAG

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)
![RAG Architecture](https://img.shields.io/badge/Architecture-Advanced--RAG-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive, modular implementation of **Retrieval-Augmented Generation (RAG)** systems using Python and Jupyter Notebooks. This repository serves as a practical guide and boilerplate covering everything from document ingestion and chunking techniques to advanced paradigms like Hybrid Search, Re-ranking, Multi-Modal RAG, and Corrective RAG (CRAG).

---

## 🌟 Key Features & Architectures

* **📦 Advanced Chunking Strategies:**
  * **Recursive Character Chunking** (Structural boundaries)
  * **Semantic Chunking** (Meaning-based boundary detection)
  * **Agentic Chunking** (LLM-driven dynamic segmentation)
* **🔍 Sophisticated Retrieval Techniques:**
  * **Multi-Query Retrieval:** Query expansion for improved recall
  * **Reciprocal Rank Fusion (RRF):** Combining rankings from multiple search streams
  * **Hybrid Search:** Dense vector search combined with sparse keyword search
  * **Cross-Encoder Re-ranking:** Post-retrieval relevance scoring for high accuracy
* **🧠 Advanced RAG Pipelines:**
  * **History-Aware Generation:** Conversational memory integration
  * **Multi-Modal RAG:** Joint processing of text and visual data
  * **Corrective RAG (CRAG):** Self-correcting retrieval validation workflow
* **💾 Vector Storage:**
  * Local persistent storage powered by **ChromaDB** (`/db/chroma_db`)

---

## 📂 Repository Structure

```text
.
├── db/
│   └── chroma_db/                         # Persistent ChromaDB vector database
├── docs/                                  # Knowledge base documents
│   ├── attention-is-all-you-need.pdf      # Research papers & PDFs
│   ├── book1.pdf, book2.pdf, book3.pdf
│   └── Google.txt, Microsoft.txt, ...     # Tech company documentation
│
├── 1_ingestion_pipeline.py                # Document loading & vector DB ingestion
├── 2_retrieval_pipeline.py                # Basic similarity retrieval
├── 3_answer_generation.py                 # Standard RAG generation pipeline
├── 4_history_aware_generation.py          # Conversational memory RAG
├── 5_recursive_character_text_spliiter.py # Character-level recursive chunking
├── 6_semantic_chunking.py                 # Semantic similarity chunking
├── 7_agentic_chunking.py                  # Agent-driven document chunking
├── 8_multi_modal_rag.ipynb                # Text + Image Multi-Modal RAG
├── 9_retrieval_methods.py                 # Benchmarking basic retrieval methods
├── 10_multi_query_retrieval.py            # Multi-query generation & retrieval
├── 11_reciprocal_rank_fusion.py           # Reciprocal Rank Fusion (RRF) algorithm
├── 12_hybrid_search.ipynb                 # Dense + Sparse hybrid retrieval
├── 13_reranker.ipynb                      # Cross-encoder re-ranking pipeline
├── 14_CRAG.ipynb                          # Corrective RAG workflow
│
├── check_models.py                        # Utility script to test model connectivity
├── synthetic_questions.txt                # Benchmark/test questions
├── requirements.txt                       # Python dependency list
└── .env                                   # API keys configuration file
```

---

## 🛠️ Quick Start & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. Create & Activate Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Environment Variables Setup
Create a `.env` file in the root directory and specify your API credentials:

```env
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
# Add any additional provider keys below
```

---

## 🏃 Execution Guide

1. **Verify Environment Setup:**
   ```bash
   python check_models.py
   ```

2. **Ingest Documents:**
   Process documents inside `/docs` and persist embeddings into ChromaDB:
   ```bash
   python 1_ingestion_pipeline.py
   ```

3. **Run Query Pipelines:**
   ```bash
   # Test basic retrieval
   python 2_retrieval_pipeline.py

   # Test complete Q&A generation
   python 3_answer_generation.py

   # Test conversational RAG with history
   python 4_history_aware_generation.py
   ```

4. **Explore Interactive Notebooks:**
   Launch VS Code or Jupyter Lab to run advanced notebooks:
   * `8_multi_modal_rag.ipynb` — Multi-modal retrieval
   * `12_hybrid_search.ipynb` — BM25 + Vector hybrid retrieval
   * `13_reranker.ipynb` — Reranking implementation
   * `14_CRAG.ipynb` — Corrective RAG framework

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.