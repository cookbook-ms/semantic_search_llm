# 🍔 Semantic Food Search Demo

This project is a Streamlit app that performs semantic search over a food dataset using FAISS indexing, SVD compression, and optional GPT-based reranking.

## 🚀 Features
- Search food items using vector embeddings
- Choose between large/small embedding models
- Apply SVD to approximate similarity
- GPT-based reranking and optional NDCG evaluation

Example: 

[Streamlit app](https://cookbook-ms-semantic-search-llm-main-bpeqha.streamlit.app/)

## 📦 Setup

1. Clone this repository and navigate into the folder:
    
```bash
git clone https://github.com/cookbook-ms/semantic_search_llm.git
cd semantic_search_llm
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

3.	Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure api for embedding and gpt responses in 'constants.py':

5. [Optional] Precompute the embeddings for the items and queries. See 'embedding_items.ipynb' and 'embedding_queries.ipynb' for details.
 
6. Run the Streamlit app:
```bash
streamlit run main.py
```