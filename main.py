import streamlit as st
import numpy as np
import pandas as pd
import faiss
import json
from search import search_and_rerank, compute_similarity_matrix
from utils import compute_ndcg

st.set_page_config(layout="wide")
_, middle_col, _ = st.columns([1, 3, 1])
middle_col.title("🍔 Semantic Food Search Demo")
_, left_col, right_col, _, _ = st.columns([1, 1, 1, 1, 1])


with left_col:
    embedding_choice = st.selectbox("Embedding Model:", ["large", "small"], key="embedding")
    gpt_model_choice = st.selectbox("GPT Model:", ["gpt-4.1", "gpt-4.1-nano", "gpt-4.1-mini"], key="gpt")
    rerank = st.checkbox("Rerank candidates using GPT?")
    compute_ndcg_flag = st.checkbox("Compute NDCG?", value=False)

with right_col:
    query_id = st.number_input("Enter your query number:", min_value=1, max_value=100, value=10)
    topk = st.number_input("Number of candidates:", min_value=1, max_value=20, value=5)
    svd_flag = st.checkbox("Use SVD for similarity?", value=False)

# --------------------------------------------------
# User Inputs
@st.cache_data
def load_stuff(embedding_choice):
    user_queries = pd.read_csv("data/queries.csv")["search_term_pt"]
    index = faiss.read_index(f"data/index_{embedding_choice}.index")         
    embedding_array = np.load(f"data/item_embeddings_{embedding_choice}.npy") 
    cleaned_df = pd.read_csv("data/5k_items_cleaned.csv")                               
    query_embeddings = np.load(f"data/query_embeddings_{embedding_choice}.npy")
    return user_queries, index, embedding_array, cleaned_df, query_embeddings
    
    
user_queries, index, embedding_array, cleaned_df, query_embeddings = load_stuff(embedding_choice)    
user_query = user_queries.iloc[query_id-1]

if svd_flag:
    u, s, vt = compute_similarity_matrix(query_embeddings, embedding_array)
    with right_col:
        top_s = st.number_input("Number of singular values:", min_value=1, max_value=100, value=100, width=300)
    reconstructed_similarity = np.dot(u[:, :top_s], np.dot(np.diag(s[:top_s]), vt[:top_s, :]))
else:
    reconstructed_similarity = None

# reconstructed_similarity = None
 
# --------------------------------------------------
# Perform Search
if left_col.button("Search") and user_query:
    candidates, rerank_indices = search_and_rerank(
        query_id,
        rerank=rerank,
        topk=topk,
        user_query=user_query,
        embedding_model_choice=embedding_choice,
        gpt_model_choice=gpt_model_choice,
        index=index,
        cleaned_df=cleaned_df,
        query_embeddings=query_embeddings,
        svd=svd_flag,
        reconstructed_similarity=reconstructed_similarity,
        compute_ndcg=compute_ndcg_flag
    )
    st.markdown(f"### 🐍 Search Query:\n**{user_query}**\n### 🔍 Results")
    # print(candidates)
    if compute_ndcg_flag:
        ndcg = compute_ndcg(rerank_indices, [candidates[rank-1]['score'] for i, rank in enumerate(rerank_indices, 1)])
        left_col.markdown(f"<h4 style='color:green;'>NDCG@{topk}: {ndcg:.4f}</h4>", unsafe_allow_html=True)

    for row_start in range(0, len(rerank_indices), 5):
        row_cols = st.columns(5)
        for i, col in enumerate(row_cols):
            if row_start + i >= len(rerank_indices):
                break
            idx, rank = row_start + i + 1, rerank_indices[row_start + i]
            item = candidates[rank - 1]
            with col:
                st.markdown(f"**{idx}. {item['name']}**")
                st.markdown(f"*{item['category_name']}*")
                if compute_ndcg_flag:
                    st.markdown(f"Score: `{item['score']}`")
                if rerank:
                    st.markdown(f"<small>Original rank: {rank}</small>", unsafe_allow_html=True)
                for url in json.loads(item['images'].replace("'", '"')):
                    st.image(url, width=150)