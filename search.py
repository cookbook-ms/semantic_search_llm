import re
import numpy as np
import requests
from embedding import embed
from constants import GPT_URL, API_KEY
from llm_pair_compare import llm_pair_compare


def search_and_rerank(query_id, rerank, topk, user_query, embedding_model_choice, gpt_model_choice, index, cleaned_df, query_embeddings, svd=False, reconstructed_similarity=None, compute_ndcg=False):
    """Perform search and rerank as in the notebook."""
    if user_query is not None:
        query_embedding = embed(user_query, embedding_model_choice)
    else:
        query_embedding = query_embeddings[query_id-1, :].tolist()

    num_candidates = 20
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), num_candidates)
    indices = indices[0]
    if svd:
        indices = np.argsort(reconstructed_similarity, axis=1)[:, -num_candidates:][:, ::-1][query_id-1]

    candidates = []
    for idx in indices:
        item = cleaned_df.iloc[idx]
        candidates.append({
            "candidate_id": item["item_id"], 
            "name": item["name"],
            "category_name": item["category_name"],
            "description": item["description"], 
            "search_key": item["search_key"],
            "images": item["images"],
        })
    rank_idx = range(1, topk+1)
    if compute_ndcg:
        win_count, total_comparisons = llm_pair_compare(
            GPT_URL=GPT_URL,
            API_KEY=API_KEY,
            user_query=user_query,
            candidates=candidates,
            rank_idx=rank_idx,
            gpt_model_choice=gpt_model_choice
        )     
        for idx in rank_idx:
            i = idx - 1 
            if total_comparisons[i] > 0:
                candidates[i]["score"] = win_count[i] 
            else:
                candidates[i]["score"] = 0.0
            
    if not rerank:
        return candidates, rank_idx
    else:
        candidates_text = ""
        for i, candidate in enumerate(candidates, 1):
            candidates_text += (
                f"{i}. [name: {candidate['name']}, category: {candidate['category_name']}, description: {candidate['description']}]"
            )
        # You are an expert in product search and recommendation. You are given now a pool of candidates, obtained from a vector search algorithm. Your task is to rank these candidates based on their relevance to the query, and select the top {topk}. This relevance is based on the query and the candidate's attributes such as name, category, and description.
        # Você é um especialista em busca e recomendação de produtos. Você recebe agora uma coleção de candidatos, obtida por um algoritmo de busca vetorial. Sua tarefa é classificá‑los com base em sua relevância para a consulta e selecionar os {topk} mais relevantes. Essa relevância deve levar em consideração a consulta e os atributos do candidato, como nome, categoria e descrição.
        prompt = f"""
        You are an expert in semantic search for food products. You are given a food query. Your task is to select the top {topk} in the candidates based on their relevance to the query.

        query:
        {user_query}
        
        Candidates:
        {candidates_text}
        
        Output in the following format: 
        1) candidate_pool_id, name
        2) candidate_pool_id, name
        """
        response = requests.post(
            GPT_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            json={"model": gpt_model_choice, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        )
        results = response.json()["choices"][0]["message"]["content"]

        rerank_idx = []
        for line in results.split("\n"):
            match = re.search(r"\d+\) (\d+)", line)
            if match:
                rerank_idx.append(int(match.group(1)))
        
        win_count, total_comparisons = llm_pair_compare(
                GPT_URL=GPT_URL,
                API_KEY=API_KEY,
                user_query=user_query,
                candidates=candidates,
                rank_idx=rerank_idx,
                gpt_model_choice=gpt_model_choice
            )        

        for idx in rerank_idx:
            i = idx - 1 
            if total_comparisons[i] > 0:
                candidates[i]["score"] = win_count[i] 
            else:
                candidates[i]["score"] = 0.0

    return candidates, rerank_idx

def compute_similarity_matrix(query_embeddings, embedding_array):
    """Compute the similarity matrix using SVD."""
    similarity_matrix = np.dot(query_embeddings, embedding_array.T)
    u, s, vt = np.linalg.svd(similarity_matrix, full_matrices=False)
    return u, s, vt