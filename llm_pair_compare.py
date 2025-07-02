import requests
from collections import defaultdict

def build_pairwise_prompt(query, item_a, item_b):
    return f"""
            Given a query: "{query}", which of the following two items is more relevant to the query?

            Item A: name: {item_a['name']}, category: {item_a['category_name']}, description: {item_a['description']}
            Item B: name: {item_b['name']}, category: {item_b['category_name']}, description: {item_b['description']}

            Output "Item A" or "Item B" only:
            """
 
def llm_pair_compare(GPT_URL, API_KEY, user_query, candidates, rank_idx, gpt_model_choice):
    win_count = defaultdict(int)
    total_comparisons = defaultdict(int)
    for i in range(len(rank_idx)):
        for j in range(i+1, len(rank_idx)):
            a_idx, b_idx = rank_idx[i]-1, rank_idx[j]-1
            item_a, item_b = candidates[a_idx], candidates[b_idx]
            prompt = build_pairwise_prompt(user_query, item_a, item_b)
            pair_response = requests.post(
                GPT_URL, 
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": gpt_model_choice,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                }       
            )
            answer = pair_response.json()["choices"][0]["message"]["content"].strip().lower()
            total_comparisons[a_idx] += 1
            total_comparisons[b_idx] += 1

            prompt_1 = build_pairwise_prompt(user_query, item_b, item_a)
            pair_response = requests.post(
                GPT_URL, 
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": gpt_model_choice,
                    "messages": [{"role": "user", "content": prompt_1}],
                    "temperature": 0,
                }       
            )
            answer_1 = pair_response.json()["choices"][0]["message"]["content"].strip().lower()
            total_comparisons[a_idx] += 1
            total_comparisons[b_idx] += 1
            if "item a" in answer and "item b" not in answer and "item a" in answer_1 and "item b" not in answer_1:
                win_count[a_idx] += 1
            elif "item b" in answer and "item a" not in answer and "item b" in answer_1 and "item a" not in answer_1:
                win_count[b_idx] += 1
            elif "item a" in answer and "item b" not in answer and "item b" in answer_1 and "item a" not in answer_1:
                win_count[a_idx] += 0.5
            elif "item b" in answer and "item a" not in answer and "item a" in answer_1 and "item b" not in answer_1:
                win_count[b_idx] += 0.5

    return win_count, total_comparisons