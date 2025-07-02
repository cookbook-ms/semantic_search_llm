import pandas as pd 
import json 

items_file = "data/5k_items_curated.csv"

def build_image_url(image_str):
    return f"https://static.ifood-static.com.br/image/upload/t_low/pratos/{image_str}"

df = pd.read_csv(items_file)


cleaned_items = []
for _, row in df.iterrows():
    item_id = row["itemId"]
    
    # parse json strings 
    try: 
        metadata = json.loads(row["itemMetadata"])
    except Exception: 
        continue 

    name = metadata.get("name", "")
    category_name = metadata.get("category_name", "")
    description = metadata.get("description", "")
    taxonomy = metadata.get("taxonomy", {})
    taxonomy_levels = [
        taxonomy.get("l0",""),
        taxonomy.get("l1",""),
        taxonomy.get("l2","")
    ]
    images = [build_image_url(image) for image in metadata.get("images", [])]
    
    ''' build search key text'''
    search_key = " ".join(
        filter(None, [name, category_name, description] + taxonomy_levels)
    )
    
    cleaned_items.append({
        "item_id": item_id,
        "name": name,
        "category_name": category_name,
        "description": description,
        "taxonomy_levels": taxonomy_levels,
        "search_key": search_key,
        "images": images        
    })
    
cleaned_df = pd.DataFrame(cleaned_items)
cleaned_df.to_csv("data/5k_items_cleaned.csv", index=False)