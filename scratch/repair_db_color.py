import pandas as pd
import json
import re
from pathlib import Path
import sys

backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

from app.search_engine import SearchEngine

def repair():
    clean_csv = backend_dir / "data" / "cleaned_products.csv"
    raw_csv = backend_dir / "data" / "raw_products.csv"
    
    colors = ["red", "blue", "black", "green", "white", "yellow", "pink", "grey", "brown", "purple", "orange", "gold", "silver", "navy", "maroon", "beige", "khaki"]
    
    # 1. Update cleaned_products.csv
    print("Repairing cleaned_products.csv...")
    df = pd.read_csv(clean_csv)
    updated_count = 0
    
    for idx, row in df.iterrows():
        title = str(row["product_title"])
        title_lower = title.lower()
        
        # Check if title explicitly contains any color keyword
        found_color = None
        for c in colors:
            # Match with word boundaries
            if re.search(rf"\b{c}\b", title_lower):
                found_color = c.capitalize()
                break
                
        if found_color:
            current_color = str(row.get("color", ""))
            if current_color != found_color:
                # Update color
                df.at[idx, "color"] = found_color
                
                # Update specifications JSON
                specs_json = row.get("specifications", "{}")
                try:
                    specs = json.loads(specs_json) if pd.notna(specs_json) and specs_json else {}
                except Exception:
                    specs = {}
                specs["color"] = found_color
                df.at[idx, "specifications"] = json.dumps(specs)
                
                # Update description if it lists features
                desc = str(row.get("description", ""))
                # If desc has "color: OldColor", replace it
                if "color: " in desc:
                    desc = re.sub(r"color: [a-zA-Z]+", f"color: {found_color}", desc)
                    df.at[idx, "description"] = desc
                
                # Regenerate embedding text
                # We can construct it same as in pipeline
                brand = row.get("brand", "Generic")
                cat = row.get("category", "Fashion")
                subcat = row.get("subcategory", "")
                price = row.get("selling_price", 0.0)
                rating = row.get("rating", 4.0)
                
                specs_str = ", ".join([f"{k}: {v}" for k, v in specs.items()])
                embedding_text = f"Product: {title}. Brand: {brand}. Category: {cat} ({subcat}). Price: Rs. {price:.0f}. Rating: {rating}/5. Specifications: {specs_str}. Description: {desc}."
                df.at[idx, "embedding_text"] = embedding_text
                
                updated_count += 1
                print(f"  Updated '{title}': color set to '{found_color}'")
                
    if updated_count > 0:
        df.to_csv(clean_csv, index=False)
        print(f"Successfully updated {updated_count} products in cleaned_products.csv.")
    else:
        print("No products needed color repair in cleaned_products.csv.")
        
    # 2. Update raw_products.csv as well
    if raw_csv.exists():
        print("\nRepairing raw_products.csv...")
        raw_df = pd.read_csv(raw_csv)
        raw_updated_count = 0
        for idx, row in raw_df.iterrows():
            title = str(row["product_title"])
            title_lower = title.lower()
            found_color = None
            for c in colors:
                if re.search(rf"\b{c}\b", title_lower):
                    found_color = c.capitalize()
                    break
            if found_color:
                specs_json = row.get("specifications", "{}")
                try:
                    specs = json.loads(specs_json) if pd.notna(specs_json) and specs_json else {}
                except Exception:
                    specs = {}
                if specs.get("color") != found_color:
                    specs["color"] = found_color
                    raw_df.at[idx, "specifications"] = json.dumps(specs)
                    raw_updated_count += 1
        if raw_updated_count > 0:
            raw_df.to_csv(raw_csv, index=False)
            print(f"Successfully updated {raw_updated_count} products in raw_products.csv.")
            
    # 3. Rebuild search index
    if updated_count > 0:
        print("\nRebuilding FAISS index...")
        se = SearchEngine()
        se.initialize(force_rebuild=True)
        print("FAISS index successfully rebuilt!")

if __name__ == "__main__":
    repair()
