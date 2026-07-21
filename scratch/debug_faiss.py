import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load directories
backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
clean_csv = backend_dir / "data" / "cleaned_products.csv"
index_path = backend_dir / "faiss_index" / "index.faiss"
meta_path = backend_dir / "faiss_index" / "index_metadata.pkl"

df = pd.read_csv(clean_csv)
print(f"Cleaned products DataFrame length: {len(df)}")

# Find any product with "Paithani" or "Banarasi" or "Saree" in the title
saree_rows = df[df["product_title"].str.contains("Banarasi|Paithani|Saree|Divastri|Wastera", case=False, na=False)]
print(f"\nFound {len(saree_rows)} matching rows in CSV:")
for idx, row in saree_rows.iterrows():
    print(f"- ID: {row['product_id']} | Title: {row['product_title']} | Cat: {row['category']} | Subcat: {row['subcategory']} | Color: {row.get('color')} | Price: {row['selling_price']}")

# Load index and metadata
index = faiss.read_index(str(index_path))
with open(meta_path, "rb") as f:
    metadata = pickle.load(f)
print(f"\nFAISS index size: {index.ntotal}")
print(f"Metadata product count: {len(metadata['product_ids'])}")

# Check last 5 entries in metadata vs dataframe
print("\nLast 5 metadata product IDs:")
print(metadata['product_ids'][-5:])
print("Last 5 DataFrame product IDs:")
print(df["product_id"].tolist()[-5:])

# Let's perform a raw FAISS search on the query
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
query_emb = model.encode(["red south indian saree"], normalize_embeddings=True).astype("float32")
distances, indices = index.search(query_emb, 30)

print("\nRaw FAISS search results (top 30):")
for d, idx in zip(distances[0], indices[0]):
    if idx == -1:
        continue
    pid = metadata["product_ids"][idx]
    matching_row = df[df["product_id"] == pid]
    if not matching_row.empty:
        title = matching_row.iloc[0]["product_title"]
        print(f"- Dist: {d:.4f} | ID: {pid} | Title: {title}")
    else:
        print(f"- Dist: {d:.4f} | ID: {pid} | [NOT FOUND IN CSV]")
