import pandas as pd
from pathlib import Path
import sys

backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

clean_csv = backend_dir / "data" / "cleaned_products.csv"
df = pd.read_csv(clean_csv)

# 1. Search for Ssahebadresses
ssaheba_rows = df[df["product_title"].str.contains("Ssahebadresses", case=False, na=False)]
print(f"Found {len(ssaheba_rows)} Ssahebadresses products:")
for idx, row in ssaheba_rows.iterrows():
    p = row.to_dict()
    print(f"Title: {p['product_title']}")
    print(f"Color: {p.get('color')}")
    print(f"Desc: {p.get('description', '')[:200]}")
    print("-" * 50)

# 2. Search for Rakhdi
rakhdi_rows = df[df["product_title"].str.contains("Rakhdi", case=False, na=False)]
print(f"\nFound {len(rakhdi_rows)} Rakhdi products:")
for idx, row in rakhdi_rows.iterrows():
    p = row.to_dict()
    print(f"Title: {p['product_title']}")
    print(f"Color: {p.get('color')}")
    print(f"Desc: {p.get('description', '')[:200]}")
    print("-" * 50)
