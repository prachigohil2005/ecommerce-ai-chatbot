import pandas as pd
from pathlib import Path

backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
clean_csv = backend_dir / "data" / "cleaned_products.csv"
df = pd.read_csv(clean_csv)

print("Total products:", len(df))
print("\nUnique categories:")
print(df["category"].value_counts())

fashion_df = df[df["category"].str.lower() == "fashion"]
print("\nTotal Fashion products:", len(fashion_df))

import sys
backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

from app.search_engine import SearchEngine
se = SearchEngine()

genders = fashion_df.apply(lambda row: se.get_product_gender(row.to_dict()), axis=1)
print("\nGender distribution in Fashion:")
print(genders.value_counts())

# Show a few sample women's fashion products
print("\nSample Women's Fashion products:")
women_fashion = fashion_df[genders == "women"]
for idx, row in women_fashion.head(10).iterrows():
    print(f"  - Title: {row['product_title']} | Brand: {row['brand']} | Subcat: {row['subcategory']} | Price: {row['selling_price']}")
