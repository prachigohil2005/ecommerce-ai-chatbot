import os
import re
import pickle
import logging
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.index_dir = settings.FAISS_INDEX_DIR
        self.clean_path = settings.DATA_CLEAN_PATH
        self.model = None
        self.index = None
        self.metadata = None
        self.df = None

    def initialize(self, force_rebuild: bool = False):
        """Initializes embedding model, loads dataset, and loads/builds the FAISS index."""
        logger.info("Initializing search engine...")
        
        # Load embedding model
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            
        # Load dataset
        if not os.path.exists(self.clean_path):
            raise FileNotFoundError(f"Cleaned dataset not found at {self.clean_path}. Run data_pipeline.py first.")
        self.df = pd.read_csv(self.clean_path)
        logger.info(f"Loaded {len(self.df)} Flipkart products for search index.")

        index_file = self.index_dir / "index.faiss"
        meta_file = self.index_dir / "index_metadata.pkl"

        if force_rebuild or not index_file.exists() or not meta_file.exists():
            self.build_index()
        else:
            logger.info(f"Loading existing FAISS index from {index_file}...")
            self.index = faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info("FAISS index loaded successfully.")

    def build_index(self):
        """Generates embeddings and builds a FAISS FlatL2/IP index."""
        logger.info("Building FAISS index from scratch...")
        os.makedirs(self.index_dir, exist_ok=True)

        # Generate embeddings
        texts = self.df["embedding_text"].astype(str).tolist()
        logger.info(f"Encoding {len(texts)} product texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")

        # Create FAISS Index (using Inner Product since we normalized embeddings for cosine similarity)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        self.index = index
        
        # Save index mapping (index position -> product ID)
        self.metadata = {
            "product_ids": self.df["product_id"].tolist(),
            "dimension": dimension
        }

        # Write to disk
        faiss.write_index(index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "index_metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
            
        logger.info("FAISS index built and saved successfully.")

    def compute_price_relevance(self, price: float, budget: Optional[float]) -> float:
        """Calculate price relevance.
        
        If budget is specified:
        - Price <= budget: relevance is 1.0
        - Price > budget: exponential decay score based on how much it exceeds budget
        If no budget is specified, defaults to 1.0.
        """
        if budget is None or budget <= 0:
            return 1.0
            
        if price <= budget:
            return 1.0
            
        # Exponential decay for products exceeding budget
        excess_ratio = (price - budget) / budget
        return float(np.exp(-5.0 * excess_ratio))

    def get_product_gender(self, product: Dict[str, Any]) -> str:
        """Determines the target gender of a product from its title, description or specifications."""
        title = str(product.get("product_title", "")).lower()
        desc = str(product.get("description", "")).lower()
        
        # Check for unisex first
        if "unisex" in title or "unisex" in desc:
            return "unisex"
            
        # Check for women's keywords with word boundaries
        if re.search(r"\b(women|female|girl|girls|lady|ladies)\b", title) or re.search(r"\b(women|female|girl|girls|lady|ladies)\b", desc):
            return "women"
            
        # Check for men's keywords with word boundaries
        if re.search(r"\b(men|male|boy|boys|gent|gents)\b", title) or re.search(r"\b(men|male|boy|boys|gent|gents)\b", desc):
            return "men"
            
        # If no gendered keywords found, default to unisex for non-apparel categories (like Smartphones, Laptops, etc.)
        cat = str(product.get("category", "")).lower()
        if cat in ["smartphones", "laptops", "skincare", "accessories"]:
            return "unisex"
            
        return "unisex"

    def search(self, query: str, category_filter: Optional[str] = None, budget_filter: Optional[float] = None, brand_filter: Optional[str] = None, top_k: int = 15, exclude_brands: Optional[List[str]] = None, negated_features: Optional[List[str]] = None, color_filter: Optional[str] = None, size_filter: Optional[str] = None, gender_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search products using FAISS semantic search and apply multi-factor ranking with automatic color/brand/gender relaxations."""
        if self.index is None or self.df is None:
            self.initialize()

        # Identify target nouns in the query for strict item-type filtering
        target_nouns = []
        for noun in ["saree", "sari", "sadi", "shirt", "jeans", "t-shirt", "tshirt", "dress", "jacket", "suit", "lehenga", "kurta", "laptop", "phone", "smartphone", "dumbbell", "yoga mat", "water bottle", "headphone", "earbud", "watch", "smartwatch", "backpack", "sunscreen", "rose water", "face wash", "facewash", "serum", "moisturizer", "cleanser", "toner"]:
            if noun in query.lower():
                if noun in ["sari", "sadi"]:
                    target_nouns.append("saree")
                elif noun in ["tshirt"]:
                    target_nouns.append("t-shirt")
                elif noun in ["phone"]:
                    target_nouns.append("smartphone")
                else:
                    target_nouns.append(noun)

        relaxed_filters = []

        # 1. Attempt strict search with requested color, brand and gender constraints
        results = self._search_internal(query, category_filter, budget_filter, brand_filter, top_k, exclude_brands, negated_features, color_filter, size_filter, gender_filter, target_nouns)

        # 2. If no matches found, retry relaxing color filter
        if not results and color_filter:
            logger.info("No matches found matching the requested color. Retrying with relaxed color filter...")
            relaxed_filters.append("color")
            results = self._search_internal(query, category_filter, budget_filter, brand_filter, top_k, exclude_brands, negated_features, None, size_filter, gender_filter, target_nouns)

        # 3. If still no matches found, retry relaxing brand constraint
        if not results and brand_filter:
            logger.info("No matches found matching the requested brand. Retrying with relaxed brand constraint...")
            relaxed_filters.append("brand")
            results = self._search_internal(query, category_filter, budget_filter, None, top_k, exclude_brands, negated_features, None, size_filter, gender_filter, target_nouns)

        # 4. If still no matches found, retry relaxing gender constraint
        if not results and gender_filter:
            logger.info("No matches found matching the requested gender. Retrying with relaxed gender constraint...")
            relaxed_filters.append("gender")
            results = self._search_internal(query, category_filter, budget_filter, None, top_k, exclude_brands, negated_features, None, size_filter, None, target_nouns)

        # Tag relaxed results
        if relaxed_filters:
            for p in results:
                p["is_alternative"] = True
                p["relaxed_filters"] = relaxed_filters
        else:
            for p in results:
                p["is_alternative"] = False
                p["relaxed_filters"] = []

        return results

    def _search_internal(self, query: str, category_filter: Optional[str], budget_filter: Optional[float], brand_filter: Optional[str], top_k: int, exclude_brands: Optional[List[str]], negated_features: Optional[List[str]], color_filter: Optional[str], size_filter: Optional[str], gender_filter: Optional[str], target_nouns: List[str]) -> List[Dict[str, Any]]:
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")
        
        # Query FAISS index (retrieve a larger pool first to filter and rank)
        retrieve_k = min(150, len(self.df))
        distances, indices = self.index.search(query_embedding, retrieve_k)
        
        # Extract matches
        hits = []
        max_reviews = self.df["review_count"].max() if len(self.df) > 0 else 1
        if max_reviews == 0:
            max_reviews = 1

        for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
                
            product_id = self.metadata["product_ids"][idx]
            # Retrieve product row
            product_row = self.df[self.df["product_id"] == product_id]
            if product_row.empty:
                continue
                
            product = product_row.iloc[0].to_dict()
            
            # Post-filtering options
            # 1. Category Filter (case-insensitive substring match)
            if category_filter and product.get("category", "").lower() != category_filter.lower():
                continue
                
            # 2. Brand Filter
            if brand_filter and brand_filter.lower() not in product.get("brand", "").lower():
                continue

            # 3. Excluded Brands Filter
            if exclude_brands:
                prod_brand = str(product.get("brand", "")).lower()
                if any(eb.lower() in prod_brand or prod_brand in eb.lower() for eb in exclude_brands):
                    continue

            # 4. Negated Features Filter
            if negated_features:
                prod_title = str(product.get("product_title", "")).lower()
                prod_desc = str(product.get("description", "")).lower()
                prod_specs = str(product.get("specifications", "")).lower()
                combined_text = f"{prod_title} {prod_desc} {prod_specs}"
                if any(nf.lower() in combined_text for nf in negated_features):
                    continue

            # Convert NaN fields to empty string
            for key, val in product.items():
                if pd.isna(val):
                    product[key] = ""

            # DE-SERIALIZE SPECIFICATIONS JSON
            try:
                specs_json = product.get("specifications", "{}")
                specs = json.loads(specs_json)
                if isinstance(specs, dict):
                    product.update(specs)
            except Exception as e:
                logger.warning(f"Error parsing specifications JSON for product {product_id}: {str(e)}")

            # 5. Strict product noun filtering (prevents showing T-shirts for a Saree query)
            if target_nouns:
                prod_title_lower = str(product.get("product_title", "")).lower()
                prod_cat_lower = str(product.get("category", "")).lower()
                prod_sub_lower = str(product.get("subcategory", "")).lower()
                
                noun_matched = False
                for noun in target_nouns:
                    if noun in prod_title_lower or noun in prod_cat_lower or noun in prod_sub_lower:
                        noun_matched = True
                        break
                    if noun == "smartphone" and ("mobile" in prod_title_lower or "phone" in prod_title_lower):
                        noun_matched = True
                        break
                    if noun == "headphone" and ("earbud" in prod_title_lower or "tws" in prod_title_lower or "headset" in prod_title_lower):
                        noun_matched = True
                        break
                if not noun_matched:
                    continue

            # 6. Color Filter
            if color_filter:
                prod_color = str(product.get("color", "")).lower()
                prod_title_lower = str(product.get("product_title", "")).lower()
                if color_filter.lower() not in prod_color and color_filter.lower() not in prod_title_lower:
                    continue

            # 7. Size Filter
            if size_filter:
                prod_size = str(product.get("size", "")).lower()
                if size_filter.lower() != prod_size:
                    continue

            # 8. Gender Filter
            if gender_filter:
                prod_gender = self.get_product_gender(product)
                if gender_filter == "women" and prod_gender not in ["women", "unisex"]:
                    continue
                elif gender_filter == "men" and prod_gender not in ["men", "unisex"]:
                    continue
                elif gender_filter == "unisex" and prod_gender != "unisex":
                    continue

            # Calculate Scores for Ranking Engine
            
            # A. Semantic Score: similarity from FAISS is dot product (cosine similarity) which is in [-1, 1], normalized to [0, 1]
            semantic_score = float((score + 1.0) / 2.0)
            
            # B. Rating Score: rating (0 to 5) divided by 5
            rating_val = float(product.get("rating", 4.0))
            rating_score = rating_val / 5.0
            
            # C. Popularity Score: log normalized reviews
            reviews_val = int(product.get("review_count", 0))
            popularity_score = float(np.log1p(reviews_val) / np.log1p(max_reviews))
            
            # D. Price Relevance Score
            price_val = float(product.get("selling_price", 0.0))
            price_relevance = self.compute_price_relevance(price_val, budget_filter)
            
            # Weighted Score calculation
            final_score = (
                0.45 * semantic_score +
                0.20 * rating_score +
                0.20 * popularity_score +
                0.15 * price_relevance
            )
            
            # Append scores for debugging and chatbot UI explanation
            product["semantic_score"] = semantic_score
            product["rating_score"] = rating_score
            product["popularity_score"] = popularity_score
            product["price_relevance"] = price_relevance
            product["final_ranking_score"] = final_score
            
            hits.append(product)

        # Sort hits by final ranking score in descending order
        hits.sort(key=lambda x: x["final_ranking_score"], reverse=True)
        
        return hits[:top_k]

if __name__ == "__main__":
    engine = SearchEngine()
    engine.initialize()
    results = engine.search("best gaming laptop under 100000", budget_filter=100000.0, top_k=3)
    for res in results:
        print(f"Title: {res['product_title']}")
        print(f"Price: Rs. {res['selling_price']} | Rating: {res['rating']} | Score: {res['final_ranking_score']:.4f}")
        print("---")
