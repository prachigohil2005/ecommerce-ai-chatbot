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

class QueryIntent:
    def __init__(self, category: Optional[str] = None, clothing_types: Optional[List[str]] = None,
                 gender: Optional[str] = None, age_group: Optional[str] = None,
                 occasion: Optional[str] = None, style: Optional[str] = None,
                 season: Optional[str] = None, color: Optional[str] = None,
                 brand: Optional[str] = None):
        self.category = category
        self.clothing_types = clothing_types or []
        self.gender = gender
        self.age_group = age_group
        self.occasion = occasion
        self.style = style
        self.season = season
        self.color = color
        self.brand = brand

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class ProductIntent:
    def __init__(self, category: Optional[str] = None, clothing_type: Optional[str] = None,
                 gender: Optional[str] = None, age_group: Optional[str] = None,
                 occasion: Optional[str] = None, style: Optional[str] = None,
                 season: Optional[str] = None, color: Optional[str] = None,
                 brand: Optional[str] = None):
        self.category = category
        self.clothing_type = clothing_type
        self.gender = gender
        self.age_group = age_group
        self.occasion = occasion
        self.style = style
        self.season = season
        self.color = color
        self.brand = brand

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class SearchResult(list):
    def __init__(self, products: List[Dict[str, Any]], confidence: float, search_stage: int,
                 expanded_queries: List[str], used_scraper: bool, match_type: str, reason: str):
        super().__init__(products)
        self.products = products
        self.confidence = confidence
        self.search_stage = search_stage
        self.expanded_query = expanded_queries  # singular
        self.expanded_queries = expanded_queries  # plural
        self.used_scraper = used_scraper
        self.match_type = match_type
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "products": self.products,
            "confidence": self.confidence,
            "search_stage": self.search_stage,
            "expanded_query": self.expanded_query,
            "expanded_queries": self.expanded_queries,
            "used_scraper": self.used_scraper,
            "match_type": self.match_type,
            "reason": self.reason
        }

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
            
            # Rebuild product intents in metadata if missing
            if "product_intents" not in self.metadata:
                logger.info("Rebuilding product intents in metadata...")
                self.metadata["product_intents"] = {}
                # Apply intent inference
                self.df["product_intent_dict"] = self.df.apply(lambda row: self.infer_product_intent(row.to_dict()).to_dict(), axis=1)
                for idx, row in self.df.iterrows():
                    self.metadata["product_intents"][row["product_id"]] = row["product_intent_dict"]
                # Save updated metadata
                with open(meta_file, "wb") as f:
                    pickle.dump(self.metadata, f)
                logger.info("Product intents cached in metadata successfully.")

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
            "dimension": dimension,
            "product_intents": {}
        }
        
        logger.info("Generating product intents for index...")
        self.df["product_intent_dict"] = self.df.apply(lambda row: self.infer_product_intent(row.to_dict()).to_dict(), axis=1)
        for idx, row in self.df.iterrows():
            self.metadata["product_intents"][row["product_id"]] = row["product_intent_dict"]

        # Write to disk
        faiss.write_index(index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "index_metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
            
        logger.info("FAISS index built and saved successfully.")

        logger.info("FAISS index built and saved successfully.")

    def _setup_intent_vocabularies(self):
        """Initializes the reference vocabularies and precomputes their embeddings for local semantic intent matching."""
        if hasattr(self, "_vocabularies_initialized") and self._vocabularies_initialized:
            return
            
        self.GENDERS = ["men", "women", "unisex"]
        self.AGE_GROUPS = ["kids", "adult"]
        self.OCCASIONS = ["formal", "casual", "party", "sports", "wedding", "festive"]
        self.STYLES = ["office", "western", "streetwear", "ethnic", "athletic"]
        self.CLOTHING_TYPES = ["pant", "dress", "shirt", "suit", "saree"]
        self.CATEGORIES = ["fashion", "shoes", "smartphones", "laptops", "skincare", "fitness", "accessories"]
        self.SEASONS = ["summer", "winter", "monsoon", "rainy"]

        # Token mapping dictionaries for explicit overrides/synonyms
        self.EXPLICIT_SYNONYMS = {
            "pant": "pant", "pants": "pant", "trouser": "pant", "trousers": "pant", "jeans": "pant", "shorts": "pant", "chinos": "pant", "leggings": "pant",
            "dress": "dress", "dresses": "dress", "frock": "dress", "frocks": "dress", "gown": "dress", "gowns": "dress", "skirt": "dress", "skirts": "dress",
            "shirt": "shirt", "shirts": "shirt", "t-shirt": "shirt", "t-shirts": "shirt", "tshirt": "shirt", "tshirts": "shirt", "top": "shirt", "tops": "shirt", "kurta": "shirt", "kurti": "shirt",
            "suit": "suit", "suits": "suit", "blazer": "suit", "blazers": "suit", "coat": "suit", "coats": "suit", "tuxedo": "suit", "tuxedos": "suit",
            "saree": "saree", "sari": "saree", "saris": "saree", "lehenga": "saree", "lehengas": "saree",
            "boy": "kids", "boys": "kids", "girl": "kids", "girls": "kids", "baby": "kids", "baba": "kids", "toddler": "kids", "infant": "kids", "kid": "kids", "kids": "kids",
            "man": "men", "men": "men", "gent": "men", "gents": "men", "male": "men",
            "woman": "women", "women": "women", "lady": "women", "ladies": "women", "female": "women",
            "business": "office", "work": "office", "interview": "office", "formal": "formal"
        }

        # Precompute embeddings
        logger.info("Precomputing intent vocabulary embeddings...")
        self.gender_embs = self.model.encode(self.GENDERS, normalize_embeddings=True)
        self.age_embs = self.model.encode(self.AGE_GROUPS, normalize_embeddings=True)
        self.occasion_embs = self.model.encode(self.OCCASIONS, normalize_embeddings=True)
        self.style_embs = self.model.encode(self.STYLES, normalize_embeddings=True)
        self.clothing_embs = self.model.encode(self.CLOTHING_TYPES, normalize_embeddings=True)
        self.category_embs = self.model.encode(self.CATEGORIES, normalize_embeddings=True)
        self.season_embs = self.model.encode(self.SEASONS, normalize_embeddings=True)
        
        # Color & Brand list
        self.COLORS = ["red", "blue", "black", "green", "white", "yellow", "pink", "grey", "brown", "purple", "orange", "gold", "silver", "mauve", "navy"]
        self.BRANDS = ["apple", "samsung", "nike", "puma", "adidas", "h&m", "roadster", "levi's", "levis", "peter england", "manyavar", "allen solly", "dell", "hp", "lenovo", "asus", "acer", "dressberry", "edenryd", "apsra", "hancock"]

        self._vocabularies_initialized = True

    def map_token_semantically(self, token: str, candidates: List[str], candidate_embeddings: np.ndarray, threshold: float = 0.82) -> Optional[str]:
        """Compares a token embedding with a list of precomputed candidate embeddings, returning the best match if above threshold."""
        try:
            token_emb = self.model.encode([token], normalize_embeddings=True)[0]
            similarities = np.dot(candidate_embeddings, token_emb)
            best_idx = np.argmax(similarities)
            if similarities[best_idx] >= threshold:
                return candidates[best_idx]
        except Exception:
            pass
        return None

    def extract_query_intent(self, query: str) -> QueryIntent:
        """Parses a search query to extract structured intent attributes using token-level semantic match."""
        self._setup_intent_vocabularies()
        
        # Tokenize query
        tokens = [t.lower() for t in re.findall(r'\b[a-zA-Z0-9]+\b', query) if len(t) > 1]
        
        category = None
        clothing_types = []
        gender = None
        age_group = None
        occasion = None
        style = None
        season = None
        color = None
        brand = None

        # 1. Exact/explicit synonym matching
        matched_tokens = {}
        for token in tokens:
            if token in self.EXPLICIT_SYNONYMS:
                val = self.EXPLICIT_SYNONYMS[token]
                if val in self.GENDERS:
                    gender = val
                elif val == "kids":
                    age_group = "kids"
                elif val in self.OCCASIONS:
                    occasion = val
                elif val in self.STYLES:
                    style = val
                elif val in self.CLOTHING_TYPES:
                    if val not in clothing_types:
                        clothing_types.append(val)
                matched_tokens[token] = val

            # Check colors
            if token in self.COLORS:
                color = token
                
        # Check brands
        q_lower = query.lower()
        for b in self.BRANDS:
            if b in q_lower:
                brand = b
                break

        # 2. Semantic matching for unmatched tokens
        for token in tokens:
            if token in matched_tokens or token in self.COLORS or (brand and token in brand):
                continue
                
            # Compare with category
            matched_cat = self.map_token_semantically(token, self.CATEGORIES, self.category_embs, threshold=0.82)
            if matched_cat:
                category = matched_cat
                continue

            # Compare with clothing types
            matched_cl = self.map_token_semantically(token, self.CLOTHING_TYPES, self.clothing_embs, threshold=0.82)
            if matched_cl:
                if matched_cl not in clothing_types:
                    clothing_types.append(matched_cl)
                continue

            # Compare with occasion
            matched_occ = self.map_token_semantically(token, self.OCCASIONS, self.occasion_embs, threshold=0.82)
            if matched_occ:
                occasion = matched_occ
                continue

            # Compare with style
            matched_st = self.map_token_semantically(token, self.STYLES, self.style_embs, threshold=0.82)
            if matched_st:
                style = matched_st
                continue

            # Compare with gender
            matched_gen = self.map_token_semantically(token, self.GENDERS, self.gender_embs, threshold=0.82)
            if matched_gen:
                gender = matched_gen
                continue

            # Compare with age group
            matched_age = self.map_token_semantically(token, self.AGE_GROUPS, self.age_embs, threshold=0.82)
            if matched_age:
                age_group = matched_age
                continue

        # If not explicitly matched, determine default values
        if not category:
            if clothing_types or occasion in ["wedding", "festive"] or style in ["office", "ethnic", "streetwear"] or gender or age_group == "kids":
                category = "fashion"
            else:
                # Semantic check of whole query category
                query_emb = self.model.encode([query], normalize_embeddings=True)[0]
                similarities = np.dot(self.category_embs, query_emb)
                best_idx = np.argmax(similarities)
                if similarities[best_idx] >= 0.50:
                    category = self.CATEGORIES[best_idx]

        # Set default age group if gender or category is specified and age group is missing
        if not age_group and category == "fashion":
            age_group = "adult"

        return QueryIntent(
            category=category,
            clothing_types=clothing_types,
            gender=gender,
            age_group=age_group,
            occasion=occasion,
            style=style,
            season=season,
            color=color,
            brand=brand
        )

    def infer_product_intent(self, product: Dict[str, Any]) -> ProductIntent:
        """Infers the structured ProductIntent of a product from its attributes, title, subcategory, specifications, and description."""
        self._setup_intent_vocabularies()
        
        title = str(product.get("product_title", "")).lower()
        desc = str(product.get("description", "")).lower()
        subcat = str(product.get("subcategory", "")).lower()
        specs = str(product.get("specifications", "")).lower()
        text_context = f"{title} {subcat} {specs} {desc}"

        # 1. Category
        cat_str = str(product.get("category", "")).lower()
        category = "fashion" if cat_str == "fashion" else cat_str
        if category not in self.CATEGORIES:
            category = "accessories"

        # 2. Clothing Type
        clothing_type = None
        if category == "fashion":
            for ct in self.CLOTHING_TYPES:
                keywords = {
                    "pant": [r"\bpant", r"\btrouser", r"\bjean", r"\bshort", r"\bchino", r"\blegging", r"\bbottom"],
                    "dress": [r"\bdress", r"\bfrock", r"\bgown", r"\bskirt", r"\bmaxi"],
                    "shirt": [r"\bshirt", r"\bt-shirt", r"\btshirt", r"\btop", r"\bkurta", r"\bkurti"],
                    "suit": [r"\bsuit", r"\bblazer", r"\bcoat", r"\btuxedo", r"\bsherwani"],
                    "saree": [r"\bsaree", r"\bsari", r"\blehenga", r"\bcholi"]
                }
                if any(re.search(pat, text_context) for pat in keywords[ct]):
                    clothing_type = ct
                    break

        # 3. Gender
        gender = self.get_product_gender(product)

        # 4. Age Group
        kids_pats = [r"\bbaby\b", r"\bbaba\b", r"\bkid\b", r"\bkids\b", r"\btoddler\b", r"\binfant\b", r"\bnewborn\b", r"\bchild\b", r"\bchildren\b", r"\bboy\b", r"\bboys\b", r"\bgirl\b", r"\bgirls\b"]
        if any(re.search(pat, text_context) for pat in kids_pats):
            age_group = "kids"
        else:
            age_group = "adult"

        # 5. Occasion
        occasion = "casual"
        for occ in self.OCCASIONS:
            keywords = {
                "formal": [r"\bformal\b", r"\boffice\b", r"\binterview\b", r"\bbusiness\b"],
                "casual": [r"\bcasual\b", r"\bdaily\b", r"\bhome\b"],
                "party": [r"\bparty\b", r"\bclub\b", r"\bevening\b", r"\bcelebrate\b"],
                "sports": [r"\bsports\b", r"\brunning\b", r"\bgym\b", r"\bworkout\b", r"\bactive\b", r"\btrekking\b", r"\bhiking\b"],
                "wedding": [r"\bwedding\b", r"\bbridal\b", r"\bgroom\b", r"\bmarriage\b"],
                "festive": [r"\bfestive\b", r"\bfestival\b", r"\bethnic\b", r"\btraditional\b"]
            }
            if any(re.search(pat, text_context) for pat in keywords[occ]):
                occasion = occ
                break

        # 6. Style
        style = "western"
        for st in self.STYLES:
            keywords = {
                "office": [r"\boffice\b", r"\bformal\b", r"\bbusiness\b", r"\binterview\b"],
                "western": [r"\bwestern\b", r"\bmodern\b"],
                "streetwear": [r"\bstreet\b", r"\bcasual\b", r"\bhoodie\b"],
                "ethnic": [r"\bethnic\b", r"\btraditional\b", r"\bindian\b"],
                "athletic": [r"\bsports\b", r"\bathletic\b", r"\bactive\b"]
            }
            if any(re.search(pat, text_context) for pat in keywords[st]):
                style = st
                break

        # 7. Season
        season = None
        for seas in self.SEASONS:
            if seas in text_context:
                season = seas
                break

        # 8. Color
        color = None
        for col in self.COLORS:
            if col in title:
                color = col
                break

        # 9. Brand
        brand = "Generic"
        brand_val = str(product.get("brand", "")).lower()
        for b in self.BRANDS:
            if b in brand_val or b in title:
                brand = b
                break

        return ProductIntent(
            category=category,
            clothing_type=clothing_type,
            gender=gender,
            age_group=age_group,
            occasion=occasion,
            style=style,
            season=season,
            color=color,
            brand=brand
        )

    def compute_intent_score(self, query_intent: QueryIntent, product_intent: ProductIntent) -> float:
        """Calculates a weighted compatibility score (0.0 to 1.0) between query intent and product intent."""
        # 1. Gender (Weight: 20%)
        if not query_intent.gender:
            gender_score = 1.0
        else:
            if product_intent.gender == "unisex" or query_intent.gender == "unisex":
                gender_score = 1.0
            elif str(query_intent.gender).lower() == str(product_intent.gender).lower():
                gender_score = 1.0
            else:
                gender_score = 0.0

        # 2. Age group (Weight: 15%)
        if not query_intent.age_group:
            age_score = 1.0
        else:
            if str(query_intent.age_group).lower() == str(product_intent.age_group).lower():
                age_score = 1.0
            else:
                age_score = 0.0

        # 3. Category (Weight: 20%)
        if not query_intent.category:
            category_score = 1.0
        else:
            if str(query_intent.category).lower() == str(product_intent.category).lower():
                category_score = 1.0
            else:
                category_score = 0.0

        # 4. Occasion (Weight: 15%)
        if not query_intent.occasion:
            occasion_score = 1.0
        else:
            if str(query_intent.occasion).lower() == str(product_intent.occasion).lower():
                occasion_score = 1.0
            else:
                occasion_score = 0.0

        # 5. Clothing Type (Weight: 20%)
        if not query_intent.clothing_types:
            clothing_score = 1.0
        else:
            if product_intent.clothing_type in query_intent.clothing_types:
                clothing_score = 1.0
            else:
                clothing_score = 0.0

        # 6. Style (Weight: 10%)
        if not query_intent.style:
            style_score = 1.0
        else:
            if str(query_intent.style).lower() == str(product_intent.style).lower():
                style_score = 1.0
            else:
                style_score = 0.0

        # Compute weighted sum
        total_score = (
            0.20 * gender_score +
            0.15 * age_score +
            0.20 * category_score +
            0.15 * occasion_score +
            0.20 * clothing_score +
            0.10 * style_score
        )
        return total_score

    def are_duplicate_products(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
        """Deduplicates products using ID, exact title cleaning, or brand + substring matching."""
        if p1.get("product_id") == p2.get("product_id"):
            return True
            
        t1 = str(p1.get("product_title", "")).lower()
        t2 = str(p2.get("product_title", "")).lower()
        
        t1_clean = re.sub(r'[^a-z0-9\s]', '', t1).strip()
        t2_clean = re.sub(r'[^a-z0-9\s]', '', t2).strip()
        
        if t1_clean == t2_clean:
            return True
            
        # Check if brand matches and one title is a substring of the other
        b1 = str(p1.get("brand", "")).lower()
        b2 = str(p2.get("brand", "")).lower()
        if b1 == b2 and b1 != "generic":
            if (t1_clean in t2_clean or t2_clean in t1_clean) and len(t1_clean) > 10 and len(t2_clean) > 10:
                return True
                
        return False

    def compute_price_relevance(self, price: float, min_budget: Optional[float], max_budget: Optional[float]) -> float:
        """Calculate price relevance based on min and max budget boundaries.
        
        - If the price is within the range [min_budget, max_budget], relevance is 1.0.
        - If the price exceeds max_budget, an exponential decay is applied.
        - If the price falls below min_budget, an exponential decay is applied (penalizes items that are too cheap).
        """
        # Case 1: No budget limits specified
        if (min_budget is None or min_budget <= 0) and (max_budget is None or max_budget <= 0):
            return 1.0
            
        # Case 2: Only max budget specified
        if min_budget is None or min_budget <= 0:
            if price <= max_budget:
                return 1.0
            excess_ratio = (price - max_budget) / max_budget
            return float(np.exp(-5.0 * excess_ratio))
            
        # Case 3: Only min budget specified
        if max_budget is None or max_budget <= 0:
            if price >= min_budget:
                return 1.0
            deficit_ratio = (min_budget - price) / min_budget
            return float(np.exp(-5.0 * deficit_ratio))
            
        # Case 4: Both min and max budgets specified
        if min_budget <= price <= max_budget:
            return 1.0
            
        if price > max_budget:
            excess_ratio = (price - max_budget) / max_budget
            return float(np.exp(-5.0 * excess_ratio))
            
        # Price is below min_budget
        deficit_ratio = (min_budget - price) / min_budget
        return float(np.exp(-5.0 * deficit_ratio))

    def get_product_gender(self, product: Dict[str, Any]) -> str:
        """Determines the target gender of a product from its title, description or specifications."""
        title = str(product.get("product_title", "")).lower()
        desc = str(product.get("description", "")).lower()
        
        # 1. Check title first (highest precision)
        if "unisex" in title:
            return "unisex"
            
        if re.search(r"\b(women|female|girl|girls|lady|ladies)\b", title):
            return "women"
            
        if re.search(r"\b(men|male|boy|boys|gent|gents)\b", title):
            return "men"
            
        # 2. Check description as fallback (lower precision)
        if "unisex" in desc:
            return "unisex"
            
        if re.search(r"\b(women|female|girl|girls|lady|ladies)\b", desc):
            return "women"
            
        if re.search(r"\b(men|male|boy|boys|gent|gents)\b", desc):
            return "men"
            
        # If no gendered keywords found, default to unisex for non-apparel categories (like Smartphones, Laptops, etc.)
        cat = str(product.get("category", "")).lower()
        if cat in ["smartphones", "laptops", "skincare", "accessories"]:
            return "unisex"
            
        return "unisex"

    def expand_query_semantically(self, query: str) -> List[str]:
        """Uses LLM to dynamically generate semantic search variations and synonyms of the query.
        
        Translates vague, local, or slang terms (e.g. 'baba suit') into standard keywords.
        """
        # Setup gateway
        if not hasattr(self, "gateway") or self.gateway is None:
            try:
                from .llm_gateway import LLMGateway
                self.gateway = LLMGateway()
            except ImportError:
                logger.error("Could not import LLMGateway for query expansion.")
                return [query]
            
        system_instruction = "You are a precise search query expander for an e-commerce catalog."
        prompt = f"""
Analyze the shopping query: "{query}"

Determine the core product type and generate a JSON list containing 2 to 4 semantically expanded search queries or synonyms.
Translate slang, local terms, or highly descriptive requests into generic product keywords.
For example:
- "baba suit" -> ["baby outfit set", "t-shirt shorts combo boy", "baby co-ord set", "kids clothes"]
- "gaming laptop under 1 lakh" -> ["gaming laptop", "RTX laptop", "high performance notebook"]
- "sunscreen for oily skin spf 50" -> ["sunscreen gel spf 50", "matte sunscreen", "oil control sunscreen"]
- "jeans for college boys" -> ["men jeans", "denim pants men", "casual boys jeans"]

Return ONLY a JSON list of strings. Do not include markdown blocks, text explanations, or triple backticks.
"""
        logger.info(f"Triggering semantic query expansion for: '{query}'")
        try:
            res = self.gateway.generate_response(query, [], system_instruction, prompt)
            if res:
                clean_text = res.strip()
                if "```json" in clean_text:
                    clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_text:
                    clean_text = clean_text.split("```")[1].split("```")[0].strip()
                
                expansions = json.loads(clean_text)
                if isinstance(expansions, list):
                    result = [query]
                    for item in expansions:
                        if isinstance(item, str) and item.strip() and item.lower() != query.lower():
                            result.append(item.strip())
                    logger.info(f"Query '{query}' expanded semantically to: {result}")
                    return result
        except Exception as e:
            logger.error(f"Failed to expand query semantically: {str(e)}")
            
        return [query]

    def validate_product_relevance(self, query_intent: QueryIntent, product: Dict[str, Any]) -> Tuple[bool, str]:
        """Calculates structured compatibility between query intent and product intent, replacing keyword mismatch logic."""
        p_id = product.get("product_id")
        p_intent_dict = None
        
        # Load from metadata cache
        if hasattr(self, "metadata") and self.metadata and "product_intents" in self.metadata:
            p_intent_dict = self.metadata["product_intents"].get(p_id)
            
        if not p_intent_dict:
            # Fallback if not found in metadata (e.g. scraped on-the-fly)
            p_intent = self.infer_product_intent(product)
            p_intent_dict = p_intent.to_dict()
            
        p_intent = ProductIntent(**p_intent_dict)
        
        # Calculate sub-scores manually to pinpoint mismatch
        # 1. Gender Mismatch
        if query_intent.gender:
            if not (p_intent.gender == "unisex" or query_intent.gender == "unisex" or query_intent.gender == p_intent.gender):
                return False, "gender mismatch"
                
        # 2. Age Mismatch
        if query_intent.age_group and query_intent.age_group != p_intent.age_group:
            return False, "age mismatch"
            
        # 3. Clothing Type Mismatch
        if query_intent.clothing_types and p_intent.clothing_type not in query_intent.clothing_types:
            return False, "clothing type mismatch"
            
        # 4. Category Mismatch
        if query_intent.category and query_intent.category != p_intent.category:
            return False, "category mismatch"
            
        # 5. Occasion Mismatch
        # If formal is requested, casual items mismatch
        if query_intent.occasion == "formal" and p_intent.occasion == "casual":
            return False, "style mismatch (casual product for formal query)"
            
        return True, ""

    def calculate_confidence(self, products: List[Dict[str, Any]], query_intent: QueryIntent) -> float:
        """Calculates search confidence score based on similarity scores and source match quality."""
        if not products:
            return 0.0
        top_prod = products[0]
        
        # Check relevance validation for confidence
        is_valid, reason = self.validate_product_relevance(query_intent, top_prod)
        if not is_valid:
            # Drop confidence heavily for irrelevant matches to force scraper fallback
            logger.info(f"Relevance validation failed for top product: '{top_prod.get('product_title')}'. Reason: {reason}. Forcing low confidence score.")
            return 0.50
            
        confidence = float(top_prod.get("semantic_score", 0.0))
        
        # Penalize confidence if we are returning alternative suggestions with relaxed filters
        if top_prod.get("is_alternative"):
            relaxed = top_prod.get("relaxed_filters", [])
            confidence -= 0.05 * len(relaxed)
            
        return max(0.0, min(1.0, confidence))

    def rank_candidates(self, candidates: List[Dict[str, Any]], query_intent: QueryIntent, 
                        min_budget: Optional[float] = None, max_budget: Optional[float] = None) -> List[Dict[str, Any]]:
        """Ranks all candidates based on the configured hybrid weighting of semantic similarity, 
        intent compatibility, product rating, popularity, and availability.
        """
        if not candidates:
            return []
            
        unique_candidates = []
        seen = set()
        for c in candidates:
            if c["product_id"] not in seen:
                seen.add(c["product_id"])
                unique_candidates.append(c)
                
        max_reviews = self.df["review_count"].max() if len(self.df) > 0 else 1
        if max_reviews == 0:
            max_reviews = 1

        for p in unique_candidates:
            # 1. Semantic score (0 to 1)
            semantic_score = p.get("semantic_score", 0.70)
            
            # 2. Intent score (0 to 1)
            p_id = p.get("product_id")
            p_intent_dict = None
            if hasattr(self, "metadata") and self.metadata and "product_intents" in self.metadata:
                p_intent_dict = self.metadata["product_intents"].get(p_id)
            if not p_intent_dict:
                p_intent_dict = self.infer_product_intent(p).to_dict()
            p_intent = ProductIntent(**p_intent_dict)
            intent_score = self.compute_intent_score(query_intent, p_intent)
            
            # 3. Rating score (0 to 1)
            rating_val = float(p.get("rating", 4.0))
            rating_score = rating_val / 5.0
            
            # 4. Popularity score (0 to 1)
            reviews_val = int(p.get("review_count", 0))
            popularity_score = float(np.log1p(reviews_val) / np.log1p(max_reviews))
            
            # 5. Availability score (0 to 1)
            avail_str = str(p.get("availability", "In Stock")).lower()
            availability_score = 1.0 if "in stock" in avail_str or avail_str == "" else 0.0
            
            # 6. Price relevance (0 to 1)
            price_val = float(p.get("selling_price", 0.0))
            price_relevance = self.compute_price_relevance(price_val, min_budget, max_budget)
            
            # Calculate final re-ranking score using weights from settings
            final_score = (
                settings.HYBRID_WEIGHT_SEMANTIC * semantic_score +
                settings.HYBRID_WEIGHT_INTENT * intent_score +
                settings.HYBRID_WEIGHT_RATING * rating_score +
                settings.HYBRID_WEIGHT_POPULARITY * popularity_score +
                settings.HYBRID_WEIGHT_AVAILABILITY * availability_score
            )
            
            # Boost/penalize based on metadata
            source_boost = 0.04 if p.get("source") == "scraped" else 0.0
            alt_penalty = -0.04 if p.get("is_alternative") else 0.0
            
            # Enrich candidate dictionary with scores for downstream consumers (like chatbot.py explanation context)
            p["semantic_score"] = semantic_score
            p["rating_score"] = rating_score
            p["popularity_score"] = popularity_score
            p["price_relevance"] = price_relevance
            p["intent_score"] = intent_score
            p["final_ranking_score"] = final_score + source_boost + alt_penalty

        unique_candidates.sort(key=lambda x: x["final_ranking_score"], reverse=True)
        return unique_candidates

    def search(self, query: str, category_filter: Optional[str] = None, budget_filter: Optional[float] = None, 
               min_budget_filter: Optional[float] = None, max_budget_filter: Optional[float] = None, 
               brand_filter: Optional[str] = None, top_k: int = 15, exclude_brands: Optional[List[str]] = None, 
               negated_features: Optional[List[str]] = None, color_filter: Optional[str] = None, 
               size_filter: Optional[str] = None, gender_filter: Optional[str] = None) -> SearchResult:
        """Main search entrypoint using progressive retrieval, query intent mapping, 
        intent compatibility checks, hybrid re-ranking, and online scraper fallback.
        """
        if self.index is None or self.df is None:
            self.initialize()

        if budget_filter is not None and max_budget_filter is None:
            max_budget_filter = budget_filter

        # Normalize gender filter
        if gender_filter and str(gender_filter).lower() not in ["men", "women", "unisex"]:
            gender_filter = None

        # 1. Parse query to structured QueryIntent
        query_intent = self.extract_query_intent(query)
        logger.info(f"Extracted QueryIntent: {query_intent.to_dict()}")

        # Override/update query intent with explicit search filters if provided
        if category_filter:
            query_intent.category = category_filter.lower()
        if gender_filter:
            query_intent.gender = gender_filter.lower()
        if color_filter:
            query_intent.color = color_filter.lower()
        if brand_filter:
            query_intent.brand = brand_filter.lower()

        # Helper to add candidates to unified pool
        candidates = []
        
        def add_to_candidates(items, source="catalog", is_alternative=False, match_reason="", relaxed_filters=None):
            for item in items:
                # Check for duplicates in existing candidate pool
                if any(self.are_duplicate_products(c, item) for c in candidates):
                    continue
                    
                item_copy = item.copy()
                item_copy["source"] = source
                
                # Check structured relevance validation
                is_valid, reason = self.validate_product_relevance(query_intent, item_copy)
                if not is_valid:
                    item_copy["is_alternative"] = True
                    item_copy["match_reason"] = reason
                    # Penalize semantic score so it drops below threshold
                    item_copy["semantic_score"] = max(0.0, item_copy.get("semantic_score", 0.70) - 0.20)
                    r_filters = relaxed_filters or []
                    if "relevance" not in r_filters:
                        r_filters.append("relevance")
                    item_copy["relaxed_filters"] = r_filters
                else:
                    item_copy["is_alternative"] = is_alternative
                    item_copy["match_reason"] = match_reason
                    item_copy["relaxed_filters"] = relaxed_filters or []
                    
                item_copy["availability"] = item.get("availability") or item.get("stock_status", "In Stock")
                # Clean up float NaN in any field
                for k, v in item_copy.items():
                    if isinstance(v, float) and v != v:
                        item_copy[k] = ""
                candidates.append(item_copy)

        # 2. Stage 1: Strict Semantic Search
        logger.info("Executing Stage 1: Strict Semantic Search...")
        stage_stage = 1
        results = self._search_internal(
            query=query, 
            category_filter=category_filter, 
            min_budget_filter=min_budget_filter, 
            max_budget_filter=max_budget_filter, 
            brand_filter=brand_filter, 
            top_k=50, 
            exclude_brands=exclude_brands, 
            negated_features=negated_features, 
            color_filter=color_filter, 
            size_filter=size_filter, 
            gender_filter=gender_filter,
            min_similarity=0.70
        )
        
        if results:
            add_to_candidates(results, source="catalog", is_alternative=False, match_reason="Exact match")

        # Rank candidates to evaluate compatibility
        ranked_pool = self.rank_candidates(candidates, query_intent)
        confidence = self.calculate_confidence(ranked_pool, query_intent)
        
        # Check fallback trigger conditions
        # Trigger fallback if:
        # A. Low overall confidence (< 0.72)
        # B. No products pass intent validation (all top candidates are alternative matches)
        # C. Empty results pool
        has_compatible_match = any(not c.get("is_alternative") for c in ranked_pool)
        
        trigger_fallback = (
            confidence < 0.72 or 
            not ranked_pool or 
            not has_compatible_match or 
            ranked_pool[0].get("is_alternative")
        )

        expanded_queries = [query]
        used_scraper = False

        # 3. Stage 2: Expanded & Relaxed Semantic Search
        if trigger_fallback:
            logger.info("Trigger conditions met. Executing Stage 2: Expanded & Relaxed Search...")
            stage_stage = 2
            
            # Fetch semantic query expansion variations using LLM
            expanded_queries = self.expand_query_semantically(query)
            
            # A. Search using expansions
            for eq in expanded_queries[1:]:
                expanded_results = self._search_internal(
                    query=eq, category_filter=category_filter, min_budget_filter=min_budget_filter,
                    max_budget_filter=max_budget_filter, brand_filter=brand_filter, top_k=30,
                    exclude_brands=exclude_brands, negated_features=negated_features, color_filter=color_filter,
                    size_filter=size_filter, gender_filter=gender_filter, min_similarity=0.68
                )
                if expanded_results:
                    add_to_candidates(
                        expanded_results, source="catalog", is_alternative=True, 
                        match_reason="Expanded semantic match", relaxed_filters=["expanded_query"]
                    )
            
            # B. Relax filters sequentially on original query
            if color_filter:
                relaxed_res = self._search_internal(
                    query=query, category_filter=category_filter, min_budget_filter=min_budget_filter,
                    max_budget_filter=max_budget_filter, brand_filter=brand_filter, top_k=30,
                    exclude_brands=exclude_brands, negated_features=negated_features, color_filter=None,
                    size_filter=size_filter, gender_filter=gender_filter
                )
                add_to_candidates(relaxed_res, source="catalog", is_alternative=True, match_reason="Relaxed color filter", relaxed_filters=["color"])

            if brand_filter:
                relaxed_res = self._search_internal(
                    query=query, category_filter=category_filter, min_budget_filter=min_budget_filter,
                    max_budget_filter=max_budget_filter, brand_filter=None, top_k=30,
                    exclude_brands=exclude_brands, negated_features=negated_features, color_filter=None,
                    size_filter=size_filter, gender_filter=gender_filter
                )
                add_to_candidates(relaxed_res, source="catalog", is_alternative=True, match_reason="Relaxed brand filter", relaxed_filters=["brand"])

            if gender_filter:
                relaxed_res = self._search_internal(
                    query=query, category_filter=category_filter, min_budget_filter=min_budget_filter,
                    max_budget_filter=max_budget_filter, brand_filter=None, top_k=30,
                    exclude_brands=exclude_brands, negated_features=negated_features, color_filter=None,
                    size_filter=size_filter, gender_filter=None
                )
                add_to_candidates(relaxed_res, source="catalog", is_alternative=True, match_reason="Relaxed gender filter", relaxed_filters=["gender"])

            # Relax similarity threshold to 0.65
            relaxed_res = self._search_internal(
                query=query, category_filter=category_filter, min_budget_filter=min_budget_filter,
                max_budget_filter=max_budget_filter, brand_filter=None, top_k=30,
                exclude_brands=exclude_brands, negated_features=negated_features, color_filter=None,
                size_filter=size_filter, gender_filter=None, min_similarity=0.65
            )
            add_to_candidates(relaxed_res, source="catalog", is_alternative=True, match_reason="Relaxed similarity threshold", relaxed_filters=["similarity"])

            # Re-rank after Stage 2 relaxation
            ranked_pool = self.rank_candidates(candidates, query_intent, min_budget_filter, max_budget_filter)
            confidence = self.calculate_confidence(ranked_pool, query_intent)
            
            has_compatible_match = any(not c.get("is_alternative") for c in ranked_pool)
            trigger_scraper = (
                confidence < 0.72 or 
                not ranked_pool or 
                not has_compatible_match or 
                ranked_pool[0].get("is_alternative")
            )
            
            # 4. Stage 3: Online Scraper Fallback
            if trigger_scraper:
                logger.info("Executing Stage 3: Triggering Online Scraper Fallback...")
                stage_stage = 3
                used_scraper = True
                try:
                    from .scraper import FlipkartScraper
                    scraper = FlipkartScraper()
                    
                    budget_suffix = ""
                    if max_budget_filter and max_budget_filter > 0:
                        target_price = int((min_budget_filter + max_budget_filter) / 2) if min_budget_filter else int(max_budget_filter * 0.8)
                        budget_suffix = f" {target_price}"
                        
                    scrape_q = query
                    clean_q = re.sub(r"^(suggest|recommend|show|find|search for)\s+me\s+some\s+", "", scrape_q, flags=re.IGNORECASE)
                    clean_q = re.sub(r"^(suggest|recommend|show|find|search for)\s+(me|some)\s+", "", clean_q, flags=re.IGNORECASE)
                    scrape_q = clean_q or query
                    
                    scraped_items = scraper.scrape_query(query=(scrape_q + budget_suffix).strip(), category=category_filter or "Fashion", limit=4)
                    
                    if scraped_items:
                        # Append scraped items to DF, compute ProductIntent and update index
                        self.add_products_to_index(scraped_items)
                        
                        # Fetch and score products directly from the index to maintain correct embeddings/scores
                        scraped_ids = [item["product_id"] for item in scraped_items]
                        scraped_results = []
                        
                        for p_id in scraped_ids:
                            row = self.df[self.df["product_id"] == p_id]
                            if not row.empty:
                                prod_dict = row.iloc[0].to_dict()
                                
                                # Deserialize specifications JSON
                                try:
                                    specs_json = prod_dict.get("specifications", "{}")
                                    specs = json.loads(specs_json)
                                    if isinstance(specs, dict):
                                        prod_dict.update(specs)
                                except Exception:
                                    pass
                                    
                                for k, v in prod_dict.items():
                                    if pd.isna(v):
                                        prod_dict[k] = ""
                                        
                                # Compute semantic similarity score manually
                                query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")
                                prod_text = str(prod_dict.get("embedding_text", ""))
                                prod_embedding = self.model.encode([prod_text], normalize_embeddings=True).astype("float32")
                                score = float(np.dot(query_embedding[0], prod_embedding[0]))
                                
                                semantic_score = float((score + 1.0) / 2.0)
                                prod_dict["semantic_score"] = semantic_score
                                scraped_results.append(prod_dict)
                                
                        if scraped_results:
                            add_to_candidates(
                                scraped_results, source="scraped", is_alternative=False, match_reason="Scraped online match"
                            )
                except Exception as e:
                    logger.error(f"Error during Flipkart online scraper execution: {str(e)}")

        # 5. Final Ranking and Metadata Enrichment
        final_ranked = self.rank_candidates(candidates, query_intent, min_budget_filter, max_budget_filter)
        final_confidence = self.calculate_confidence(final_ranked, query_intent)
        
        # Attach confidence and match reason metadata to top products
        for p in final_ranked:
            p["search_confidence"] = final_confidence
            if p.get("is_alternative"):
                # Determine specific mismatch reason
                is_valid, reason = self.validate_product_relevance(query_intent, p)
                p["match_reason"] = reason if not is_valid else "Similar match alternative"
            else:
                p["match_reason"] = p.get("match_reason") or "Exact match"

        # 6. Determine Search Result Status / Match Type
        if not final_ranked:
            match_type = "NO_MATCH"
            reason = "No candidates found across any retrieval stages."
        elif used_scraper and any(p.get("source") == "scraped" and not p.get("is_alternative") for p in final_ranked):
            match_type = "SCRAPED_MATCH"
            reason = "Exact local matches not found; successfully retrieved online search results."
        elif final_ranked[0].get("is_alternative"):
            match_type = "SIMILAR_MATCH"
            reason = f"Exact matches not found. Suggesting alternatives: {final_ranked[0].get('match_reason')}"
        else:
            match_type = "EXACT_MATCH"
            reason = "Found compatible matching products in our database."

        logger.info(f"Final search returned {len(final_ranked)} products. Match Type: {match_type}, Confidence: {final_confidence:.4f}")
        
        # Return SearchResult object (which behaves as a list of top top_k products but exposes search metadata)
        top_products = final_ranked[:top_k]
        return SearchResult(
            products=top_products,
            confidence=final_confidence,
            search_stage=stage_stage,
            expanded_queries=expanded_queries,
            used_scraper=used_scraper,
            match_type=match_type,
            reason=reason
        )

    def _search_internal(self, query: str, category_filter: Optional[str], min_budget_filter: Optional[float], 
                         max_budget_filter: Optional[float], brand_filter: Optional[str], top_k: int, 
                         exclude_brands: Optional[List[str]], negated_features: Optional[List[str]], 
                         color_filter: Optional[str], size_filter: Optional[str], gender_filter: Optional[str], 
                         min_similarity: float = 0.70) -> List[Dict[str, Any]]:
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
            price_relevance = self.compute_price_relevance(price_val, min_budget_filter, max_budget_filter)
            
            # Apply semantic similarity threshold to filter out completely irrelevant products
            if semantic_score < min_similarity:
                continue

            # Weighted Score calculation prioritizing semantic match to avoid popularity bias
            final_score = (
                0.70 * semantic_score +
                0.10 * rating_score +
                0.10 * popularity_score +
                0.10 * price_relevance
            )
            
            # Append scores for debugging and chatbot UI explanation
            product["semantic_score"] = semantic_score
            product["rating_score"] = rating_score
            product["popularity_score"] = popularity_score
            product["price_relevance"] = price_relevance
            product["final_ranking_score"] = final_score
            
            hits.append(product)

        # Sort hits by final ranking score in descending order, prioritizing dynamically scraped items at the top
        hits.sort(key=lambda x: (1 if "dynamic" in str(x.get("subcategory", "")).lower() else 0, x["final_ranking_score"]), reverse=True)
        
        return hits[:top_k]

    def add_products_to_index(self, new_products: List[Dict[str, Any]]):
        """Dynamically appends new products to the dataframe, computes embeddings, adds to FAISS index, and saves to disk."""
        if not new_products:
            return
            
        logger.info(f"Adding {len(new_products)} new products to the FAISS index...")
        
        try:
            # 1. Clean/Prepare new products as a DataFrame
            from .data_pipeline import DataPipeline
            pipeline = DataPipeline()
            new_df = pd.DataFrame(new_products)
            new_df = pipeline.clean_dataset(new_df)
            
            # 2. Append to current dataframe
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            
            # 3. Compute embeddings for new products
            texts = new_df["embedding_text"].astype(str).tolist()
            embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embeddings = np.array(embeddings).astype("float32")
            
            # 4. Add to FAISS index
            self.index.add(embeddings)
            
            # 5. Update metadata
            new_product_ids = new_df["product_id"].tolist()
            self.metadata["product_ids"].extend(new_product_ids)
            
            if "product_intents" not in self.metadata:
                self.metadata["product_intents"] = {}
            for idx, row in new_df.iterrows():
                p_id = row["product_id"]
                p_intent = self.infer_product_intent(row.to_dict())
                self.metadata["product_intents"][p_id] = p_intent.to_dict()
            
            # 6. Save updated data and index to disk
            self.df.to_csv(self.clean_path, index=False)
            faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
            with open(self.index_dir / "index_metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
                
            # Also append to raw products CSV so they sync
            if os.path.exists(settings.DATA_RAW_PATH):
                try:
                    raw_df = pd.read_csv(settings.DATA_RAW_PATH)
                    raw_df = pd.concat([raw_df, pd.DataFrame(new_products)], ignore_index=True)
                    raw_df.to_csv(settings.DATA_RAW_PATH, index=False)
                    logger.info("Successfully appended new products to raw products CSV.")
                except Exception as e:
                    logger.error(f"Error appending to raw products CSV: {str(e)}")
                    
            logger.info(f"Successfully added and indexed {len(new_products)} new products.")
        except Exception as e:
            logger.error(f"Error dynamically adding products to index: {str(e)}")

if __name__ == "__main__":
    engine = SearchEngine()
    engine.initialize()
    results = engine.search("best gaming laptop under 100000", budget_filter=100000.0, top_k=3)
    for res in results:
        print(f"Title: {res['product_title']}")
        print(f"Price: Rs. {res['selling_price']} | Rating: {res['rating']} | Score: {res['final_ranking_score']:.4f}")
        print("---")
