import re
import json
import random
import logging
import requests
from typing import List, Dict, Any, Tuple, Optional
import google.generativeai as genai
from .config import settings
from .search_engine import SearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ECommerceChatbot:
    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.groq_api_key = settings.GROQ_API_KEY
        self.deepseek_api_key = settings.DEEPSEEK_API_KEY
        self.mistral_api_key = settings.MISTRAL_API_KEY
        self.qwen_api_key = settings.QWEN_API_KEY
        self.llama_api_key = settings.LLAMA_API_KEY

        # Define persona system instruction
        self.system_instruction = """
You are a warm, knowledgeable, and helpful human sales associate in a premium Indian e-commerce store (representing Flipkart listings). 
Your goal is to guide the user to the best products based on their needs, just like a high-end store assistant.

YOUR PERSONA & INSTRUCTIONS:
1. Speak in a friendly, helpful, conversational, and natural tone.
2. If the user's intent is a greeting, welcome them warmly, introduce yourself, and ask how you can help them shop today. Mention the categories we support: Smartphones, Laptops, Fashion (Dresses, Shirts, Jeans), Shoes, Fitness, Skincare, and Accessories. Do NOT recommend specific products yet.
3. If the user's request is vague or lacks details (Is Vague Request is True), politely recommend 2-3 of the popular products provided in the context, but simultaneously ask 1-2 friendly clarifying questions to help them narrow down their preferences (e.g. their budget range, brand preference, or specific use-case).
4. Explain *why* you are recommending the top products. Connect their specs to the customer's lifestyle, needs, or use-case.
5. Summarize the key strengths (pros) and weaknesses (cons) of the top 2-3 recommendations.
6. If the customer's budget is exceeded, mention alternatives or explain why a slightly higher priced item is worth the premium.
7. Avoid database technical jargon like "FAISS", "retrieved documents", "score", "context", or "embeddings".
"""
        
        # Configure Gemini API
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction
            )
            logger.info("Gemini API configured successfully with System Instructions.")
        else:
            self.model = None
            logger.warning("GEMINI_API_KEY not found in configuration. Chatbot will run in OFFLINE/MOCK fallback mode.")

    def parse_budget(self, text: str) -> Optional[float]:
        """Rule-based regex backup for budget extraction (e.g. 'under 40000', 'under 50k')."""
        text = text.lower()
        
        # Match pattern "under 40000", "under rs 40000", "below 40k", etc.
        # Handle 'k' multiplier (e.g. 40k -> 40000)
        k_matches = re.findall(r"(?:under|below|under rs\.?|budget of|within|less than)\s*(\d+)\s*k\b", text)
        if k_matches:
            return float(k_matches[0]) * 1000.0
            
        numeric_matches = re.findall(r"(?:under|below|under rs\.?|budget of|within|less than|max|maximum)\s*(?:rs\.?|inr|₹)?\s*(\d+)", text)
        if numeric_matches:
            return float(numeric_matches[0])
            
        # Match simple "under ₹40,000" or similar
        comma_matches = re.findall(r"(?:under|below|within|less than)\s*(?:rs\.?|inr|₹)?\s*(\d{1,3}(?:,\d{3})+)", text)
        if comma_matches:
            return float(comma_matches[0].replace(",", ""))
            
        return None

    def extract_category_rule_based(self, text: str) -> Optional[str]:
        """Rule-based backup for category extraction."""
        text = text.lower()
        mapping = {
            "smartphone": "Smartphones",
            "phone": "Smartphones",
            "mobile": "Smartphones",
            "iphone": "Smartphones",
            "galaxy": "Smartphones",
            "laptop": "Laptops",
            "notebook": "Laptops",
            "macbook": "Laptops",
            "computer": "Laptops",
            "shirt": "Fashion",
            "t-shirt": "Fashion",
            "tshirt": "Fashion",
            "jeans": "Fashion",
            "dress": "Fashion",
            "frock": "Fashion",
            "kurti": "Fashion",
            "top": "Fashion",
            "trousers": "Fashion",
            "skirt": "Fashion",
            "gown": "Fashion",
            "maxi": "Fashion",
            "cardigan": "Fashion",
            "sweater": "Fashion",
            "hoodie": "Fashion",
            "sari": "Fashion",
            "saree": "Fashion",
            "sadi": "Fashion",
            "lehenga": "Fashion",
            "kurta": "Fashion",
            "jacket": "Fashion",
            "suit": "Fashion",
            "tuxedo": "Fashion",
            "sherwani": "Fashion",
            "clothing": "Fashion",
            "clothes": "Fashion",
            "fashion": "Fashion",
            "shoe": "Shoes",
            "sneaker": "Shoes",
            "boot": "Shoes",
            "sandal": "Shoes",
            "slipper": "Shoes",
            "footwear": "Shoes",
            "heels": "Shoes",
            "flats": "Shoes",
            "sunscreen": "Skincare",
            "sun screen": "Skincare",
            "spf": "Skincare",
            "serum": "Skincare",
            "moisturizer": "Skincare",
            "cream": "Skincare",
            "cleanser": "Skincare",
            "facewash": "Skincare",
            "face wash": "Skincare",
            "toner": "Skincare",
            "lotion": "Skincare",
            "gel": "Skincare",
            "skincare": "Skincare",
            "dumbbell": "Fitness",
            "yoga": "Fitness",
            "gym": "Fitness",
            "fitness": "Fitness",
            "water bottle": "Fitness",
            "shaker": "Fitness",
            "protein": "Fitness",
            "creatine": "Fitness",
            "headphone": "Accessories",
            "earbud": "Accessories",
            "watch": "Accessories",
            "smartwatch": "Accessories",
            "backpack": "Accessories",
            "bag": "Accessories",
            "charger": "Accessories",
            "cable": "Accessories",
            "goggles": "Accessories",
            "gogles": "Accessories",
            "sunglasses": "Accessories",
            "glasses": "Accessories",
            "trip": "Fashion",
            "travel": "Fashion",
            "vacation": "Fashion",
            "rajasthan": "Fashion",
            "kashmir": "Fashion",
            "masoorie": "Fashion",
            "mussoorie": "Fashion",
            "goa": "Fashion",
            "garba": "Fashion",
            "navratri": "Fashion",
            "diwali": "Fashion",
            "wedding": "Fashion",
            "interview": "Fashion"
        }
        for keyword, category in mapping.items():
            if keyword in text:
                return category
        return None

    def _extract_entities_via_groq(self, query: str, history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Fallback NLU extraction using Groq's Llama 3 API."""
        if not self.groq_api_key:
            return None
        
        # Format history context
        history_context = ""
        for turn in history[-4:]:
            role = "Customer" if turn["role"] == "user" else "Sales Associate"
            history_context += f"{role}: {turn['content']}\n"
            
        system_instruction = "You are a precise JSON extractor. Analyze the customer query and chat history to extract structured shopping metadata."
        
        prompt = f"""
CHAT HISTORY:
{history_context}

NEW CUSTOMER QUERY:
"{query}"

Return ONLY a JSON object with the following fields (no explanations, markdown blocks, or other text):
{{
  "intent": "string (one of: 'greeting', 'recommendation', 'comparison', 'filtering', 'general query', 'clarification', 'out_of_scope')",
  "is_out_of_scope": "boolean (true if the query is asking for products, services, or topics completely unrelated to our store categories [Smartphones, Laptops, Fashion, Shoes, Skincare, Fitness], false otherwise)",
  "comparison_targets": "array of strings (if comparison intent, list the specific products/brands mentioned, e.g. ['iPhone 15 Pro', 'Galaxy S24'])",
  "place": "string or null (if the query mentions a place, city, region, or country, e.g. 'Rajasthan', 'Goa', 'Kashmir', 'London')",
  "weather_context": "string or null (the typical weather/climate of that place, e.g. 'hot and dry', 'cold and snowy', 'tropical and humid')",
  "cultural_style": "string or null (traditional or cultural style associated with that place, e.g. 'bandhani block prints', 'woolens and pheran', 'beachwear')",
  "occasion_or_festival": "string or null (the event or occasion, e.g. 'Garba', 'Diwali', 'job interview', 'school', 'office')",
  "event_requirements": "string or null (the style of clothing/accessories required, e.g. 'traditional ethnic', 'formal wear', 'school bags')",
  "exclude_brands": "array of strings",
  "negated_features": "array of strings",
  "color": "string or null",
  "size": "string or null",
  "gender": "string or null (one of: 'men', 'women', 'unisex')",
  "clarification_questions": "array of strings or null (if the query is vague, provide 1-2 category-specific clarification questions, otherwise null)"
}}
"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            res_json = response.json()
            content = res_json["choices"][0]["message"]["content"].strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error extracting metadata via Groq: {str(e)}")
            return None

    def _generate_response_via_groq(self, user_query: str, history: List[Dict[str, str]], prompt: str) -> Optional[str]:
        """Fallback conversational response generation using Groq's Llama 3 API."""
        if not self.groq_api_key:
            return None
            
        system_instruction = self.system_instruction
        
        # Build message history in standard OpenAI format
        messages = [{"role": "system", "content": system_instruction}]
        for turn in history:
            messages.append({
                "role": "user" if turn["role"] == "user" else "assistant",
                "content": turn["content"]
            })
        messages.append({"role": "user", "content": prompt})
        
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.7
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error generating response via Groq: {str(e)}")
            return None
    def _call_openai_compatible_api(self, api_key: str, endpoint: str, model: str, messages: List[Dict[str, Any]], temperature: float = 0.0, response_format: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Calls any OpenAI-compatible chat completion endpoint."""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            if response_format:
                payload["response_format"] = response_format
                
            response = requests.post(endpoint, json=payload, headers=headers, timeout=12)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI-compatible endpoint ({endpoint}, model: {model}): {str(e)}")
            return None

    def _extract_entities_via_fallback_llms(self, query: str, history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Try other LLMs (DeepSeek, Mistral, Qwen, Llama/Together) for metadata extraction."""
        # Format history context
        history_context = ""
        for turn in history[-4:]:
            role = "Customer" if turn["role"] == "user" else "Sales Associate"
            history_context += f"{role}: {turn['content']}\n"
            
        system_instruction = "You are a precise JSON extractor. Analyze the customer query and chat history to extract structured shopping metadata."
        
        prompt = f"""
CHAT HISTORY:
{history_context}

NEW CUSTOMER QUERY:
"{query}"

Return ONLY a JSON object with the following fields (no explanations, markdown blocks, or other text):
{{
  "intent": "string (one of: 'greeting', 'recommendation', 'comparison', 'filtering', 'general query', 'clarification', 'out_of_scope')",
  "is_out_of_scope": "boolean (true if the query is asking for products, services, or topics completely unrelated to our store categories [Smartphones, Laptops, Fashion, Shoes, Skincare, Fitness], false otherwise)",
  "comparison_targets": "array of strings (if comparison intent, list the specific products/brands mentioned, e.g. ['iPhone 15 Pro', 'Galaxy S24'])",
  "place": "string or null (if the query mentions a place, city, region, or country, e.g. 'Rajasthan', 'Goa', 'Kashmir', 'London')",
  "weather_context": "string or null (the typical weather/climate of that place, e.g. 'hot and dry', 'cold and snowy', 'tropical and humid')",
  "cultural_style": "string or null (traditional or cultural style associated with that place, e.g. 'bandhani block prints', 'woolens and pheran', 'beachwear')",
  "occasion_or_festival": "string or null (the event or occasion, e.g. 'Garba', 'Diwali', 'job interview', 'school', 'office')",
  "event_requirements": "string or null (the style of clothing/accessories required, e.g. 'traditional ethnic', 'formal wear', 'school bags')",
  "exclude_brands": "array of strings",
  "negated_features": "array of strings",
  "color": "string or null",
  "size": "string or null",
  "gender": "string or null (one of: 'men', 'women', 'unisex')",
  "clarification_questions": "array of strings or null (if the query is vague, provide 1-2 category-specific clarification questions, otherwise null)"
}}
"""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        # 1. Try DeepSeek
        if self.deepseek_api_key:
            logger.info("Attempting NLU extraction via DeepSeek...")
            res = self._call_openai_compatible_api(
                api_key=self.deepseek_api_key,
                endpoint="https://api.deepseek.com/v1/chat/completions",
                model="deepseek-chat",
                messages=messages,
                response_format={"type": "json_object"}
            )
            if res:
                try:
                    return json.loads(res)
                except Exception:
                    pass
                    
        # 2. Try Mistral
        if self.mistral_api_key:
            logger.info("Attempting NLU extraction via Mistral...")
            res = self._call_openai_compatible_api(
                api_key=self.mistral_api_key,
                endpoint="https://api.mistral.ai/v1/chat/completions",
                model="mistral-large-latest",
                messages=messages,
                response_format={"type": "json_object"}
            )
            if res:
                try:
                    return json.loads(res)
                except Exception:
                    pass

        # 3. Try Qwen (DashScope compatible endpoint)
        if self.qwen_api_key:
            logger.info("Attempting NLU extraction via Qwen...")
            res = self._call_openai_compatible_api(
                api_key=self.qwen_api_key,
                endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                model="qwen-plus",
                messages=messages
            )
            if res:
                try:
                    clean_text = res.strip()
                    if "```json" in clean_text:
                        clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_text:
                        clean_text = clean_text.split("```")[1].split("```")[0].strip()
                    return json.loads(clean_text)
                except Exception:
                    pass

        # 4. Try Llama via Together
        if self.llama_api_key:
            logger.info("Attempting NLU extraction via Llama/Together...")
            res = self._call_openai_compatible_api(
                api_key=self.llama_api_key,
                endpoint="https://api.together.xyz/v1/chat/completions",
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                messages=messages,
                response_format={"type": "json_object"}
            )
            if res:
                try:
                    return json.loads(res)
                except Exception:
                    pass

        return None

    def _generate_response_via_fallback_llms(self, user_query: str, history: List[Dict[str, str]], prompt: str) -> Optional[str]:
        """Try other LLMs (DeepSeek, Mistral, Qwen, Llama/Together) for response generation."""
        system_instruction = self.system_instruction
        messages = [{"role": "system", "content": system_instruction}]
        for turn in history:
            messages.append({
                "role": "user" if turn["role"] == "user" else "assistant",
                "content": turn["content"]
            })
        messages.append({"role": "user", "content": prompt})

        # 1. Try DeepSeek
        if self.deepseek_api_key:
            logger.info("Attempting response generation via DeepSeek...")
            res = self._call_openai_compatible_api(
                api_key=self.deepseek_api_key,
                endpoint="https://api.deepseek.com/v1/chat/completions",
                model="deepseek-chat",
                messages=messages,
                temperature=0.7
            )
            if res:
                return res

        # 2. Try Mistral
        if self.mistral_api_key:
            logger.info("Attempting response generation via Mistral...")
            res = self._call_openai_compatible_api(
                api_key=self.mistral_api_key,
                endpoint="https://api.mistral.ai/v1/chat/completions",
                model="mistral-large-latest",
                messages=messages,
                temperature=0.7
            )
            if res:
                return res

        # 3. Try Qwen
        if self.qwen_api_key:
            logger.info("Attempting response generation via Qwen...")
            res = self._call_openai_compatible_api(
                api_key=self.qwen_api_key,
                endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                model="qwen-plus",
                messages=messages,
                temperature=0.7
            )
            if res:
                return res

        # 4. Try Llama via Together
        if self.llama_api_key:
            logger.info("Attempting response generation via Llama/Together...")
            res = self._call_openai_compatible_api(
                api_key=self.llama_api_key,
                endpoint="https://api.together.xyz/v1/chat/completions",
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                messages=messages,
                temperature=0.7
            )
            if res:
                return res

        return None

    def _merge_and_validate_nlu(self, data: Optional[Dict[str, Any]], fallback: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Helper to validate, merge and sanitize extracted metadata."""
        if not data:
            return None
            
        if not data.get("is_out_of_scope"):
            data["is_out_of_scope"] = False
        
        if data.get("is_out_of_scope") or data.get("intent") == "out_of_scope":
            data["category"] = None
            data["intent"] = "out_of_scope"
            data["is_out_of_scope"] = True
        elif not data.get("category"):
            data["category"] = fallback["category"]
            
        if not data.get("budget"):
            data["budget"] = fallback["budget"]
        if not data.get("search_query"):
            data["search_query"] = fallback["search_query"]
        if "exclude_brands" not in data or not data["exclude_brands"]:
            data["exclude_brands"] = fallback["exclude_brands"]
        if "negated_features" not in data or not data["negated_features"]:
            data["negated_features"] = fallback["negated_features"]
        if not data.get("color"):
            data["color"] = fallback["color"]
        if not data.get("size"):
            data["size"] = fallback["size"]
        if not data.get("gender"):
            data["gender"] = fallback["gender"]
        if not data.get("intent"):
            data["intent"] = fallback["intent"]
        if "comparison_targets" not in data or not data["comparison_targets"]:
            data["comparison_targets"] = fallback["comparison_targets"]
        if not data.get("place"):
            data["place"] = None
        if not data.get("weather_context"):
            data["weather_context"] = None
        if not data.get("cultural_style"):
            data["cultural_style"] = None
        if not data.get("occasion_or_festival"):
            data["occasion_or_festival"] = None
        if not data.get("event_requirements"):
            data["event_requirements"] = None
        if "clarification_questions" not in data:
            data["clarification_questions"] = None
            
        return data

    def extract_entities_and_intent(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyzes the query using LLM to extract intent, category, budget, brand, features, etc."""
        # 1. Parse comparison intent and targets rule-based
        query_clean = query.lower()
        fallback_intent = "recommendation"
        comparison_targets = []
        
        # Detect if comparison words exist
        is_compare_query = any(word in query_clean for word in ["compare", " vs ", " versus ", "difference between", " better ", " or "])
        
        if is_compare_query:
            fallback_intent = "comparison"
            
            # Split-based extraction (e.g., "A vs B" or "Compare A and B")
            if " vs " in query_clean:
                parts = query.split(" vs ")
                comparison_targets = [p.replace("compare", "").replace("Compare", "").strip() for p in parts]
            elif " versus " in query_clean:
                parts = query.split(" versus ")
                comparison_targets = [p.replace("compare", "").replace("Compare", "").strip() for p in parts]
            elif "difference between" in query_clean:
                parts = query_clean.split("difference between")
                if len(parts) > 1 and " and " in parts[1]:
                    sub_parts = parts[1].split(" and ")
                    comparison_targets = [sp.strip() for sp in sub_parts]
            elif "compare " in query_clean:
                # Match "compare <target1> with/to/and <target2>"
                match = re.search(r"compare\s+(.+?)\s+(?:with|to|and)\s+(.+)", query_clean)
                if match:
                    comparison_targets = [match.group(1).strip(), match.group(2).strip()]
                else:
                    parts = query_clean.split("compare")
                    if len(parts) > 1 and " and " in parts[1]:
                        sub_parts = parts[1].split(" and ")
                        comparison_targets = [sp.strip() for sp in sub_parts]
            
            # Brand-based extraction backup if no targets extracted yet
            if not comparison_targets:
                brands = ["dell", "hp", "acer", "samsung", "infinix", "primebook", "asus", "lenovo", "msi", 
                          "zara", "allen solly", "peter england", "levi", "roadster", "h&m", "aroma", "philips", 
                          "boat", "noise", "vivo", "mivi", "ptron", "zebronics", "apple"]
                found_brands = []
                for b in brands:
                    if b in query_clean:
                        idx = query_clean.find(b)
                        found_brands.append(query[idx:idx+len(b)].strip())
                if len(found_brands) >= 2:
                    comparison_targets = found_brands

        # Single brand extraction
        brand_val = None
        if not comparison_targets:
            brands = ["dell", "hp", "acer", "samsung", "infinix", "primebook", "asus", "lenovo", "msi", 
                      "zara", "allen solly", "peter england", "levi's", "roadster", "h&m", "aroma", "philips", 
                      "boat", "noise", "vivo", "mivi", "ptron", "zebronics", "apple", "manyavar"]
            for b in brands:
                if b in query_clean:
                    idx = query_clean.find(b)
                    brand_val = query[idx:idx+len(b)].strip()
                    break

        # Rule-based features and lifestyle extraction backup
        features_list = []
        for word in ["size s", "size m", "size l", "cotton", "denim", "linen", "polyester", "silk", "wool", "slim fit", "regular fit", "relaxed fit", "tuxedo", "suit", "sherwani", "lehenga", "saree"]:
            if word in query_clean:
                features_list.append(word)
        for word in ["men", "women", "unisex", "male", "female", "girl", "boy"]:
            if word in query_clean:
                features_list.append(word)
        for word in ["dry skin", "oily skin", "sensitive skin", "combination skin", "acne", "wrinkles", "spf", "sun protection", "hydration", "moisturizing"]:
            if word in query_clean:
                features_list.append(word)
        for word in ["gaming", "student", "office", "programming", "developer", "coding", "business", "camera", "battery", "display", "ram"]:
            if word in query_clean:
                features_list.append(word)
        for word in ["running", "walking", "sports", "casual", "formal", "gym"]:
            if word in query_clean:
                features_list.append(word)

        lifestyle_val = None
        for word in ["wedding", "party", "goa", "college", "office", "gym", "running"]:
            if word in query_clean:
                lifestyle_val = word
                break

        # Carry over context from history for rule-based fallback
        prev_category = None
        prev_gender = None
        prev_place = None
        prev_occasion = None
        prev_brand = None
        
        # Scan history from most recent to oldest
        for turn in reversed(history):
            content = turn.get("content", "").lower()
            
            # Extract category from previous messages
            if not prev_category:
                prev_category = self.extract_category_rule_based(content)
                
            # If the category has changed, do not carry over previous context (like gender, place, occasion, brand)
            current_category = self.extract_category_rule_based(query_clean)
            if current_category and prev_category and current_category.lower() != prev_category.lower():
                break
                
            # Extract gender from previous messages
            if not prev_gender:
                for word in ["women", "female", "girl", "lady", "ladies"]:
                    if word in content:
                        prev_gender = "women"
                        break
                if not prev_gender:
                    for word in ["men", "male", "boy", "gent", "gents"]:
                        if word in content:
                            prev_gender = "men"
                            break
                if not prev_gender:
                    if "unisex" in content:
                        prev_gender = "unisex"
                        
            # Extract place from previous messages
            if not prev_place:
                for word in ["kashmir", "himachal", "manali", "shimla", "ladakh", "masoorie", "mussoorie", "goa", "rajasthan", "jaipur"]:
                    if word in content:
                        prev_place = "Kashmir" if "kashmir" in word else word.capitalize()
                        break
                        
            # Extract occasion from previous messages
            if not prev_occasion:
                for word in ["garba", "navratri", "diwali", "wedding", "school", "college", "interview", "office", "freshers"]:
                    if word in content:
                        prev_occasion = "freshers party" if "freshers" in word else word.capitalize()
                        break

        # Simple fallback parsing for color and size
        color_val = None
        color_match = re.search(r"\b(red|blue|black|green|white|yellow|pink|grey|brown|purple|orange|gold|silver)\b", query_clean)
        if color_match:
            color_val = color_match.group(1).capitalize()
            
        size_val = None
        size_match = re.search(r"\b(size\s+([sml]|xl|xxl)|s\b|m\b|l\b|xl\b|xxl\b)", query_clean)
        if size_match:
            size_val = size_match.group(1).upper().replace("SIZE ", "").strip()

        # Simple fallback parsing for gender
        current_gender = None
        for word in ["women", "female", "girl", "lady", "ladies"]:
            if word in query_clean:
                current_gender = "women"
                break
        if not current_gender:
            for word in ["men", "male", "boy", "gent", "gents"]:
                if word in query_clean:
                    current_gender = "men"
                    break
        if not current_gender:
            if "unisex" in query_clean:
                current_gender = "unisex"
                
        gender_val = current_gender or prev_gender

        # Simple fallback parsing for negations (e.g. "not samsung", "without fragrance")
        exclude_brands_fallback = []
        negated_features_fallback = []
        
        negation_match = re.findall(r"(?:no|not|except|without|excluding|free of)\s+(\w+(?:\s+\w+)?)", query_clean)
        for neg in negation_match:
            brands = ["dell", "hp", "acer", "samsung", "infinix", "primebook", "asus", "lenovo", "msi", 
                      "zara", "allen solly", "peter england", "levi's", "roadster", "h&m", "aroma", "philips", 
                      "boat", "noise", "vivo", "mivi", "ptron", "zebronics", "apple", "manyavar", "xiaomi", "neutrogena", "derma co", "mamaearth", "pilgrim", "aqualogica", "foxtale"]
            matched_brand = None
            for b in brands:
                if b in neg:
                    matched_brand = b
                    break
            if matched_brand:
                exclude_brands_fallback.append(matched_brand)
            else:
                negated_features_fallback.append(neg)

        # Local place fallback extraction
        query_lower = query.lower()
        place_val = None
        weather_val = None
        culture_val = None
        for word in ["kashmir", "himachal", "manali", "shimla", "ladakh", "masoorie", "mussoorie"]:
            if word in query_lower:
                place_val = "Kashmir" if "kashmir" in word else word.capitalize()
                weather_val = "cold winter weather" if "kashmir" in word else "cold and hilly weather"
                culture_val = "warm woolens"
                break
        if not place_val:
            for word in ["goa", "beach"]:
                if word in query_lower:
                    place_val = "Goa"
                    weather_val = "tropical beach weather"
                    culture_val = "light beachwear, floral styles"
                    break
        if not place_val:
            for word in ["rajasthan", "rajasthani", "jaipur"]:
                if word in query_lower:
                    place_val = "Rajasthan"
                    weather_val = "hot and dry desert weather"
                    culture_val = "traditional block prints, ethnic bandhani styles"
                    break
                    
        # If not in current query, inherit from history
        if not place_val and prev_place:
            place_val = prev_place
            p_lower = place_val.lower()
            if any(w in p_lower for w in ["kashmir", "himachal", "manali", "shimla", "ladakh", "masoorie", "mussoorie"]):
                weather_val = "cold winter weather" if "kashmir" in p_lower else "cold and hilly weather"
                culture_val = "warm woolens"
            elif any(w in p_lower for w in ["goa", "beach"]):
                weather_val = "tropical beach weather"
                culture_val = "light beachwear, floral styles"
            elif any(w in p_lower for w in ["rajasthan", "rajasthani", "jaipur"]):
                weather_val = "hot and dry desert weather"
                culture_val = "traditional block prints, ethnic bandhani styles"

        # Local occasion fallback extraction
        occasion_val = None
        requirements_val = None
        for word in ["garba", "navratri"]:
            if word in query_lower:
                occasion_val = "Garba"
                requirements_val = "traditional ethnic wear"
                break
        if not occasion_val:
            for word in ["diwali", "wedding"]:
                if word in query_lower:
                    occasion_val = word.capitalize()
                    requirements_val = "designer ethnic wear"
                    break
        if not occasion_val:
            for word in ["school", "college"]:
                if word in query_lower:
                    occasion_val = word
                    requirements_val = "school bags and everyday wear"
                    break
        if not occasion_val:
            for word in ["interview", "office"]:
                if word in query_lower:
                    occasion_val = word
                    requirements_val = "formal suits and blazers"
                    break
        if not occasion_val:
            for word in ["freshers", "party"]:
                if word in query_lower:
                    occasion_val = "freshers party" if "freshers" in word else "party"
                    requirements_val = "party dress"
                    break

        # If not in current query, inherit from history
        if not occasion_val and prev_occasion:
            occasion_val = prev_occasion
            o_lower = occasion_val.lower()
            if "freshers" in o_lower or "party" in o_lower:
                requirements_val = "party dress"
            elif "garba" in o_lower or "navratri" in o_lower:
                requirements_val = "traditional ethnic wear"
            elif "diwali" in o_lower or "wedding" in o_lower:
                requirements_val = "designer ethnic wear"
            elif "school" in o_lower or "college" in o_lower:
                requirements_val = "school bags and everyday wear"
            elif "interview" in o_lower or "office" in o_lower:
                requirements_val = "formal suits and blazers"

        out_of_scope_keywords = ["tea", "coffee", "pizza", "burger", "food", "car", "flight", "hotel", "restaurant", "movie", "booking", "cab", "taxi"]
        is_out_of_scope_fallback = any(w in query_lower for w in out_of_scope_keywords)
        
        category_val = self.extract_category_rule_based(query) or prev_category or ("Fashion" if place_val or occasion_val else None)
        if is_out_of_scope_fallback:
            category_val = None
            fallback_intent = "out_of_scope"

        # Rule-based query expansion for common vague context queries as a fallback
        search_query_val = query
        q_lower = query_lower
        occ_lower = occasion_val.lower() if occasion_val else ""
        plc_lower = place_val.lower() if place_val else ""
        
        if "navratri" in q_lower or "garba" in q_lower or "navratri" in occ_lower or "garba" in occ_lower:
            search_query_val = "ethnic lehenga saree kurti chaniya choli traditional"
        elif "diwali" in q_lower or "wedding" in q_lower or "diwali" in occ_lower or "wedding" in occ_lower:
            search_query_val = "designer lehenga saree kurti wedding traditional ethnic"
        elif any(w in q_lower for w in ["kashmir", "winter", "cold", "masoorie", "mussoorie", "shimla", "manali"]) or any(w in plc_lower for w in ["kashmir", "himachal", "manali", "shimla", "masoorie", "mussoorie"]):
            search_query_val = "winter jacket sweater warm shawl coat cardigan jacket hoodie"
        elif "goa" in q_lower or "beach" in q_lower or "goa" in plc_lower or "beach" in plc_lower:
            search_query_val = "shorts dress maxi t-shirt beachwear floral"
        elif any(w in q_lower for w in ["interview", "office", "formal"]) or any(w in occ_lower for w in ["interview", "office", "formal"]):
            search_query_val = "formal shirt trousers suit blazer coat"
        else:
            # Short query refinement logic
            if len(query.split()) <= 2:
                parts = [query]
                if gender_val:
                    parts.append(gender_val)
                if requirements_val:
                    parts.append(requirements_val)
                elif category_val == "Fashion" or prev_category == "Fashion":
                    parts.append("clothing dress shirt")
                elif category_val == "Shoes" or prev_category == "Shoes":
                    parts.append("shoes sneakers")
                search_query_val = " ".join(parts)

        fallback = {
            "intent": fallback_intent,
            "category": category_val,
            "budget": self.parse_budget(query),
            "brand": brand_val,
            "features": features_list,
            "lifestyle": lifestyle_val,
            "comparison_targets": [t for t in comparison_targets if t],
            "search_query": search_query_val,
            "exclude_brands": exclude_brands_fallback,
            "negated_features": negated_features_fallback,
            "color": color_val,
            "size": size_val,
            "gender": gender_val,
            "place": place_val,
            "weather_context": weather_val,
            "cultural_style": culture_val,
            "occasion_or_festival": occasion_val,
            "event_requirements": requirements_val,
            "is_out_of_scope": is_out_of_scope_fallback
        }
        
        if not self.model:
            if self.groq_api_key:
                logger.info("Gemini model not configured. Using Groq fallback for metadata extraction.")
                groq_data = self._extract_entities_via_groq(query, history)
                merged = self._merge_and_validate_nlu(groq_data, fallback)
                if merged:
                    return merged
            
            # Try other fallback LLMs (DeepSeek, Mistral, Qwen, Llama/Together)
            logger.info("Groq failed or not configured. Trying other fallback LLMs...")
            fallback_data = self._extract_entities_via_fallback_llms(query, history)
            merged = self._merge_and_validate_nlu(fallback_data, fallback)
            if merged:
                return merged
                
            return fallback

        # Format historical turns for context
        history_context = ""
        for turn in history[-4:]:
            role = "Customer" if turn["role"] == "user" else "Sales Associate"
            history_context += f"{role}: {turn['content']}\n"

        prompt = f"""
Analyze the following e-commerce customer query and chat history to extract structured shopping metadata.

CHAT HISTORY:
{history_context}

NEW CUSTOMER QUERY:
"{query}"

IMPORTANT CONTEXT RULES:
1. STRICT TOPIC RESET ON CATEGORY CHANGE: If the Customer's new query changes categories or changes the primary product type (e.g., switching from clothing/Fashion to Shoes, Laptops to Skincare, or dress/frock to athletic shoes), you MUST immediately treat it as a completely new search topic and session.
2. STRICT ENTITY RESET: When a topic reset occurs:
   - Reset the brand, budget, size, color, lifestyle, features, and comparison targets.
   - You MUST reset and clear 'gender' (set to null if not explicitly mentioned in the new query) and DO NOT carry over any previous gender from history (e.g. do not carry over "women" or "girl" if they are now asking for general shoes or men's products).
   - You MUST reset and clear 'lifestyle', 'place', 'weather_context', 'cultural_style', 'occasion_or_festival', and 'event_requirements' to null. Do NOT carry over previous events, trips, occasions, or styles (e.g., if they previously asked about a trip to Rajasthan or Garba, discard them completely when they search for shoes, laptops, or general products).
3. NO CONVERSATIONAL CONTAMINATION: Do not let previous constraints bleed into the new product context. The extracted metadata MUST reflect ONLY the new query unless the customer explicitly references the previous parameters (e.g., "same brand", "under that budget", "in the same size").

Return ONLY a JSON object with the following fields (no explanations, markdown blocks, or other text):
{{
  "intent": "string (one of: 'greeting', 'recommendation', 'comparison', 'filtering', 'general query', 'clarification', 'out_of_scope')",
  "is_out_of_scope": "boolean (true if the query is asking for products, services, or topics completely unrelated to our store categories [Smartphones, Laptops, Fashion, Shoes, Skincare, Fitness], false otherwise)",
  "category": "string or null (strictly standard names: 'Smartphones', 'Laptops', 'Fashion', 'Shoes', 'Skincare', 'Fitness', 'Accessories'. Note: If the customer mentions a trip, travel, holiday, or visiting a place, set this to 'Fashion' or 'Accessories' depending on whether they need clothes or gear.)",
  "budget": "number or null (extracted maximum numerical budget, convert '50k' to 50000, etc.)",
  "brand": "string or null (e.g. 'Apple', 'Samsung', 'Nike', 'Minimalist')",
  "features": "array of strings (e.g. ['camera', 'battery', 'cotton', 'dry skin', 'lightweight'])",
  "lifestyle": "string or null (e.g. 'Goa trip', 'college studies', 'gym beginner', 'anti-aging')",
  "comparison_targets": "array of strings (if comparison intent, list the specific products/brands mentioned, e.g. ['iPhone 15 Pro', 'Galaxy S24'])",
  "search_query": "string or null (optimized search keywords for a vector database. Strip conversational fillers. Crucial: Translate vague contextual/occasional/weather/cultural/travel terms into concrete database product terms. E.g.: 'Navratri/Garba' -> 'ethnic lehenga saree kurti'; 'Kashmir' or 'Masoorie' or 'cold place trip' -> 'winter jacket sweater warm shawl coat cardigan hoodie'; 'Goa' or 'beach' -> 'beachwear shorts dress maxi t-shirt'; 'job interview' -> 'formal shirt trousers suit blazer')",
  "exclude_brands": "array of strings (any brand the customer explicitly does NOT want, e.g. ['Samsung', 'HP'])",
  "negated_features": "array of strings (any specifications or ingredients the customer wants to avoid, e.g. ['oil', 'fragrance', 'leather', 'Intel'])",
  "color": "string or null (any specific color requested, e.g. 'Red', 'Blue', 'Black')",
  "size": "string or null (any specific size requested, e.g. 'S', 'M', 'L', 'XL')",
  "gender": "string or null (one of: 'men', 'women', 'unisex')",
  "place": "string or null (if the query mentions a place, city, region, or country, e.g. 'Rajasthan', 'Goa', 'Kashmir', 'London')",
  "weather_context": "string or null (the typical weather/climate of that place, e.g. 'hot and dry', 'cold and snowy', 'tropical and humid')",
  "cultural_style": "string or null (traditional or cultural style associated with that place, e.g. 'bandhani block prints', 'woolens and pheran', 'beachwear')",
  "occasion_or_festival": "string or null (the event or occasion, e.g. 'Garba', 'Diwali', 'job interview', 'school', 'office')",
  "event_requirements": "string or null (the style of clothing/accessories required, e.g. 'traditional ethnic', 'formal wear', 'school bags')",
  "clarification_questions": "array of strings or null (if the query is vague or lacks details, provide 1-2 dynamic, category-specific clarification questions, otherwise null)"
}}
"""
        try:
            response = self.model.generate_content(prompt)
            clean_text = response.text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(clean_text)
            
            return self._merge_and_validate_nlu(data, fallback)
        except Exception as e:
            logger.error(f"Error extracting metadata via LLM: {str(e)}. Trying Groq fallback.")
            if self.groq_api_key:
                groq_data = self._extract_entities_via_groq(query, history)
                merged = self._merge_and_validate_nlu(groq_data, fallback)
                if merged:
                    logger.info("Successfully extracted metadata via Groq fallback.")
                    return merged
            
            # Try other fallback LLMs (DeepSeek, Mistral, Qwen, Llama/Together)
            logger.info("Groq failed or not configured. Trying other fallback LLMs...")
            fallback_data = self._extract_entities_via_fallback_llms(query, history)
            merged = self._merge_and_validate_nlu(fallback_data, fallback)
            if merged:
                logger.info("Successfully extracted metadata via fallback LLMs.")
                return merged
                
            logger.warning("All metadata extraction fallbacks failed. Using local rule-based fallback.")
            return fallback

    def get_clarification_questions(self, category: Optional[str], budget: Optional[float], brand: Optional[str]) -> List[str]:
        """Returns follow-up clarification questions if info is missing."""
        questions = []
        if not category:
            questions.append("What type of product are you looking for? (e.g. Smartphone, Laptop, Skincare, Shoes, Fitness, Fashion)")
            return questions
            
        if category == "Smartphones":
            if not budget:
                questions.append("What is your budget? (e.g. under ₹20,000, under ₹50,000)")
            if not brand:
                questions.append("Do you have any brand preference? (e.g. Samsung, Apple, Vivo)")
            questions.append("What is most important to you: Camera quality, Battery life, Gaming performance, or Display?")
            
        elif category == "Laptops":
            if not budget:
                questions.append("What is your approximate budget? (e.g. under ₹40,000, around ₹75,000)")
            questions.append("What is your main use case: Student work, Gaming, Software Engineering, or Daily browsing?")
            
        elif category == "Fashion":
            questions.append("Is this for a Man, Woman, or Unisex? And what style/size are you looking for?")
            
        elif category == "Shoes":
            questions.append("What is the main purpose of the shoes? (e.g. Gym workout, Running, Casual wear, or Office formal)")
            
        elif category == "Skincare":
            questions.append("What is your skin type? (e.g. Oily, Dry, Sensitive, Combination)")
            questions.append("What skin benefits are you looking for? (e.g. Sun protection, Hydration, Acne control)")
            
        elif category == "Fitness":
            questions.append("Are you looking for dumbbells, yoga mats, water bottles, or supplements?")
            
        else:
            questions.append("Can you tell me more about your specific preferences or requirements?")
            
        return questions

    def build_comparison_table(self, targets: List[str], category: Optional[str]) -> Optional[Dict[str, Any]]:
        """Finds matching products in DB and formats a side-by-side comparison."""
        if not targets:
            return None
            
        products = []
        for target in targets:
            hits = self.search_engine.search(target, category_filter=category, top_k=1)
            if hits:
                products.append(hits[0])

        if len(products) < 2:
            if products:
                cat = products[0]["category"]
                alternatives = self.search_engine.search(f"best {cat}", category_filter=cat, top_k=2)
                for alt in alternatives:
                    if alt["product_id"] != products[0]["product_id"] and len(products) < 3:
                        products.append(alt)
            else:
                return None

        # Build specs sheet
        headers = ["Aspect"] + [p["product_title"][:30] + "..." for p in products]
        comparison_rows = []
        
        # Add common rows
        comparison_rows.append(["Price",] + [f"₹{p['selling_price']:.0f}" for p in products])
        comparison_rows.append(["Rating",] + [f"★ {p['rating']} ({p['review_count']} reviews)" for p in products])
        
        # Extract unique list of specification keys present in these products
        all_spec_keys = set()
        spec_fields = [
            "ram", "storage", "battery", "camera", "processor", "display", 
            "operating_system", "gpu", "battery_backup", "display_size",
            "material", "color", "size", "fit", "gender", "style",
            "type", "comfort_level", "purpose", "skin_type", "ingredients",
            "benefits", "spf", "compatibility"
        ]
        for p in products:
            for sf in spec_fields:
                if sf in p and p[sf] and str(p[sf]).strip() != "":
                    all_spec_keys.add(sf)

        for key in sorted(all_spec_keys):
            row = [key.replace("_", " ").capitalize()]
            for p in products:
                row.append(str(p.get(key, "N/A")))
            comparison_rows.append(row)
            
        # Add generated Pros and Cons
        pros_row = ["Key Pros"]
        cons_row = ["Key Cons"]
        for p in products:
            price = p["selling_price"]
            rating = p["rating"]
            
            pros = []
            cons = []
            
            if rating >= 4.4:
                pros.append("Highly rated by buyers")
            if price < 15000 and p["category"] == "Smartphones":
                pros.append("Highly budget-friendly")
            elif price < 40000 and p["category"] == "Laptops":
                pros.append("Excellent entry pricing")
                
            if "5000 mAh" in str(p.get("battery", "")) or "6000 mAh" in str(p.get("battery", "")):
                pros.append("Large battery backup")
            if "AMOLED" in str(p.get("display", "")):
                pros.append("Premium vibrant AMOLED screen")
            if "RTX" in str(p.get("gpu", "")):
                pros.append("Powerful GPU")
            if "Cotton" in str(p.get("material", "")):
                pros.append("Pure cotton comfort")
                
            if not pros:
                pros.append("Great value for money")
                pros.append("Sturdy build quality")
                
            if rating < 4.0:
                cons.append("Average rating")
            if price > 60000 and p["category"] == "Smartphones":
                cons.append("Expensive premium pricing")
            if not cons:
                cons.append("Limited color options")
                
            pros_row.append(", ".join(pros[:2]))
            cons_row.append(", ".join(cons[:2]))
            
        comparison_rows.append(pros_row)
        comparison_rows.append(cons_row)

        return {
            "products": products,
            "headers": headers,
            "rows": comparison_rows
        }

    def chat(self, user_query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process user message, search products, generate sales associate conversational answer."""
        logger.info(f"Processing message: {user_query}")
        
        # 1. Parse metadata and intent
        meta = self.extract_entities_and_intent(user_query, history)
        logger.info(f"Extracted metadata: {meta}")

        intent = meta.get("intent", "recommendation")
        category = meta.get("category")
        budget = meta.get("budget")
        brand = meta.get("brand")
        features = meta.get("features", [])
        lifestyle = meta.get("lifestyle")
        comparison_targets = meta.get("comparison_targets", [])
        exclude_brands = meta.get("exclude_brands", [])
        negated_features = meta.get("negated_features", [])

        # Resolve comparison target pronouns (like "this", "it") using assistant recommendation history
        if comparison_targets:
            resolved_targets = []
            for target in comparison_targets:
                if str(target).lower() in ["this", "it", "previous option", "that option", "the recommended product", "the previous one", "that"]:
                    prev_product = None
                    for turn in reversed(history):
                        if turn["role"] == "assistant":
                            # Extract product title inside **...**
                            matches = re.findall(r"\*\*(.*?)\*\*", turn["content"])
                            if matches:
                                prev_product = matches[0]
                                break
                    if prev_product:
                        resolved_targets.append(prev_product)
                    else:
                        resolved_targets.append(target)
                else:
                    resolved_targets.append(target)
            comparison_targets = resolved_targets

        color = meta.get("color")
        size = meta.get("size")
        gender = meta.get("gender")
        place = meta.get("place")
        weather_context = meta.get("weather_context")
        cultural_style = meta.get("cultural_style")
        occasion_or_festival = meta.get("occasion_or_festival")
        event_requirements = meta.get("event_requirements")

        # Check for greeting or vague query
        query_clean = user_query.strip().lower().rstrip("?.!")
        greetings_keywords = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "yo", "sup", "hola"]
        is_greeting = query_clean in greetings_keywords or any(query_clean.startswith(g + " ") for g in greetings_keywords)
        if is_greeting:
            intent = "greeting"

        is_vague = False
        if intent != "greeting":
            is_vague = (
                (len(user_query.split()) < 3 and not category and not lifestyle and not brand and not features) or 
                (not category and intent in ["recommendation", "filtering"] and not brand and not lifestyle and not features and not budget)
            )
            # A query is never vague if a specific place or occasion is mentioned
            if place or occasion_or_festival:
                is_vague = False

        retrieved_products = []
        comparison_data = None
        is_strictly_available = True

        if not is_greeting:
            # 2. Trigger semantic retrieval
            search_query = meta.get("search_query") or user_query
            if not search_query.strip():
                search_query = user_query
                
            query_to_search = search_query
            # If vague and category exists, do a default popular search for "soft recommendations"
            if is_vague and category:
                query_to_search = f"best {category}"
                
            retrieved_products = self.search_engine.search(
                query=query_to_search,
                category_filter=category,
                budget_filter=budget,
                brand_filter=brand,
                top_k=8,
                exclude_brands=exclude_brands,
                negated_features=negated_features,
                color_filter=color,
                size_filter=size,
                gender_filter=gender
            )
            
            # Validate that the requested product/brand is actually available in the store
            # Only perform strict availability check if it is NOT a vague query
            is_strictly_available = True
            if intent in ["recommendation", "comparison"] and not is_vague:
                if not retrieved_products:
                    is_strictly_available = False
                elif retrieved_products[0].get("semantic_score", 1.0) < 0.68:
                    is_strictly_available = False
                elif retrieved_products[0].get("is_alternative") == True:
                    is_strictly_available = False
                
                # If a specific brand is parsed, verify we have it in stock
                if brand and retrieved_products:
                    brand_matched = any(
                        brand.lower() in str(p.get("brand", "")).lower() or 
                        str(p.get("brand", "")).lower() in brand.lower() 
                        for p in retrieved_products[:3]
                    )
                    if not brand_matched:
                        is_strictly_available = False

                # Check for place or event-specific keyword matching to prevent hallucinations on unavailable items
                if is_strictly_available and retrieved_products:
                    match_keywords = []
                    if place:
                        match_keywords.append(place.lower())
                        if place.lower() == "rajasthan" or place.lower() == "rajasthani":
                            match_keywords.extend(["rajasthan", "rajasthani", "jaipur", "bandhani", "block print", "ethnic"])
                        elif place.lower() == "kashmir" or place.lower() == "kashmiri":
                            match_keywords.extend(["kashmir", "kashmiri", "pheran", "pashmina", "wool"])
                        elif place.lower() == "goa" or place.lower() == "goan":
                            match_keywords.extend(["goa", "goan", "beachwear", "beach", "floral"])
                    if occasion_or_festival:
                        match_keywords.append(occasion_or_festival.lower())
                        if "garba" in occasion_or_festival.lower() or "navratri" in occasion_or_festival.lower():
                            match_keywords.extend(["garba", "navratri", "chaniya", "lehenga", "saree", "ethnic"])
                        elif "school" in occasion_or_festival.lower():
                            match_keywords.extend(["school", "uniform", "backpack", "shoes"])
                        elif "interview" in occasion_or_festival.lower() or "office" in occasion_or_festival.lower():
                            match_keywords.extend(["interview", "office", "formal", "suit", "blazer", "shirt"])
                            
                    query_words = re.findall(r"\b[a-zA-Z]{3,}\b", user_query.lower())
                    stop_words = {
                        "the", "and", "for", "with", "under", "over", "above", "below", "from", "about",
                        "best", "good", "cheap", "nice", "great", "show", "find", "suggest", "recommend",
                        "wear", "clothes", "clothing", "item", "items", "product", "products", "buy", "shop",
                        "look", "looking", "want", "need", "search", "store", "online", "flipkart", "please",
                        "some", "many", "more", "most", "what", "which", "who", "whom", "this", "that", "these",
                        "those", "here", "there", "when", "where", "why", "how", "are", "was", "were", "have",
                        "has", "had", "does", "did", "doing", "would", "should", "could", "must"
                    }
                    for qw in query_words:
                        if qw not in stop_words and len(qw) > 3:
                            match_keywords.append(qw)
                            
                    specific_terms = list(set([kw for kw in match_keywords if len(kw) > 3]))
                    for term in specific_terms:
                        if term in [
                            "casual", "formal", "dress", "shirt", "suit", "jacket", "active", "style",
                            "cold", "warm", "hot", "cool", "summer", "winter", "spring", "autumn",
                            "weather", "climate", "feel", "feeling", "going", "trip", "travel", 
                            "holiday", "vacation", "tour", "visit", "visiting", "tomorrow", "today", 
                            "yesterday", "love", "like", "want", "need", "suggest", "recommend"
                        ]:
                            continue
                        term_found = False
                        for p in retrieved_products[:3]:
                            prod_title = str(p.get("product_title", "")).lower()
                            prod_desc = str(p.get("description", "")).lower()
                            prod_brand = str(p.get("brand", "")).lower()
                            prod_specs = str(p.get("specifications", "")).lower()
                            prod_cat = str(p.get("category", "")).lower()
                            if (term in prod_title or 
                                term in prod_desc or 
                                term in prod_brand or 
                                term in prod_cat or
                                term in prod_specs):
                                term_found = True
                                break
                        if not term_found:
                            is_strictly_available = False
                            break

            # 4. Handle comparisons
            if intent == "comparison" or len(comparison_targets) >= 2:
                comparison_data = self.build_comparison_table(comparison_targets, category)
                if comparison_data:
                    retrieved_products = comparison_data["products"]

        # 3. Check for clarification trigger:
        clarifications = []
        if meta.get("clarification_questions"):
            clarifications = [q for q in meta.get("clarification_questions") if q]
            
        if not clarifications and (is_vague or (intent == "clarification") or (not category and not retrieved_products and not is_greeting)):
            clarifications = self.get_clarification_questions(category, budget, brand)

        # 5. Generate conversational response
        response_content = ""
        
        # Build product context
        product_context = ""
        if not retrieved_products:
            product_context = "No products match the requested criteria in our store. We do NOT have this item/combination in stock."
        else:
            for rank, p in enumerate(retrieved_products[:5]):
                # Build specs list
                specs_list = []
                for sk in ["ram", "storage", "battery", "camera", "processor", "display", "operating_system", "gpu", "material", "comfort_level", "skin_type", "spf", "usage", "color", "size"]:
                    if sk in p and p[sk]:
                        specs_list.append(f"{sk}: {p[sk]}")
                specs_str = ", ".join(specs_list)

                is_alt = p.get("is_alternative", False)
                alt_info = f" | Alternative Match: True (Relaxed filters: {p.get('relaxed_filters')})" if is_alt else ""

                product_context += (
                    f"Product Rank {rank+1}{alt_info}:\n"
                    f"Title: {p['product_title']}\n"
                    f"Product ID: {p['product_id']}\n"
                    f"Category: {p['category']} | Brand: {p['brand']}\n"
                    f"Selling Price: Rs. {p['selling_price']:.0f} (Original: Rs. {p['original_price']:.0f}, Discount: {p['discount_percentage']:.0f}% off)\n"
                    f"Rating: {p['rating']} ({p['review_count']} reviews)\n"
                    f"Specs: {specs_str}\n"
                    f"Description: {p['description']}\n"
                    f"Search relevance scores: semantic similarity: {p['semantic_score']:.2f}, popularity: {p['popularity_score']:.2f}, price relevance: {p['price_relevance']:.2f}\n"
                    f"--------------------\n"
                )
            
        prompt = f"""
CONTEXT INFORMATION:
- Intended category: {category}
- Target budget: {budget}
- Brand preference: {brand}
- Lifestyle / Use Case: {lifestyle}
- Extracted features: {features}
- Color requested: {color}
- Size requested: {size}
- Intent: {intent}
- Is Vague Request: {is_vague}
- Is strictly in stock (exact matches found): {is_strictly_available}
- Place mentioned: {place}
- Weather of that place: {weather_context}
- Cultural style of that place: {cultural_style}
- Occasion or festival: {occasion_or_festival}
- Occasion/festival requirements: {event_requirements}

TOP PRODUCT MATCHES:
{product_context}

CUSTOMER'S LATEST QUERY:
"{user_query}"

Respond to the customer's latest query naturally following your persona instructions. Do not repeat the customer's query, just reply directly as the sales associate.
"""

        if not is_strictly_available or any(p.get("is_alternative") for p in retrieved_products):
            prompt += f"""
IMPORTANT CONVERSATIONAL RULES (STRICTLY OBSERVE):
1. Politely and humbly inform the customer that the specific item, brand, style, event-wear, or cultural item they requested (e.g., traditional Rajasthani wear, Garba clothes, school uniforms, etc.) is currently not available or out of stock in our collection.
2. Apologize and offer to recommend the best available items from our store catalog that are practical, suitable, and highly relevant for the location's climate ({weather_context or "the weather"}), the culture/event requirements ({cultural_style or event_requirements or "the occasion"}), and the customer's gender.
3. If they asked for a place/occasion (e.g. Rajasthan, Kashmir, Garba, school, interview), explain why the recommended alternatives are a good choice (for example: if Rajasthan is hot, suggest light breathable cotton shirts/dresses from our catalog; if Kashmir is cold, suggest warm windbreaker jackets/sweaters; if for school, suggest school backpacks and clean sports shoes; if for Garba, suggest beautiful sarees or lehengas we *do* have).
4. Make sure your recommendations strictly match the target gender (e.g. recommend women's wear if they are shopping for a woman).
5. Do NOT fabricate or claim that the available products have the unavailable place/cultural features. Never say a generic Zara dress has 'Rajasthani style' or 'Rajasthani flair' or that a generic shirt is from Kashmir if they are not. Be completely honest and transparent about their actual features and brands.
6. Do not mention "database", "is_strictly_available", "filters", or "alternative flags". Talk like a helpful, warm store associate.
"""

        if self.model:
            try:
                # Map history roles ('assistant' -> 'model') for Gemini chat session
                formatted_history = []
                for turn in history[-6:]:
                    role = "user" if turn["role"] == "user" else "model"
                    formatted_history.append({
                        "role": role,
                        "parts": [turn["content"]]
                    })
                chat_session = self.model.start_chat(history=formatted_history)
                response = chat_session.send_message(prompt)
                response_content = response.text.strip()
            except Exception as e:
                logger.error(f"Error generating LLM response via Gemini: {str(e)}. Trying Groq fallback.")
                groq_response = self._generate_response_via_groq(user_query, history, prompt)
                if groq_response:
                    logger.info("Successfully generated response via Groq fallback.")
                    response_content = groq_response
                else:
                    logger.info("Groq failed. Trying fallback LLMs (DeepSeek, Mistral, Qwen, Llama/Together)...")
                    fallback_response = self._generate_response_via_fallback_llms(user_query, history, prompt)
                    if fallback_response:
                        response_content = fallback_response
                    else:
                        logger.warning("All response generation fallbacks failed. Using offline fallback response.")
                        response_content = self._get_fallback_mock_response(user_query, intent, retrieved_products, clarifications, is_strictly_available=is_strictly_available, place=place, occasion_or_festival=occasion_or_festival)
        else:
            # Gemini not configured
            groq_response = None
            if self.groq_api_key:
                logger.info("Gemini not configured. Generating response directly via Groq.")
                groq_response = self._generate_response_via_groq(user_query, history, prompt)
                
            if groq_response:
                logger.info("Successfully generated response directly via Groq.")
                response_content = groq_response
            else:
                logger.info("Groq not configured or failed. Trying fallback LLMs (DeepSeek, Mistral, Qwen, Llama/Together)...")
                fallback_response = self._generate_response_via_fallback_llms(user_query, history, prompt)
                if fallback_response:
                    response_content = fallback_response
                else:
                    logger.warning("All response generation fallbacks failed. Using offline fallback response.")
                    response_content = self._get_fallback_mock_response(user_query, intent, retrieved_products, clarifications, is_strictly_available=is_strictly_available, place=place, occasion_or_festival=occasion_or_festival)

        return {
            "intent": intent,
            "category": category,
            "budget": budget,
            "brand": brand,
            "features": features,
            "lifestyle": lifestyle,
            "color": color,
            "size": size,
            "gender": gender,
            "response": response_content,
            "products": retrieved_products[:5],
            "comparison": comparison_data,
            "clarification_questions": clarifications,
            "place": place,
            "weather_context": weather_context,
            "cultural_style": cultural_style,
            "occasion_or_festival": occasion_or_festival,
            "event_requirements": event_requirements
        }

    def _get_fallback_mock_response(self, query: str, intent: str, products: List[Dict[str, Any]], clarifications: List[str], is_strictly_available: bool = True, place: Optional[str] = None, occasion_or_festival: Optional[str] = None) -> str:
        """Fallback conversational responder in case Gemini API is not configured or fails."""
        greeting = "Hello! I am your virtual sales associate today. "
        
        if intent == "out_of_scope":
            return "I apologize, but our store currently specializes only in Smartphones, Laptops, Fashion & Clothing, Shoes, Skincare, and Fitness gear. We don't carry any tea, food, cars, or other products/services at this time. Let me know if you would like to search for something else in our catalog!"

        if intent == "greeting":
            return greeting + "How can I help you shop today? We specialize in Smartphones, Laptops, Fashion (Dresses, Shirts, Jeans), Shoes, Fitness gear, Skincare, and Accessories! What type of product are you looking for?"

        if not is_strictly_available:
            resp = "I'm sorry, but we don't have the specific item, traditional style, or event-wear you requested "
            if place:
                resp += f"(like traditional items for {place}) "
            elif occasion_or_festival:
                resp += f"(like specific outfits for {occasion_or_festival}) "
            resp += "available in our store right now. "
            if products:
                resp += f"However, as a comfortable and practical alternative, let me suggest the **{products[0]['product_title']}** (Rs. {products[0]['selling_price']:.0f}) from our collection. Would you like to check it out, or search for other categories like Fashion or Accessories?"
            else:
                resp += "Would you like to search for other products we carry in Smartphones, Laptops, Fashion, Skincare, or Accessories?"
            return resp

        if not products or intent == "clarification":
            if clarifications:
                qs = "\n".join([f"- {q}" for q in clarifications])
                return f"I'd love to help you find the perfect product! To give you the best recommendation, could you tell me:\n{qs}"
            return greeting + "Could you tell me what kind of product you are looking for (e.g. Laptop, Phone, Sunscreen, Shoes) and if you have a budget or brand preference?"

        if intent == "comparison" and len(products) >= 2:
            p1 = products[0]
            p2 = products[1]
            resp = greeting + f"Here is a side-by-side comparison between **{p1['product_title']}** and **{p2['product_title']}**.\n\n"
            resp += f"**Key Differences:**\n"
            resp += f"- **Price**: {p1['product_title'][:20]}... is listed at ₹{p1['selling_price']:.0f}, while {p2['product_title'][:20]}... is ₹{p2['selling_price']:.0f}.\n"
            resp += f"- **Rating**: {p1['product_title'][:20]}... has a rating of {p1['rating']} ★ ({p1['review_count']} reviews) vs {p2['rating']} ★ ({p2['review_count']} reviews) for {p2['product_title'][:20]}....\n"
            
            # Check for specs difference
            for key in ["processor", "ram", "storage", "material", "comfort_level", "skin_type", "gpu"]:
                if key in p1 and key in p2 and p1[key] and p2[key] and p1[key] != p2[key]:
                    resp += f"- **{key.replace('_', ' ').capitalize()}**: {p1[key]} vs {p2[key]}.\n"
                    
            resp += f"\nI have compiled the full side-by-side comparison sheet for you in the **Product Comparer** tab on the right side of the screen!"
            return resp

        p1 = products[0]
        resp = greeting + f"I found some excellent options for your request. Let me recommend the **{p1['product_title']}**.\n\n"
        resp += f"**Why I recommend this:**\n"
        resp += f"- It's listed on Flipkart at a great price of ₹{p1['selling_price']:.0f} (marked down from ₹{p1['original_price']:.0f}, {p1['discount_percentage']:.0f}% off).\n"
        resp += f"- It holds a solid user rating of {p1['rating']} ★ based on {p1['review_count']} verified buyer reviews.\n"
        
        # Spec based explanation
        spec_bullets = []
        for key in ["processor", "ram", "storage", "battery", "camera", "material", "comfort_level", "skin_type", "ingredients", "usage", "compatibility", "gpu"]:
            if key in p1 and p1[key]:
                spec_bullets.append(f"- **{key.replace('_', ' ').capitalize()}**: {p1[key]}")
        
        if spec_bullets:
            resp += "\n**Key Specifications:**\n" + "\n".join(spec_bullets) + "\n"

        resp += f"\n**Strengths & Weaknesses:**\n"
        resp += f"- **Pros**: Highly reliable, excellent user reviews, and delivers outstanding value in this price bracket.\n"
        resp += f"- **Cons**: Color options or size availability might be limited depending on current stock.\n"
        
        if len(products) > 1:
            p2 = products[1]
            resp += f"\n**Alternative Option:**\n"
            resp += f"If you want to compare, you can also look at the **{p2['product_title']}** priced at ₹{p2['selling_price']:.0f}. It has a rating of {p2['rating']} ★.\n"
            
        if clarifications:
            resp += f"\n\nTo narrow this down further, let me ask: {random.choice(clarifications)}"
            
        return resp
