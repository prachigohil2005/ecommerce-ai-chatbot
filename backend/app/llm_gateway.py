import os
import json
import logging
import requests
import re
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai

logger = logging.getLogger("llm_gateway")

class LLMGateway:
    def __init__(self):
        # 1. Parse Gemini keys (comma-separated list)
        gemini_env = os.getenv("GEMINI_API_KEY", "")
        self.gemini_keys = [k.strip() for k in gemini_env.split(",") if k.strip()]
        self.current_gemini_idx = 0
        
        # 2. Parse Groq keys (comma-separated list)
        groq_env = os.getenv("GROQ_API_KEY", "")
        self.groq_keys = [k.strip() for k in groq_env.split(",") if k.strip()]
        self.current_groq_idx = 0
        
        # Fallback keys
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
        self.qwen_api_key = os.getenv("QWEN_API_KEY", "")
        self.llama_api_key = os.getenv("LLAMA_API_KEY", "")
        
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        logger.info(f"LLM Gateway initialized with {len(self.gemini_keys)} Gemini key(s) and {len(self.groq_keys)} Groq key(s).")

    def extract_metadata(self, query: str, history: List[Dict[str, Any]], system_instruction: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Extract structured metadata from query using Gemini/Groq keys with rotation, and fallbacks."""
        # 1. Try Gemini rotating keys
        if self.gemini_keys:
            attempts = 0
            while attempts < len(self.gemini_keys):
                idx = self.current_gemini_idx
                api_key = self.gemini_keys[idx]
                try:
                    logger.info(f"Attempting metadata extraction via Gemini (Key index: {idx})...")
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        model_name=self.model_name,
                        system_instruction=system_instruction
                    )
                    response = model.generate_content(prompt)
                    clean_text = response.text.strip()
                    if "```json" in clean_text:
                        clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_text:
                        clean_text = clean_text.split("```")[1].split("```")[0].strip()
                    data = json.loads(clean_text)
                    return data
                except Exception as e:
                    err_str = str(e)
                    logger.error(f"Gemini API key index {idx} NLU error: {err_str}")
                    # Rotate key on any failure
                    self.current_gemini_idx = (self.current_gemini_idx + 1) % len(self.gemini_keys)
                    attempts += 1

        # 2. Try Groq rotating keys
        if self.groq_keys:
            attempts = 0
            while attempts < len(self.groq_keys):
                idx = self.current_groq_idx
                api_key = self.groq_keys[idx]
                try:
                    logger.info(f"Attempting metadata extraction via Groq (Key index: {idx})...")
                    history_context = ""
                    for turn in history[-4:]:
                        role = "Customer" if turn["role"] == "user" else "Sales Associate"
                        history_context += f"{role}: {turn['content']}\n"
                    
                    groq_prompt = f"CHAT HISTORY:\n{history_context}\nNEW CUSTOMER QUERY:\n\"{query}\"\n{prompt}"
                    
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": "You are a precise JSON extractor. Analyze the customer query and chat history to extract structured shopping metadata."},
                            {"role": "user", "content": groq_prompt}
                        ],
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"}
                    }
                    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=10)
                    res.raise_for_status()
                    content = res.json()["choices"][0]["message"]["content"].strip()
                    data = json.loads(content)
                    return data
                except Exception as e:
                    err_str = str(e)
                    logger.error(f"Groq API key index {idx} NLU error: {err_str}")
                    # Rotate key on any failure
                    self.current_groq_idx = (self.current_groq_idx + 1) % len(self.groq_keys)
                    attempts += 1

        # 3. Try other fallbacks
        # DeepSeek
        if self.deepseek_api_key:
            try:
                logger.info("Attempting NLU extraction via DeepSeek...")
                data = self._call_openai_compatible_nlu(
                    self.deepseek_api_key, "https://api.deepseek.com/v1/chat/completions", "deepseek-chat", query, history, prompt
                )
                if data: return data
            except Exception as e:
                logger.error(f"DeepSeek NLU error: {e}")
                
        # Mistral
        if self.mistral_api_key:
            try:
                logger.info("Attempting NLU extraction via Mistral...")
                data = self._call_openai_compatible_nlu(
                    self.mistral_api_key, "https://api.mistral.ai/v1/chat/completions", "mistral-large-latest", query, history, prompt
                )
                if data: return data
            except Exception as e:
                logger.error(f"Mistral NLU error: {e}")

        # Qwen
        if self.qwen_api_key:
            try:
                logger.info("Attempting NLU extraction via Qwen...")
                data = self._call_openai_compatible_nlu(
                    self.qwen_api_key, "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen-max", query, history, prompt
                )
                if data: return data
            except Exception as e:
                logger.error(f"Qwen NLU error: {e}")

        # Llama/Together
        if self.llama_api_key:
            try:
                logger.info("Attempting NLU extraction via Together Llama...")
                data = self._call_openai_compatible_nlu(
                    self.llama_api_key, "https://api.together.xyz/v1/chat/completions", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", query, history, prompt
                )
                if data: return data
            except Exception as e:
                logger.error(f"Together Llama NLU error: {e}")

        return None

    def generate_response(self, user_query: str, history: List[Dict[str, Any]], system_instruction: str, prompt: str) -> Optional[str]:
        """Generate conversational response using Gemini/Groq keys with rotation, and fallbacks."""
        # 1. Try Gemini rotating keys
        if self.gemini_keys:
            attempts = 0
            while attempts < len(self.gemini_keys):
                idx = self.current_gemini_idx
                api_key = self.gemini_keys[idx]
                try:
                    logger.info(f"Attempting response generation via Gemini (Key index: {idx})...")
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        model_name=self.model_name,
                        system_instruction=system_instruction
                    )
                    formatted_history = []
                    for turn in history[-6:]:
                        role = "user" if turn["role"] == "user" else "model"
                        formatted_history.append({
                            "role": role,
                            "parts": [turn["content"]]
                        })
                    chat_session = model.start_chat(history=formatted_history)
                    response = chat_session.send_message(prompt)
                    return response.text.strip()
                except Exception as e:
                    err_str = str(e)
                    logger.error(f"Gemini API key index {idx} response error: {err_str}")
                    # Rotate key on any failure
                    self.current_gemini_idx = (self.current_gemini_idx + 1) % len(self.gemini_keys)
                    attempts += 1

        # 2. Try Groq rotating keys
        if self.groq_keys:
            attempts = 0
            while attempts < len(self.groq_keys):
                idx = self.current_groq_idx
                api_key = self.groq_keys[idx]
                try:
                    logger.info(f"Attempting response generation via Groq (Key index: {idx})...")
                    messages = [{"role": "system", "content": system_instruction}]
                    for turn in history[-6:]:
                        role = "user" if turn["role"] == "user" else "assistant"
                        messages.append({"role": role, "content": turn["content"]})
                    messages.append({"role": "user", "content": prompt})
                    
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "llama-3.3-70b-versatile",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 1024
                    }
                    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=15)
                    res.raise_for_status()
                    return res.json()["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    err_str = str(e)
                    logger.error(f"Groq API key index {idx} response error: {err_str}")
                    # Rotate key on any failure
                    self.current_groq_idx = (self.current_groq_idx + 1) % len(self.groq_keys)
                    attempts += 1

        # 3. Try other fallbacks
        # DeepSeek
        if self.deepseek_api_key:
            try:
                logger.info("Attempting response generation via DeepSeek...")
                res = self._call_openai_compatible_response(
                    self.deepseek_api_key, "https://api.deepseek.com/v1/chat/completions", "deepseek-chat", history, system_instruction, prompt
                )
                if res: return res
            except Exception as e:
                logger.error(f"DeepSeek response generation error: {e}")
                
        # Mistral
        if self.mistral_api_key:
            try:
                logger.info("Attempting response generation via Mistral...")
                res = self._call_openai_compatible_response(
                    self.mistral_api_key, "https://api.mistral.ai/v1/chat/completions", "mistral-large-latest", history, system_instruction, prompt
                )
                if res: return res
            except Exception as e:
                logger.error(f"Mistral response generation error: {e}")

        # Qwen
        if self.qwen_api_key:
            try:
                logger.info("Attempting response generation via Qwen...")
                res = self._call_openai_compatible_response(
                    self.qwen_api_key, "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen-max", history, system_instruction, prompt
                )
                if res: return res
            except Exception as e:
                logger.error(f"Qwen response generation error: {e}")

        # Llama/Together
        if self.llama_api_key:
            try:
                logger.info("Attempting response generation via Together Llama...")
                res = self._call_openai_compatible_response(
                    self.llama_api_key, "https://api.together.xyz/v1/chat/completions", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", history, system_instruction, prompt
                )
                if res: return res
            except Exception as e:
                logger.error(f"Together Llama response generation error: {e}")

        return None

    def _call_openai_compatible_nlu(self, api_key: str, endpoint: str, model: str, query: str, history: List[Dict[str, Any]], prompt: str) -> Optional[Dict[str, Any]]:
        history_context = ""
        for turn in history[-4:]:
            role = "Customer" if turn["role"] == "user" else "Sales Associate"
            history_context += f"{role}: {turn['content']}\n"
        
        full_prompt = f"CHAT HISTORY:\n{history_context}\nNEW CUSTOMER QUERY:\n\"{query}\"\n{prompt}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise JSON extractor. Analyze the customer query and chat history to extract structured shopping metadata."},
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.0
        }
        res = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"].strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)

    def _call_openai_compatible_response(self, api_key: str, endpoint: str, model: str, history: List[Dict[str, Any]], system_instruction: str, prompt: str) -> Optional[str]:
        messages = [{"role": "system", "content": system_instruction}]
        for turn in history[-6:]:
            role = "user" if turn["role"] == "user" else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        res = requests.post(endpoint, json=payload, headers=headers, timeout=15)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
