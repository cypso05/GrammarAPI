"""
NLP Processing Module with FLAN-T5 Integration

Features:
- FLAN-T5 based summarization and paraphrasing
- Domain-aware processing
- Production-grade error handling
- Dual processing pipelines
"""

import asyncio
import logging
import os
from flask import Blueprint, jsonify, request
from functools import lru_cache
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from typing import Dict, List
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class FLANProcessor:
    def __init__(self, model_size: str = "small", domain: str = "general"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = f"google/flan-t5-{model_size}"
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            logger.info(f"Loaded {self.model_name} on {self.device}")
            
            # Initialize pipelines
            self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
            self.paraphraser = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
            
            # Domain configuration
            self.domain = domain
            self.domain_terms = self._load_domain_terms(domain)
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError("Model loading failed")

    def _load_domain_terms(self, domain: str) -> Dict:
        return {}

    def _prepare_prompt(self, text: str, task: str) -> str:
        base_prompts = {
            "summarize": "Summarize this {domain} text: {text}",
            "paraphrase": "Rephrase this {domain} text using different words while preserving meaning: {text}",
            "simplify": "Simplify this {domain} text for a general audience: {text}"
        }
        return base_prompts[task].format(domain=self.domain, text=text)

    async def summarize(self, text: str, **kwargs) -> str:
        return await self._process_text(text=text, task="summarize", default_params={"max_length": 200, "min_length": 60,"temperature": 0.9, "num_beams": 3, "do_sample":True, "repetition_penalty": 4.0}, **kwargs)
    
    async def paraphrase(self, text: str, **kwargs) -> str:
        return await self._process_text(text=text, task="paraphrase", default_params={"max_length": 200, "min_length": 60,"temperature": 0.9, "num_beams": 3, "do_sample":True, "repetition_penalty": 6.0}, **kwargs)

    async def _process_text(self, text: str, task: str, default_params: Dict, **kwargs) -> str:
        try:
            prompt = self._prepare_prompt(text, task)
            params = {**default_params, **kwargs}
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            outputs = self.model.generate(**inputs, **params)
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return text

    @lru_cache(maxsize=1000)
    async def batch_process(self, texts: List[str], task: str) -> List[str]:
        return await asyncio.gather(*[self._process_text(text, task) for text in texts])

    def health_check(self) -> Dict:
        return {
            "model": self.model_name,
            "device": str(self.device),
            "domain": self.domain,
            "cache_size": self.batch_process.cache_info().currsize
        }

    def analyze_text(self, text: str) -> Dict:
        """Analyze the text and return word count, sentence count, and key phrases."""
        from collections import Counter
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        nltk.download('punkt')
        nltk.download('stopwords')

        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        word_count = len(words)
        sentence_count = len(sentences)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        word_freq = Counter(filtered_words)
        most_common_words = word_freq.most_common(10)

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "most_common_words": most_common_words
        }

