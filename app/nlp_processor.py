"""
NLP Processing Module with Dual-Pipeline Architecture

Features:
- Fast pipeline for real-time interactions (low latency)
- Deep pipeline for complex analysis (high accuracy)
- Full functional parity between pipelines
- Domain-specific optimizations
- Production-grade error handling
- Comprehensive monitoring
"""
import re
from flask import Blueprint, jsonify, request
import nltk
import logging
import os
import time
from typing import Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import asyncio
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline  
)
from symspellpy import SymSpell
import language_tool_python

from sklearn.feature_extraction.text import TfidfVectorizer  
from nltk.corpus import wordnet, stopwords
import json
from flask import Flask
from readability import getmeasures
import spacy
import readability
from app.parts_of_speech import (
    ADJECTIVES, ADVERBS, INTERJECTIONS, IRREGULAR_VERBS, PRONOUNS, 
    CONJUNCTIONS, PREPOSITIONS, DETERMINERS, INTERJECTIONS_DICT, 
    ARTICLES, CONTRACTIONS, COMMON_ERRORS, MEDICAL_TERMS, UK_US_SPELLING)

from app.domain_terms import domain_terms  # Custom domain terms
import sentencepiece
from huggingface_hub import HfApi, hf_hub_download
from collections import defaultdict
from dotenv import load_dotenv
import torch
from symspellpy.symspellpy import SymSpell

# Convert the list to a dictionary
domain_terms_dict = {term: True for term in domain_terms}

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler('nlp_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face authentication
hf_token = os.getenv('HUGGINGFACE_API_KEY')
if hf_token:
    api = HfApi()
    api.set_access_token(hf_token)
else:
    logger.warning("HUGGINGFACE_API_KEY not found in environment variables")

# Initialize Flask app
app = Flask(__name__)


# Global configuration and rules
CONFIG = {
    "max_text_length": 5000,
    "batch_size": 32,
    "cache_size": 10000,
    "timeout": 30.0,
    "gpu_fallback": True,
    "medical_rules": {
        r"\b(he|she)\s(patient)\b": "the patient",
        r"\b(\d+)(mmHg)\b": r"\1 mmHg",
        r"\b(sever)\b": "severe"
    }
}

SPECIAL_TERMS = ["insulin", "NLP", "AI", "ABASAGLAR", "glargine", "mellitus"]

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = os.path.join(os.path.dirname(__file__), 'frequency_dictionary_en_82_765.txt')
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Initialize language_tool_python for grammar checking
language_tool = language_tool_python.LanguageTool('en-US')

# Initialize T5 model and tokenizer
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)





class NLPProcessor:
    def __init__(self, domain: str = "general"):
        self.CONFIG = CONFIG  # Global configuration (assumed defined elsewhere)
        self.start_time = time.time()
        # Initialize SymSpell, language_tool, T5 etc. (omitted for brevity)
        # ...
        # Load spaCy model for tokenization
        self.nlp = spacy.load("en_core_web_sm")
        # Load Google FLAN-T5 components
        self.flan_t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.flan_t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flan_t5_model.to(self.device)
        
        # Initialize spelling components
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = os.path.join(os.path.dirname(__file__), 'frequency_dictionary_en_82_765.txt')
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        
        # Add domain terms to the spelling dictionary
        for term in domain_terms:
            self.sym_spell.create_dictionary_entry(term, 100000)

        # Initialize language tool
        self.language_tool = language_tool_python.LanguageTool("en-US")
        
        # Initialize domain-specific components
        self.domain = domain
        self.domain_terms = domain_terms
        self.domain_keywords = MEDICAL_TERMS if domain == 'medical' else None
        

        # Initialize grammar correction model
        self.grammar_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
        self.grammar_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
        self.grammar_model.to(self.device)



    # Synonym and Antonym Functions with Example Generation
    @lru_cache(maxsize=1000)
    def get_synonyms(self, word: str, mode: str = 'fast') -> Dict:
        """Enhanced synonym lookup with example sentences."""
        syns = set()
        # Get synonyms from WordNet.
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syns.add(lemma.name().replace('_', ' '))
        
        # Incorporate domain-specific synonyms.
        if word in self.domain_terms:
            syns.update(self.domain_terms[word].get('synonyms', []))
        
        # Generate example sentences for the first five synonyms using FLAN-T5.
        examples = {syn: self._generate_example_sentence(syn, "general") for syn in sorted(syns)[:5]}
        return examples

    @lru_cache(maxsize=1000)
    def get_antonyms(self, word: str, mode: str = 'fast') -> Dict:
        """Enhanced antonym lookup with example sentences."""
        ants = set()
        # Get antonyms from WordNet.
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    ants.add(lemma.antonyms()[0].name().replace('_', ' '))
        
        # Incorporate domain-specific antonyms.
        if word in self.domain_terms:
            ants.update(self.domain_terms[word].get('antonyms', []))
        
        # Generate example sentences for the first five antonyms using FLAN-T5.
        examples = {ant: self._generate_example_sentence(ant, "general") for ant in sorted(ants)[:5]}
        return examples 
    
    def _generate_example_sentence(self, word: str, context: str) -> str:
        """
        Use Google FLAN-T5 to generate a natural example sentence that uses the given word.
        The sentence should illustrate the meaning of the word in everyday language.
        """
        prompt = (
        f"Generate a natural, context-rich example sentence that uses the word '{word}' appropriately in everyday conversation. "
        f"The sentence should clearly illustrate the meaning of '{word}' and be grammatically correct. Here are some examples:\n"
        f"1. 'Eloquent': 'The speaker's eloquent speech captivated the audience with its clarity and persuasiveness.'\n"
        f"2. 'Resilient': 'After the storm, the resilient community rebuilt their homes stronger than before.'\n"
        f"Now, write a sentence for the word '{word}':"
    
        )
        inputs = self.flan_t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.flan_t5_model.generate(
            inputs,
            max_length=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            num_return_sequences=1
        )
        sentence = self.flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return sentence

        
    def _initialize_language_resources(self):
            """Initialize language-related resources"""
            self.irregular_verbs = IRREGULAR_VERBS
            self.pronouns = PRONOUNS
            self.prepositions = PREPOSITIONS
            self.articles = ARTICLES
            self.conjunctions = CONJUNCTIONS
            self.contractions = CONTRACTIONS
            self.common_errors = COMMON_ERRORS
            self.medical_terms = MEDICAL_TERMS
            self.uk_us_spelling = UK_US_SPELLING
            
            # Additional parts of speech
            self.adjectives = ADJECTIVES
            self.adverbs = ADVERBS
            self.interjections = INTERJECTIONS
            self.determiners = DETERMINERS
            self.interjections_dict = INTERJECTIONS_DICT
        
    def _spacy_tokenize(self, text: str) -> List[str]:
        """Tokenize using spaCy."""
        doc = self.nlp(text)
        return [token.text for token in doc]
    

    def _split_and_correct(self, text: str, model_type: str) -> str:
        """Use the specified model (FLAN-T5 or native T5) to correct spacing for a single token."""
        if model_type == "flan":
            tokenizer = self.flan_t5_tokenizer
            model = self.flan_t5_model
        else:
            tokenizer = self.t5_tokenizer
            model = self.t5_model

        prompt = f"Split merged words and correct spacing: \"{text}\""
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        split_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return split_text

    
    def _base_word_splitting(self, text: str, model_type: str) -> str:
        tokens = self._spacy_tokenize(text)
        corrected_tokens = []
        for token in tokens:
            if token.isspace() or all(c in '.,!?;:' for c in token):
                corrected_tokens.append(token)
            else:
                if len(token) > 15 and token.upper() not in SPECIAL_TERMS:
                    corrected = self._split_and_correct(token, model_type)
                    corrected_tokens.append(corrected)
                else:
                    corrected_tokens.append(token)
        result = " ".join(corrected_tokens)
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        logging.info(f"Corrected split words: {result}")
        return result

    def _processing_pipeline(self, text: str, mode: str, model_type: str) -> str:
        logging.info(f"Starting text processing for: {text}")
        text = self._base_spelling(text, model_type)
        text = self._base_word_splitting(text, model_type)
        text = self._base_grammar_check(text, model_type)
        text = self._base_grammar_model(text, model_type)
        text = self._base_punctuation(text, model_type)
        logging.info(f"Processed text: {text}")
        return text
    


    def fast_process_text(self, text: str, model_type: "flan") -> str:
        """End-to-end fast processing pipeline."""
        return self._processing_pipeline(text, mode='fast', model_type=model_type)

    def _split_scientific_term(self, term: str) -> List[str]:
            """Split a term into meaningful scientific words."""
            # Try splitting using domain-specific terms
            for i in range(len(term), 0, -1):
                if term[:i] in self.domain_terms:
                    return [term[:i]] + self._split_scientific_term(term[i:])
            return [term] if term else []

    async def _init_resources(self):
            """Initialize all required resources asynchronously"""
            try:
                # Initialize spelling components
                self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                dictionary_path = os.path.join(os.path.dirname(__file__), 'frequency_dictionary_en_82_765.txt')
                self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

                # Add domain terms to the spelling dictionary
                for term in domain_terms:
                    self.sym_spell.create_dictionary_entry(term, 100000)

                # Initialize language tool
                self.language_tool = language_tool_python.LanguageTool("en-US")
                nltk.download("wordnet", quiet=True)

                # Load ML models asynchronously
                await self._load_models()

                # Initialize irregular verbs, pronouns, prepositions, articles, contractions, common errors, medical terms, and UK/US spelling
                self._initialize_language_resources()

                logger.info("Resource initialization completed")

            except Exception as e:
                logger.error(f"Initialization failed: {str(e)}")
                raise RuntimeError("NLP processor initialization failed") 

    async def _load_models(self):
        """Load ML models asynchronously with error handling and fallback"""
        try:
            # FLAN-T5 components
            self.flan_t5_tokenizer = await asyncio.to_thread(T5Tokenizer.from_pretrained, "google/flan-t5-small")
            self.flan_t5_model = await asyncio.to_thread(T5ForConditionalGeneration.from_pretrained, "google/flan-t5-small")

            # Native T5 components for generation (used in example sentence generation)
            self.t5_tokenizer = await asyncio.to_thread(T5Tokenizer.from_pretrained, "t5-small")
            self.t5_model = await asyncio.to_thread(T5ForConditionalGeneration.from_pretrained, "t5-small")

            # Grammar correction model
            self.grammar_tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, "prithivida/grammar_error_correcter_v1")
            self.grammar_model = await asyncio.to_thread(AutoModelForSeq2SeqLM.from_pretrained, "prithivida/grammar_error_correcter_v1")

            logger.info("ML models loaded successfully")

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            if CONFIG['gpu_fallback']:
                logger.warning("Falling back to CPU-only mode")
                await self._load_models_cpu()
            else:
                raise

                    
    async def _load_models_cpu(self):
            """Fallback model loading for CPU-only mode"""
            # Implement CPU fallback loading if necessary
            pass

    def _correct_spelling(self, text: str) -> str:
            """Correct spelling using a language model."""
            tokenizer = self.flan_t5_tokenizer
            model = self.flan_t5_model
            prompt = f"Correct the spelling: \"{text}\""
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected_text
        
            
    
    def _base_grammar_model(self, text: str, mode: str) -> str:
            """Base grammar model correction using a sequence-to-sequence model."""
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            corrected_sentences = []
            logging.info("Starting grammar correction with model.")

            for sentence in sentences:
                inputs = self.grammar_tokenizer("gec: " + sentence, return_tensors="pt", max_length=128, truncation=True)
                outputs = self.grammar_model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
                corrected_sentences.append(self.grammar_tokenizer.decode(outputs[0], skip_special_tokens=True))

            corrected_text = " ".join(corrected_sentences)
            logging.info(f"Corrected grammar with model: {corrected_text}")
            return corrected_text


    def _initialize_language_resources(self):
        """Initialize language-related resources"""
        self.irregular_verbs = {
            "took": "take", "taken": "take", "thought": "think",
            "came": "come", "gone": "go", "went": "go",
            "seen": "see", "saw": "see", "given": "give", "gave": "give",
            "written": "write", "wrote": "write", "spoken": "speak", "spoke": "speak",
            "driven": "drive", "drove": "drive", "eaten": "eat", "ate": "eat",
            "begun": "begin", "began": "begin", "ridden": "ride", "rode": "ride",
            "swum": "swim", "swam": "swim", "flown": "fly", "flew": "fly",
            "forgotten": "forget", "forgot": "forget", "sung": "sing", "sang": "sing"
        }
                
        self.pronouns = {
            "he": "they",
            "she": "they",
            "him": "them",
            "her": "them",
            "his": "their",
            "hers": "theirs",
            "I": "we",
            "me": "us",
            "my": "our",
            "mine": "ours",
            "your": "your",
            "yours": "yours"
        }

        self.prepositions = {
            "on": ["upon", "onto"],
            "in": ["inside", "within"],
            "at": ["near", "by"],
            "by": ["next to", "alongside"],
            "with": ["using", "via"],
            "about": ["regarding", "concerning"],
            "from": ["out of", "since"],
            "to": ["towards", "until"],
            "for": ["on behalf of", "in favor of"]
        }

        

        self.contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "wouldn't": "would not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "it's": "it is",
            "they're": "they are",
            "we're": "we are",
            "you're": "you are"
        }

        self.common_errors = {
            "your": "you're",
            "you're": "your",
            "there": "their",
            "their": "there",
            "they're": "their",
            "affect": "effect",
            "effect": "affect",
            "accept": "except",
            "except": "accept",
            "than": "then",
            "then": "than",
            "loose": "lose",
            "lose": "loose",
            "its": "it's",
            "it's": "its"
        }

        self.medical_terms = {
            "bp": "blood pressure",
            "HR": "heart rate",
            "htn": "hypertension",
            "DM": "diabetes mellitus",
            "MI": "myocardial infarction",
            "CABG": "coronary artery bypass graft",
            "Rx": "prescription",
            "sx": "symptoms",
            "dx": "diagnosis",
            "px": "prognosis",
            "fx": "fracture",
            "mg": "milligrams",
            "mcg": "micrograms",
            "IV": "intravenous",
            "IM": "intramuscular",
            "q.d.": "once daily",
            "b.i.d.": "twice daily",
            "t.i.d.": "three times daily",
            "q.i.d.": "four times daily"
        }

        self.uk_us_spelling = {
            "colour": "color",
            "favourite": "favorite",
            "theatre": "theater",
            "analyse": "analyze",
            "realise": "realize",
            "travelling": "traveling",
            "defence": "defense",
            "licence": "license",
            "programme": "program",
            "cheque": "check",
            "grey": "gray",
            "metre": "meter"
        }

    def deep_process_text(self, text, model_type: str = "flan"):
            # Step 1: Normalize to lowercase
            text = text.lower()

            # Process the text (e.g., handle irregular verbs, pronouns, etc.)
            words = text.split()
            processed_words = []

            for word in words:
                if word in self.pronouns:
                    processed_words.append(self.pronouns[word])
                elif word in self.irregular_verbs:
                    processed_words.append(self.irregular_verbs[word])
                else:
                    processed_words.append(word)

            # Step 2: Capitalize the pronoun "I" correctly
            processed_text = ' '.join(processed_words)
            processed_text = processed_text.replace(' i ', ' I ')
            processed_text = processed_text.replace(' i,', ' I,')
            processed_text = processed_text.replace(' i.', ' I.')

            # Capitalize the first letter of the processed text
            if processed_text:
                processed_text = processed_text[0].upper() + processed_text[1:]

            return self._processing_pipeline(text, mode='deep', model_type=model_type)

    def _warmup_models(self):
        """Warm up models with sample inputs"""
        warmup_samples = [
            "Sample text for model warmup",
            "Another warmup example"
        ]
        try:
            for text in warmup_samples:
                _ = self.fast_process_text(text, model_type='flan')
                _ = self.deep_process_text(text, model_type='flan')
                _ = self.fast_process_text(text, model_type='native')
                _ = self.deep_process_text(text, model_type='native')
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")


 # Helper Functions (integrated from the provided snippet)

    def _has_punctuation(self, word: str) -> bool:
        """Checks if a word has punctuation marks at the beginning or end."""
        punctuation_pattern = r'^[.,!?;:]|[.,!?;:]$'
        result = bool(re.search(punctuation_pattern, word))
        logging.debug(f"_has_punctuation({word}) = {result}")
        return result


    def _clean_word_for_check(self, word: str) -> str:
        """Removes punctuation while preserving the original word structure."""
        cleaned_word = re.sub(r'[^\w\s]', '', word)
        logging.debug(f"_clean_word_for_check({word}) = {cleaned_word}")
        return cleaned_word


    # Base Pipeline Implementations

    def _base_spelling(self, text: str, mode: str) -> str:
        """Base spelling correction implementation."""
        words = re.findall(r'\S+|\s+', text)
        corrected_words = []
        logging.info("Starting spelling correction.")
        
        for word in words:
            if word.isspace():
                corrected_words.append(word)
                continue

            leading_punct = re.match(r'^[^\w\s]+', word)
            trailing_punct = re.search(r'[^\w\s]+$', word)
            clean_word = re.sub(r'[^\w\s]', '', word)

            if clean_word:
                if clean_word.lower() in self.irregular_verbs:
                    corrected = self.irregular_verbs[clean_word.lower()]
                else:
                    suggestions = self.sym_spell.lookup(clean_word, verbosity=2, max_edit_distance=2)
                    corrected = suggestions[0].term if suggestions else clean_word

                if leading_punct:
                    corrected = leading_punct.group() + corrected
                if trailing_punct:
                    corrected = corrected + trailing_punct.group()

                corrected_words.append(corrected)
            else:
                corrected_words.append(word)

        corrected_text = ''.join(corrected_words)
        logging.info(f"Corrected spelling: {corrected_text}")
        return corrected_text


    def _base_grammar_check(self, text: str, mode: str) -> str:
        """Base grammar check implementation using LanguageTool."""
        # Sort matches in reverse order to avoid shifting offsets
        matches = sorted(self.language_tool.check(text), key=lambda x: x.offset, reverse=True)
        for match in matches:
            if match.replacements:
                text = text[:match.offset] + match.replacements[0] + text[match.offset + match.errorLength:]
        logging.info(f"Grammar checked: {text}")
        return text

    

    # Fast Pipeline Functions
    @lru_cache(maxsize=1000)
    def fast_correct_spelling(self, text: str) -> str:
        """Optimized spelling correction"""
        return self._processing_pipeline(text, mode='fast', model_type=model_type)

    def fast_grammar_check(self, text: str) -> str:
        """Rapid grammar correction"""
        return self._processing_pipeline(text, mode='fast', model_type=model_type)

    def fast_correct_grammar_with_model(self, texts: List[str], model_type: str = 'flan') -> List[str]:
        """Model-based grammar correction (fast)"""
        corrected_texts = [self._processing_pipeline(text, mode='fast', model_type=model_type) for text in texts]
        return corrected_texts

    def fast_process_text(self, text: str, model_type: str = "flan") -> str:
        """End-to-end fast processing pipeline."""
        return self._processing_pipeline(text, mode='fast', model_type=model_type)
        
    def fast_split_merged_words(self, texts: List[str], model_type: str = "flan") -> List[str]:
            split_texts = [self._base_word_splitting(text, model_type) for text in texts]
            return split_texts
    

    # Deep Pipeline Functions (Full Parity)
   

    def deep_grammar_check(self, text: str) -> str:
        """Contextual grammar correction"""
        return self._base_grammar_check(text, mode='deep')

    def deep_correct_grammar_with_model(self, text: str) -> str:
        """Enhanced model-based grammar correction"""
        return self._base_grammar_model(text, mode='deep')

    
    def deep_split_merged_words(self, text: str, model_type: str = "flan") -> str:
            """Deep split merged words using the specified model with additional spelling check."""
            split_text = self._split_and_correct(text, model_type)
            corrected_text = self._correct_spelling(split_text)
            return corrected_text
    

    def deep_correct_spelling(self, text: str) -> str:
        """Deep spelling correction using the specified model."""
        return self._correct_spelling(text)
   

    def _correct_spelling(self, text: str) -> str:
            """Correct spelling using a language model."""
            tokenizer = self.flan_t5_tokenizer
            model = self.flan_t5_model
            prompt = f"Correct the spelling: \"{text}\""
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected_text


    def clear_cache(self):
        """Clear processing caches"""
        self.fast_correct_spelling.cache_clear()
        self.get_synonyms.cache_clear()
        self.get_antonyms.cache_clear()
        
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        metrics = {
            "uptime": time.time() - self.start_time,
            "memory_usage": os.getpid(),  # Simplified: process ID as a placeholder
            "cache_status": {
                "spelling": self.fast_correct_spelling.cache_info(),
                "synonyms": self.get_synonyms.cache_info(),
                "antonyms": self.get_antonyms.cache_info()
            },
            "models_loaded": bool(getattr(self, 'flan_t5_model', None) and getattr(self, 't5_model', None))
        }
        return metrics

    def health_check(self) -> Dict:
        """System health status"""
        return {
            "models_loaded": bool(getattr(self, 'flan_t5_model', None) and getattr(self, 't5_model', None)),
            "memory_usage": os.getpid(),  # simplified for illustration
            "uptime": time.time() - self.start_time,
            "cache_status": {
                "spelling": self.fast_correct_spelling.cache_info(),
                "synonyms": self.get_synonyms.cache_info(),
                "antonyms": self.get_antonyms.cache_info()
            }
        }

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=False)
        logging.info("NLP processor shutdown completed")

    def _base_punctuation(self, text: str, mode: str) -> str:
        """Base punctuation correction with heuristic for questions."""
        text = text.strip()
        # If already ends with proper punctuation, return as-is.
        if text.endswith(('.', '?', '!')):
            logging.info(f"Punctuation already present: {text}")
            return text

        # Simple heuristic: if the sentence starts with interrogative words, end with '?'
        interrogatives = ('what', 'how', 'why', 'when', 'where', 'which')
        words = text.split()
        if words and words[0].lower().startswith(interrogatives):
            text = text + "?"
        else:
            text = text + "."
        logging.info(f"Added punctuation: {text}")
        return text

    async def flan_t5_correct(self, text: str) -> str:
        """
        Asynchronously uses FLAN-T5 to rephrase and polish the text and then chains T5 to perform final grammar, punctuation, 
        and style corrections.
        
        FLAN-T5 is instructed to analyze the sentence carefully (splitting merged words, correcting errors, and preserving 
        domain-specific terms) and return a valid JSON object between the markers BEGIN_JSON and END_JSON with exactly two keys:
          - "primary": a string with the best corrected version.
          - "alternatives": a list of up to three alternative suggestions.
          
        The T5 step then uses the FLAN-T5 primary correction as input with a detailed instruction:
          "You are a professional editor. Carefully review the following text. Correct any remaining grammar, punctuation, 
          and style errors, and ensure the text is natural, coherent, and clear. Preserve any domain-specific terms exactly, 
          and do not add extra commentary. Output only the final corrected text."
        
        The final output is returned as a JSON string.
        """
        # ---------- FLAN-T5 Rephrasing Step ----------
        flan_t5_prompt = (
            "Rewrite the following sentence with improved punctuation, grammar, and clarity. "
            "Analyze the sentence carefully as if you were a professional human editor. "
            "Split any merged words, correct errors step by step, and preserve all domain-specific terms exactly. "
            "Return your output as a valid JSON object enclosed between the markers BEGIN_JSON and END_JSON. "
            "The JSON object must have exactly two keys: \"primary\" and \"alternatives\". "
            "\"primary\" should contain the best corrected version of the sentence, and "
            "\"alternatives\" should be a list (with up to three items) of alternative corrections. "
            "Do not output any additional text.\n\n"
            "Use the following exact format:\n"
            "BEGIN_JSON\n"
            "{\n"
            "  \"primary\": \"\",\n"
            "  \"alternatives\": []\n"
            "}\n"
            "END_JSON\n\n"
            f"Input: \"{text}\""
        )

        # Tokenize and generate using FLAN-T5
        flan_t5_inputs = self.flan_t5_tokenizer(flan_t5_prompt, return_tensors="pt")
        flan_t5_outputs = self.flan_t5_model.generate(**flan_t5_inputs, max_length=512)
        flan_t5_result = self.flan_t5_tokenizer.decode(flan_t5_outputs[0], skip_special_tokens=True)

        # Extract JSON object from FLAN-T5 output
        flan_t5_json_str = flan_t5_result.split("BEGIN_JSON")[1].split("END_JSON")[0].strip()
        flan_t5_json = json.loads(flan_t5_json_str)
        primary_correction = flan_t5_json["primary"]

        # ---------- T5 Grammar Correction Step ----------
        t5_prompt = (
            "You are a professional editor. Carefully review the following text. Correct any remaining grammar, punctuation, "
            "and style errors, and ensure the text is natural, coherent, and clear. Preserve any domain-specific terms exactly, "
            "and do not add extra commentary. Output only the final corrected text.\n\n"
            f"Text: \"{primary_correction}\""
        )

        # Tokenize and generate using T5
        t5_inputs = self.t5_tokenizer(t5_prompt, return_tensors="pt")
        t5_outputs = self.t5_model.generate(**t5_inputs, max_length=512)
        t5_result = self.t5_tokenizer.decode(t5_outputs[0], skip_special_tokens=True)

        # Construct final JSON output
        final_output = {
            "primary": t5_result,
            "alternatives": flan_t5_json["alternatives"]
        }

        return json.dumps(final_output)
    
    def _is_meaningful(self, word: str) -> bool:
            """Enhanced meaningful check that properly handles punctuation."""
            clean_word = self._clean_word_for_check(word)
            if not clean_word:
                logging.debug(f"Word '{word}' is not meaningful (cleaned: '{clean_word}')")
                return False

            result = (bool(wordnet.synsets(clean_word)) or 
                    bool(self.sym_spell.lookup(clean_word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)))
            logging.debug(f"Word '{word}' is {'meaningful' if result else 'not meaningful'} (cleaned: '{clean_word}')")
            return result

    def _clean_word_for_check(self, word: str) -> str:
        """Clean a word for meaningful check."""
        return re.sub(r'[^\w\s]', '', word).strip().lower()
    
    #=========function for adding punctuations and sanitizing articles for publishing.

    def _process_with_t5(self, text: str) -> str:
        """
        Use T5 model to process text for punctuation, spelling, and grammar correction.
        The prompt also asks to provide the original words/phrases in brackets for comparison.
        """
        prompt = f"Correct the text for punctuation, spelling, and grammar. Also, provide the original word or phrase in brackets for comparison: \"{text}\""
        inputs = self.t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.t5_model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        corrected_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._format_output_with_brackets(text, corrected_text)

    def generate_alternative_suggestions(self, text: str) -> dict:
        """
        Generate alternative suggestions using the FLAN-T5 model.
        """
        # Use FLAN-T5 for generating alternative suggestions
        paraphrase_prompt = (
            "Rephrase the following text in three different ways while keeping the meaning unchanged and ensuring "
            "proper grammar and punctuation. Text: " + text
        )
        paraphrase_inputs = self.flan_t5_tokenizer.encode(paraphrase_prompt, return_tensors="pt", max_length=1024, truncation=True)
        paraphrase_outputs = self.flan_t5_model.generate(
            paraphrase_inputs,
            max_length=1024,
            do_sample=True,
            top_p=0.9,
            temperature=0.75,
            num_return_sequences=3,
            early_stopping=True
        )
        suggestions = [self._clean_output(self.flan_t5_tokenizer.decode(output, skip_special_tokens=True))
                    for output in paraphrase_outputs]

        # Post-process suggestions to ensure proper capitalization and punctuation
        suggestions = [self._ensure_proper_capitalization_and_punctuation(suggestion) for suggestion in suggestions]

        return {
            "corrected_text": self._process_with_t5(text),
            "original_text": text,
            "suggestions": suggestions
        }

    def _clean_output(self, text: str) -> str:
        """
        Clean the output text by removing unnecessary spaces and correcting common errors.
        """
        return text.strip()

    def normalize_text(self, text: str) -> str:
        """
        Use spaCy to segment text into sentences and then ensure each sentence
        is trimmed and its first character capitalized.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip().capitalize() for sent in doc.sents if sent.text.strip()]
        return " ".join(sentences)

    def add_punctuation(self, text: str) -> dict:
        """
        Correct grammar, punctuation, and style by:
        1. Using the native T5 model with a simplified rewriting prompt.
        2. Normalizing the result with spaCy segmentation.
        3. Generating alternative suggestions using FLAN-T5.
        Returns a dictionary with the original text, the final corrected version,
        and a list of alternative suggestions.
        """
        # --- Step 1: Use native T5 for grammar and punctuation correction ---
        # Use a simpler prompt to avoid echoing the instructions.
        grammar_prompt = (
            "Rewrite the following text so that it is grammatically correct, properly punctuated, "
            "and each sentence starts with a capital letter. Text: " + text
        )
        grammar_inputs = self.t5_tokenizer.encode(grammar_prompt, return_tensors="pt", max_length=1024, truncation=True)
        grammar_outputs = self.t5_model.generate(grammar_inputs, max_length=1024, num_beams=5, early_stopping=True)
        t5_output = self.t5_tokenizer.decode(grammar_outputs[0], skip_special_tokens=True)
        corrected_text = self._clean_output(t5_output)

        # --- Step 2: Normalize using spaCy to enforce sentence segmentation and capitalization ---
        normalized_text = self.normalize_text(corrected_text)
        
        # --- Step 3: Generate alternative suggestions using FLAN-T5 based on normalized text ---
        paraphrase_prompt = (
            "Rephrase the following text in three different ways while keeping the meaning unchanged and ensuring "
            "proper grammar and punctuation. Text: " + normalized_text
        )
        paraphrase_inputs = self.flan_t5_tokenizer.encode(paraphrase_prompt, return_tensors="pt", max_length=1024, truncation=True)
        paraphrase_outputs = self.flan_t5_model.generate(
            paraphrase_inputs,
            max_length=1024,
            do_sample=True,
            top_p=0.9,
            temperature=0.75,
            num_return_sequences=3,
            early_stopping=True
        )
        suggestions = [self._clean_output(self.flan_t5_tokenizer.decode(output, skip_special_tokens=True))
                    for output in paraphrase_outputs]

        return {
            "original_text": text,
            "corrected_text": normalized_text,
            "suggestions": suggestions
        }
