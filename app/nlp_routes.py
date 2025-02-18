from flask import Blueprint, jsonify, request
from app.nlp_processor import NLPProcessor
from app.nlp_processor import CONFIG
from nltk.corpus import wordnet
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from flask import Flask, request, jsonify
from transformers import pipeline
import spacy
from collections import defaultdict
import os
from dotenv import load_dotenv
import time 
import re
from app.summary import FLANProcessor
from flask import Blueprint, jsonify, request
import asyncio
import logging



# Load environment variables
load_dotenv()

# Create blueprint
nlp_bp = Blueprint('nlp', __name__, url_prefix='/')

# Instantiate the NLP processor with the desired domain (set via the NLP_DOMAIN env variable)
nlp_processor = NLPProcessor()
# Initialize the FLANProcessor
processor = FLANProcessor(model_size="base", domain="general")


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

app = Flask(__name__)
CORS(app)


# routes for summary and parapharase, summary.py
# Route for summarization
@nlp_bp.route('/summarize', methods=['POST'])
async def summarize():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    summary = await processor.summarize(text)
    return jsonify({"summary": summary})

# Route for paraphrasing
@nlp_bp.route('/paraphrase', methods=['POST'])
async def paraphrase():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    paraphrase = await processor.paraphrase(text)
    return jsonify({"paraphrase": paraphrase})


# Combined route for both summarization and paraphrasing
@nlp_bp.route('/process_summary', methods=['POST'])
async def process():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Get optional parameters
    summarize_flag = data.get('summarize', False)
    paraphrase_flag = data.get('paraphrase', False)
    
    response = {}
    
    if summarize_flag:
        response['summary'] = await processor.summarize(text)
    
    if paraphrase_flag:
        response['paraphrase'] = await processor.paraphrase(text)
    
    if not response:
        return jsonify({"error": "No processing option selected"}), 400
    
    return jsonify(response)

# Health check route
@nlp_bp.route('/health_summary', methods=['GET'])
async def health():
    return jsonify(processor.health_check())



# Run async initialization for NLPproceessor routes 
try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(nlp_processor._init_resources())
    nlp_processor._warmup_models()
except Exception as e:
    logging.error(f"Failed to initialize NLP processor: {str(e)}")
    raise RuntimeError("NLP processor initialization failed")


@nlp_bp.route('/process_text', methods=['POST'])
def process_text():
    """Endpoint for text processing with mode selection"""
    data = request.get_json()
    text = data.get('text', '')
    mode = data.get('mode', 'fast')
    model_type = data.get('model_type', 'flan')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > nlp_processor.CONFIG["max_text_length"]:
        return jsonify({"error": "Text exceeds maximum length"}), 400
    
    try:
        if mode == 'fast':
            result = nlp_processor.fast_process_text(text, model_type)
        elif mode == 'deep':
            result = nlp_processor.deep_process_text(text)
        else:
            return jsonify({"error": "Invalid processing mode"}), 400
            
        return jsonify({"processed_text": result}), 200
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return jsonify({"error": "Text processing failed"}), 500

@nlp_bp.route('/fast_correct_spelling', methods=['POST'])
def fast_correct_spelling():
    try:
        data = request.get_json()  # Parse JSON data from the request
        if not isinstance(data, list):
            logger.warning(f"Invalid input: {data}. Expected a list of strings.")
            return jsonify({"error": "Input should be a list of strings"}), 400

        model_type = request.args.get('model_type', 'flan')
        corrected_texts = [nlp_processor.fast_process_text(text, model_type) for text in data]
        return jsonify(corrected_texts)

    except ValueError as e:
        logger.error(f"ValueError in fast_correct_spelling: {e}")
        return jsonify({"error": str(e)}, 400) # include error
    except TypeError as e:
        logger.error(f"TypeError in fast_correct_spelling: {e}")
        return jsonify({"error": str(e)}, 400)  # include error
    except Exception as e:
        logger.exception(f"Unexpected error in fast_correct_spelling")
        return jsonify({"error": "An unexpected error occurred"}), 500



@nlp_bp.route('/synonyms', methods=['GET'])
def get_synonyms_querry():
    """Get synonyms for a word"""
    word = request.args.get('word', '')
    mode = request.args.get('mode', 'fast')
    
    if not word:
        return jsonify({"error": "No word provided"}), 400
    
    try:
        synonyms = nlp_processor.get_synonyms(word, mode)
        return jsonify({"word": word, "synonyms": synonyms}), 200
    except Exception as e:
        logging.error(f"Synonym lookup error: {str(e)}")
        return jsonify({"error": "Synonym lookup failed"}), 500


@nlp_bp.route('/antonyms', methods=['GET'])
def get_antonyms_querry():
    """Get antonyms for a word"""
    word = request.args.get('word', '')
    mode = request.args.get('mode', 'fast')
    
    if not word:
        return jsonify({"error": "No word provided"}), 400
    
    try:
        antonyms = nlp_processor.get_antonyms(word, mode)
        return jsonify({"word": word, "antonyms": antonyms}), 200
    except Exception as e:
        logging.error(f"Antonym lookup error: {str(e)}")
        return jsonify({"error": "Antonym lookup failed"}), 500
    

@nlp_bp.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    try:
        status = nlp_processor.health_check()
        return jsonify(status), 200
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        return jsonify({"error": "Health check failed"}), 500

@nlp_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Performance metrics endpoint"""
    try:
        metrics = nlp_processor.get_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        logging.error(f"Metrics retrieval error: {str(e)}")
        return jsonify({"error": "Metrics retrieval failed"}), 500


@nlp_bp.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Cache clearing endpoint"""
    try:
        nlp_processor.clear_cache()
        return jsonify({"message": "Cache cleared successfully"}), 200
    except Exception as e:
        logging.error(f"Cache clear error: {str(e)}")
        return jsonify({"error": "Cache clearance failed"}), 500

@nlp_bp.route('/fast/split', methods=['POST'])
def fast_split_merged_words():
    try:
        data = request.get_json()
        if not isinstance(data, list):
            raise ValueError("Input should be a list of strings")
        split_texts = nlp_processor.fast_split_merged_words(data)
        return jsonify({'split_texts': split_texts})
    except Exception as e:
        logger.error(f"Error in fast_split_merged_words: {e}")
        return jsonify({'error': str(e)}), 500

@nlp_bp.route('/fast/grammar_model', methods=['POST'])
def fast_correct_grammar_with_model():
    try:
        data = request.get_json()
        texts = data.get('text', [])
        if not isinstance(texts, list):
            raise ValueError("Input should be a list of strings")
        model_type = request.args.get('model_type', 'flan')
        corrected_texts = nlp_processor.fast_correct_grammar_with_model(texts, model_type)
        return jsonify({'corrected_texts': corrected_texts})
    except Exception as e:
        logger.error(f"Error in fast_correct_grammar_with_model: {e}")
        return jsonify({'error': str(e)}), 500


@nlp_bp.route("/deep/spelling", methods=["POST"])  # fix this, change logic
def deep_correct_spelling():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = nlp_processor.deep_correct_spelling(text)
        return jsonify({"result": result}), 200
    except Exception as e:
        logging.error(f"Deep spelling correction error: {str(e)}")
        return jsonify({"error": "Deep spelling correction failed"}), 500

@nlp_bp.route("/deep/split", methods=["POST"]) # not working right
def deep_split_merged_words():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = nlp_processor.deep_split_merged_words(text)
        return jsonify({"result": result}), 200
    except Exception as e:
        logging.error(f"Deep word splitting error: {str(e)}")
        return jsonify({"error": "Deep word splitting failed"}), 500

@nlp_bp.route("/deep/grammar_check", methods=["POST"])  # not working right
def deep_grammar_check():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = nlp_processor.deep_grammar_check(text)
        return jsonify({"result": result}), 200
    except Exception as e:
        logging.error(f"Deep grammar check error: {str(e)}")
        return jsonify({"error": "Deep grammar check failed"}), 500

@nlp_bp.route("/deep/grammar_model", methods=["POST"])
def deep_correct_grammar_with_model():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = nlp_processor.deep_correct_grammar_with_model(text)
        return jsonify({"result": result}), 200
    except Exception as e:
        logging.error(f"Deep grammar model correction error: {str(e)}")
        return jsonify({"error": "Deep grammar model correction failed"}), 500

@nlp_bp.route("/deep/process", methods=["POST"])
def deep_process_text():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = nlp_processor.deep_process_text(text)
        return jsonify({"result": result}), 200
    except Exception as e:
        logging.error(f"Deep text processing error: {str(e)}")
        return jsonify({"error": "Deep text processing failed"}), 500
    
# process synonyms with URL parameter

@nlp_bp.route("/synonyms/<word>", methods=["GET"])
def get_synonym_word (word):
    # Optionally, mode can be provided as a query parameter (default is 'fast')
    mode = request.args.get("mode", "fast")
    try:
        result = nlp_processor.get_synonyms(word, mode=mode)
        return jsonify({"result": result})
    except Exception as e:
        abort(500, description=str(e))


@nlp_bp.route("/antonyms/<word>", methods=["GET"])
def get_antonyms_word(word):
    mode = request.args.get("mode", "fast")
    try:
        result = nlp_processor.get_antonyms(word, mode=mode)
        return jsonify({"result": result})
    except Exception as e:
        abort(500, description=str(e))

@nlp_bp.route("/add/punctuation", methods=["POST"])
def deep_add_punctuation():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = nlp_processor.add_punctuation(text)
    return jsonify(result)



