from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import nltk
import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
import os

# Import the blueprint from route.py
from app.nlp_routes import nlp_bp  # Corrected import statement

# Load environment variables
load_dotenv()

# Hugging Face authentication
hf_token = os.getenv('HUGGINGFACE_API_KEY')
api = HfApi()
if hf_token:
    api.set_access_token(hf_token)
else:
    logging.warning("HUGGINGFACE_API_KEY not found in environment variables")

app = Flask(__name__)
CORS(app)

# Set up rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Register the grammar blueprint with a URL prefix (e.g., /grammar)
app.register_blueprint(nlp_bp, url_prefix='/')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
