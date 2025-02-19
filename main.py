from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import nltk
import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
import os
import requests
import zipfile

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

# Function to download and extract LanguageTool
def download_and_extract_language_tool(url, extract_to, retries=3):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    zip_path = os.path.join(extract_to, "LanguageTool-stable.zip")
    
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
            
            # Save the ZIP file
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            logging.info("LanguageTool downloaded and extracted successfully.")
            return
        except requests.RequestException as e:
            logging.error(f"Error: Failed to download the file. Attempt {attempt + 1} of {retries}. Exception: {e}")
        except zipfile.BadZipFile as e:
            logging.error(f"Error: Bad ZIP file. Attempt {attempt + 1} of {retries}. Exception: {e}")

# URL for downloading LanguageTool
language_tool_url = "https://languagetool.org/download/LanguageTool-stable.zip"
# Directory where LanguageTool will be extracted
language_tool_dir = os.path.join(os.getcwd(), "language_tool")

# Download and extract LanguageTool
download_and_extract_language_tool(language_tool_url, language_tool_dir)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
