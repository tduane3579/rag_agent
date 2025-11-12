import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    return {
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
        'pdf_path': 'data/documents/all_ocr.pdf',
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'collection_name': 'elliott_wave_kb'
    }
