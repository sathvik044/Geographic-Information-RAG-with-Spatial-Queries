"""
Configuration settings for the Geographic Information RAG System.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample_data"
GEOGRAPHIC_DATA_DIR = DATA_DIR / "geographic_data"
SATELLITE_IMAGES_DIR = DATA_DIR / "satellite_images"

# Database settings
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Logging settings
LOG_DIR = BASE_DIR / "logs"
LOG_LEVEL = "INFO"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CRS = "EPSG:4326"  # WGS84

# Application settings
STREAMLIT_PORT = 8501
STREAMLIT_HOST = "0.0.0.0"

# Query settings
DEFAULT_N_RESULTS = 5
MAX_N_RESULTS = 20
DEFAULT_RADIUS_KM = 100

# Satellite analysis settings
SUPPORTED_IMAGE_FORMATS = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
MAX_IMAGE_SIZE_MB = 100

# Spatial indexing settings
SPATIAL_INDEX_BUFFER_DISTANCE = 0.01  # ~1km in degrees
DEFAULT_CLUSTERING_DISTANCE = 0.01

# Performance settings
CACHE_TTL = 3600  # 1 hour
MAX_CONCURRENT_QUERIES = 10

# Create directories if they don't exist
def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        DATA_DIR,
        SAMPLE_DATA_DIR,
        GEOGRAPHIC_DATA_DIR,
        SATELLITE_IMAGES_DIR,
        CHROMA_DB_DIR,
        LOG_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Environment variables
def get_env_var(key, default=None):
    """Get environment variable with fallback to default."""
    return os.getenv(key, default)

# Override settings with environment variables
EMBEDDING_MODEL = get_env_var("GEOGRAPHIC_EMBEDDING_MODEL", EMBEDDING_MODEL)
STREAMLIT_PORT = int(get_env_var("STREAMLIT_PORT", STREAMLIT_PORT))
STREAMLIT_HOST = get_env_var("STREAMLIT_HOST", STREAMLIT_HOST)
LOG_LEVEL = get_env_var("LOG_LEVEL", LOG_LEVEL) 