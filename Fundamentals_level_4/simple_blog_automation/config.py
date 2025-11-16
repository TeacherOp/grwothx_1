"""
Configuration Management for SEO Blog Automation

This module centralizes all configuration for the Flask application,
making it easier for students to understand environment setup and app configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class"""

    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    BUCKET_ENDPOINT = os.getenv("BUCKET_ENDPOINT")
    BUCKET_REGION = os.getenv("BUCKET_REGION")

    # Flask Configuration
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
    PORT = int(os.getenv("FLASK_PORT", 5000))  # Standard Flask port
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    HOST = "0.0.0.0"

    # Directory Configuration
    BASE_DIR = Path(__file__).parent

    # Data directories
    DATA_DIR = BASE_DIR / 'data'
    IMAGES_DIR = DATA_DIR / 'generated_images'
    BLOGS_DIR = DATA_DIR / 'generated_blogs'
    CSV_FILE = DATA_DIR / 'blog_data.csv'

    # Brand context file
    BRAND_CONTEXT_FILE = BASE_DIR / 'brand_context.txt'

    # Claude Model Configuration
    MODEL_NAME = "claude-sonnet-4-5-20250929"

    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        missing = []

        if not cls.ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY")

        # Supabase is optional but warn if partially configured
        supabase_keys = [cls.SUPABASE_URL, cls.SUPABASE_SERVICE_KEY]
        if any(supabase_keys) and not all(supabase_keys):
            missing.append("Incomplete Supabase configuration (need both URL and SERVICE_KEY)")

        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cls.BLOGS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_brand_context(cls):
        """Load brand context from file"""
        if cls.BRAND_CONTEXT_FILE.exists():
            return cls.BRAND_CONTEXT_FILE.read_text()
        return ""