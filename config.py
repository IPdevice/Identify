"""Centralized configuration for API keys and endpoints.

Store sensitive tokens in environment variables instead of hard-coding them in source files.

Expected environment variables:
    API_KEY_OPENAI
    API_KEY_DEEPSEEK
    BASE_URL_OPENAI (optional)
    BASE_URL_DEEPSEEK (optional)
"""

import os

API_KEY_OPENAI: str = os.getenv("API_KEY_OPENAI", "")

# Default endpoints can be overridden via environment
BASE_URL_OPENAI: str = os.getenv("BASE_URL_OPENAI", "")
