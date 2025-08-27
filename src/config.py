import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(dotenv_path=".env")

# Azure Speech (STT)
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# ElevenLabs TTS
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")


# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")