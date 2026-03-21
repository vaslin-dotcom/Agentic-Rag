import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# LANGSMITH
os.environ["LANGCHAIN_API_KEY"]      = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]   = "true"
os.environ["LANGCHAIN_PROJECT"]      = "Agentic-RAG-Mahabharata"

# PRIMARY — NVIDIA (40 RPM, no RPD limit)
NVIDIA_BASE_URL        = "https://integrate.api.nvidia.com/v1"
NVIDIA_THINK_MODEL = "moonshotai/kimi-k2-instruct-0905"  # 0.83s, best query quality
NVIDIA_GEN_MODEL   = "moonshotai/kimi-k2-instruct"
CHAT_MODEL = "llama-3.3-70b-versatile"

# FALLBACK — Groq (30 RPM, 1K RPD — use when NVIDIA hits 40 RPM)
GROQ_BASE_URL          = "https://api.groq.com/openai/v1"
THINK_MODEL            = "llama-3.3-70b-versatile"
THINK_MODEL_ALT        = "moonshotai/kimi-k2-instruct-0905"
GENERATION_MODEL       = "meta/llama-4-scout-17b-16e-instruct"
GENERATION_MODEL_ALT   = "openai/gpt-oss-120b"