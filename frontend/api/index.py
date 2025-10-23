"""
Vercel API endpoint - imports the main FastAPI app
"""
import sys
import os

# Add the parent directory to the path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import after path modification (this is intentional and necessary)
from app import app  # noqa: E402

# This is what Vercel will use as the entry point
handler = app