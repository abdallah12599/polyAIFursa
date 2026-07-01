import os
import sys

# Make `import app` work when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Provide a known, capability-rich model and a dummy key so importing `app`
# never reaches out to a real provider. Construction of the chat model is lazy,
# so no network call happens with these values.
os.environ.setdefault("MODEL", "openai:gpt-5.4-mini")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("YOLO_SERVICE_URL", "http://localhost:8080")
