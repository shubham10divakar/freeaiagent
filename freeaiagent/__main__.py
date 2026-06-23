"""Enable `python -m freeaiagent ...` (used by the SDK to spawn the server)."""
from .cli import app

if __name__ == "__main__":
    app()
