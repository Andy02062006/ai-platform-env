import sys
import os
from pathlib import Path

# Ensure the server directory is in the path for imports
ROOT_DIR = Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from server.inference import main
except ImportError:
    # Fallback for environments where server is already in path
    from inference import main

if __name__ == "__main__":
    main()
