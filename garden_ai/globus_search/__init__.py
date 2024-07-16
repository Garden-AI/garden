import os
from pathlib import Path
from dotenv import load_dotenv

dotenv_path = Path(f"{__file__}/../..").resolve() / ".env"
load_dotenv(dotenv_path=dotenv_path)
_IS_DISABLED = bool(os.getenv("GARDEN_DISABLE_SEARCH_INDEX"))
