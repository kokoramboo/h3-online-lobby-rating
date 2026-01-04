import os
from pathlib import Path

def load_env(env_path=".env"):
    """Simple parser for .env files without external dependencies."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Load env variables at module import
load_env()

# Project Identity
PROJECT_NAME = "H3 Online Lobby Rating System"

# Paths
MATCHES_CSV = Path(os.getenv("MATCHES_CSV", "data/matches.csv"))
PLAYERS_CSV = Path(os.getenv("PLAYERS_CSV", "data/players.csv"))
TEMPLATE_GROUPS_JSON = Path(os.getenv("TEMPLATE_GROUPS_JSON", "template_groups.json"))
WORKING_DIR = Path(os.getenv("WORKING_DIR", "data/groups"))
RATINGS_SCRIPT = os.getenv("RATINGS_SCRIPT", "rating_system.py")

# Output Configuration
RATINGS_FILENAME = os.getenv("RATINGS_FILENAME", "ratings.csv")
PRIORS_FILENAME = os.getenv("PRIORS_FILENAME", "priors.csv")
PARAMS_FILENAME = os.getenv("PARAMS_FILENAME", "params.json")
PARAMS_FILENAME = os.getenv("PARAMS_FILENAME", "params.json")

def get_group_dir(group_name):
    """
    Returns the directory for a specific group (e.g., data/groups/JC_Family/).
    Each group directory stores: 
    - matches.csv (filtered)
    - phase1.csv/phase2.csv (priors)
    - params.json (calibrated items)
    - ratings.csv (final)
    """
    d = WORKING_DIR / group_name
    d.mkdir(parents=True, exist_ok=True)
    return d

# Global/Default Params
TEMPLATE_PARAMS_JSON = WORKING_DIR / "template_params.json"

# Ensure global directories exist
if not WORKING_DIR.exists():
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
