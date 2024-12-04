from pathlib import Path
from typing import Dict, Any

import yaml

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config