"""Location of the large frozen-feature caches (configs/paths.yaml `cache_root`, default outputs/shared_cache).
Moved to the 8 TB drive on 2026-10-16 (user: the main SSD fills up and the system hangs)."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
_cfg = yaml.safe_load(open(REPO / "configs/paths.yaml"))
CACHE = Path(_cfg.get("cache_root", REPO / "outputs/shared_cache"))                 # featsets / extraction caches
COMPACT = Path(_cfg.get("compact_cache_root", CACHE / "compact"))                   # extract_compact.py targets
