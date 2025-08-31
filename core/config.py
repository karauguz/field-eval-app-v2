 
# core/config.py
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).parent.parent.resolve()
(BASE_DIR / "static").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "templates").mkdir(parents=True, exist_ok=True)
DATA_DIR = (BASE_DIR / "data").resolve()

CATS_ORDER: List[str] = ["Açılış", "İhtiyaç Belirleme", "Analiz", "Sunum", "Kapanış", "Genel"]
CATS_SET = {c.lower() for c in CATS_ORDER}