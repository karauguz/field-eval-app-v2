import unicodedata
from typing import Optional, List, Tuple
from core.config import CATS_ORDER

def _norm(s: Optional[str]) -> str:
    if not s: return ""
    s = str(s).strip().lower()
    s = (s.replace("ı","i").replace("ğ","g").replace("ü","u")
           .replace("ş","s").replace("ö","o").replace("ç","c"))
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def canon_cat(label: str) -> str:
    key = _norm(label)
    # eşanlamlar
    if key in {"acilis","acilış","hazirlik"}: return "Açılış"
    if key in {"ihtiyac","ihtiyac-belirleme","ihtiyac belirleme","ihtiyac tespiti","ihtiyac analizi"}: return "İhtiyaç Belirleme"
    if key in {"analiz","analysis","analys","veri analiz","ims"}: return "Analiz"
    if key in {"sunum","presentation","mesaj","msg"}: return "Sunum"
    if key in {"kapanis","kapanış","closing","kapanis & takip"}: return "Kapanış"
    if key in {"genel","general"}: return "Genel"
    # doğrudan tam eşleşme
    for c in CATS_ORDER:
        if _norm(c) == key:
            return c
    return "Genel"

def parse_category_tags(raw: Optional[str]) -> List[Tuple[str, float]]:
    """
    "Açılış, Sunum"           -> [(Açılış,0.5),(Sunum,0.5)]
    "Açılış:0.7; Sunum:0.3"   -> oranlara göre
    boş/None                  -> [(Genel,1.0)]
    """
    if not raw or str(raw).strip() == "":
        return [("Genel", 1.0)]

    parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
    parsed: List[Tuple[str, Optional[float]]] = []
    total_ratio = 0.0
    has_ratio = False

    for p in parts:
        if ":" in p:
            name, ratio = p.split(":", 1)
            try:
                r = float(str(ratio).replace(",", "."))
                has_ratio = True
                parsed.append((canon_cat(name), max(r, 0.0)))
                total_ratio += max(r, 0.0)
            except:
                parsed.append((canon_cat(name), None))
        else:
            parsed.append((canon_cat(p), None))

    # aynı kategori tekrarlarını birleştir
    agg: dict = {}
    for name, r in parsed:
        if has_ratio:
            agg[name] = float(agg.get(name, 0.0) or 0.0) + (r or 0.0)
        else:
            agg[name] = None

    if has_ratio and total_ratio > 0:
        return [(k, float(v)/total_ratio) for k, v in agg.items() if float(v) > 0]
    else:
        n = max(len(agg), 1)
        share = 1.0 / n
        return [(k, share) for k in agg.keys()]

# ======================================