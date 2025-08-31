from pathlib import Path
from sqlalchemy import text
import pandas as pd
from core.config import DATA_DIR, CATS_ORDER
from core.database import engine
from core.categories import parse_category_tags, _norm

def read_csv_guess(path: Path):
    if not path.exists(): return None
    for sep in [",",";","\\t","|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] == 1 and sep != ",":
                continue
            return df
        except Exception:
            continue
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _df_has_radar_tags(df) -> bool:
    possible = {"acilis", "kapan", "sunum", "ihtiyac", "analiz", "genel"}
    col = None
    
    for i, c in enumerate([x.lower() for x in df.columns]):
        if c in ["örümcek ağı kategori", "örümcek agi kategori", "orumcek agi kategori", 
                 "spider kategori", "radar kategori"]:
            col = df.columns[i]
            break
    
    if not col:
        for i, c in enumerate([x.lower() for x in df.columns]):
            if c in ["kategori", "kategoriler", "category", "categories", "bolum", "bölüm", "section"]:
                col = df.columns[i]
                break
    
    if not col: 
        return False
        
    sample = [str(v) for v in df[col].dropna().astype(str).head(20).tolist()]
    has_radar = any(any(tok in _norm(v) for tok in possible) for v in sample)
    return has_radar

def load_questions_from_csv(path: Path):
    df = read_csv_guess(path)
    if df is None:
        raise RuntimeError("Questions CSV parse failed")

    cols = [c.lower() for c in df.columns]
    text_col = None; weight_col = None; cat_col = None
    
    for i, c in enumerate(cols):
        if c in ["soru", "question", "text", "soru_metni", "soru metni", "prompt", 
                 "anket soruları", "anket_sorulari", "anket sorulari"]:
            text_col = df.columns[i]
            break
    
    for i, c in enumerate(cols):
        if c in ["katsayi", "katsayı", "weight", "puan", "çarpan", "carpan", "skor"]:
            weight_col = df.columns[i]
            break
    
    for i, c in enumerate(cols):
        if c in ["örümcek ağı kategori", "örümcek agi kategori", "orumcek agi kategori", 
                 "spider kategori", "radar kategori"]:
            cat_col = df.columns[i]
            break
    
    if cat_col is None:
        for i, c in enumerate(cols):
            if c in ["kategori", "kategoriler", "category", "categories", "bolum", "bölüm", "section"]:
                cat_col = df.columns[i]
                break
    
    if text_col is None: 
        text_col = df.columns[0]
    if weight_col is None and len(df.columns) > 1: 
        weight_col = df.columns[1]
    
    records = []
    for _, row in df.iterrows():
        raw_t = row.get(text_col) if text_col in df.columns else None
        if raw_t is None or (isinstance(raw_t, float) and pd.isna(raw_t)): 
            continue
        t = str(raw_t).strip()
        if not t or t.lower() in {"nan", "none", "nat"}: 
            continue

        w = 1.0
        if weight_col and weight_col in df.columns:
            w_raw = row.get(weight_col)
            if w_raw is not None and not (isinstance(w_raw, float) and pd.isna(w_raw)):
                try: 
                    w = float(str(w_raw).replace(",", "."))  
                except: 
                    w = 1.0

        c = None
        if cat_col and cat_col in df.columns:
            cat_raw = row.get(cat_col)
            if cat_raw is not None and not (isinstance(cat_raw, float) and pd.isna(cat_raw)):
                c = str(cat_raw).strip() or None

        records.append((t, w, c))
    
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM questions"))
        for t, w, c in records:
            conn.execute(text("INSERT INTO questions(text, weight, category) VALUES (:t,:w,:c)"),
                         {"t": t, "w": w, "c": c})

def init_questions():
    preferred = [
        DATA_DIR / "5_Anket_Soruları - Sheet1.csv",
        DATA_DIR / "5_Anket_Sorulari.csv",
        DATA_DIR / "5_Anket_Soruları.csv",
    ]
    
    others = []
    for p in DATA_DIR.glob("*.csv"):
        name = _norm(p.name)
        if "anket" in name and "soru" in name and p not in preferred:
            others.append(p)

    candidates = [p for p in preferred if p.exists()] + others

    chosen = None
    for p in candidates:
        df = read_csv_guess(p)
        if df is not None and _df_has_radar_tags(df):
            chosen = p
            break
    
    if not chosen:
        for p in candidates:
            if p.exists():
                chosen = p
                break

    if chosen:
        load_questions_from_csv(chosen)
    else:
        # Fallback
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM questions"))
            for t, w, c in [
                ("Ziyaret öncesi hazırlıklar eksiksiz yapıldı mı?", 1.0, "Açılış"),
                ("Ürün mesajı net ve doğru aktarıldı mı?", 1.5, "Sunum"),
                ("Kapanış ve takip adımı planlandı mı?", 1.2, "Kapanış"),
            ]:
                conn.execute(text("INSERT INTO questions(text, weight, category) VALUES (:t,:w,:c)"),
                             {"t": t, "w": w, "c": c})

