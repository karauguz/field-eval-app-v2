# main.py - Tam çalışır auth + radar sistemi
from collections import defaultdict
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Set, List, Tuple
from pathlib import Path
import os
import unicodedata
import urllib.parse
import hashlib
import random

import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, Depends, Cookie
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# --------------------------- Paths / App ---------------------------
BASE_DIR = Path(__file__).parent.resolve()
(BASE_DIR / "static").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "templates").mkdir(parents=True, exist_ok=True)
DATA_DIR = (BASE_DIR / "data").resolve()

templates = Jinja2Templates(directory=str((BASE_DIR / "templates").as_posix()))

# --------------------------- DB ---------------------------
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./field_eval.db")
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)

if DB_URL.startswith("sqlite"):
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DB_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="Saha Gözlem Formu")
app.mount("/static", StaticFiles(directory=str((BASE_DIR / "static").as_posix())), name="static")

# --------------------------- Radar Kategorileri ---------------------------
CATS_ORDER: List[str] = ["Açılış", "İhtiyaç Belirleme", "Analiz", "Sunum", "Kapanış", "Genel"]
CATS_SET = {c.lower() for c in CATS_ORDER}

def _norm(s: Optional[str]) -> str:
    if not s: return ""
    s = str(s).strip().lower()
    s = (s.replace("ı","i").replace("ğ","g").replace("ü","u")
           .replace("ş","s").replace("ö","o").replace("ç","c"))
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def canon_cat(label: str) -> str:
    key = _norm(label)
    
    # Açılış kategorisi
    if key in {"acilis", "açılış", "hazirlik", "ziyaret oncesi", "ziyaret öncesi"}: 
        return "Açılış"
    
    # İhtiyaç Belirleme kategorisi
    if key in {"ihtiyac belirleme", "ihtiyaç belirleme", "ihtiyac", "ihtiyac-belirleme"}:
        return "İhtiyaç Belirleme"
    
    # Analiz kategorisi
    if key in {"analiz", "analysis", "analys", "veri analiz"}:
        return "Analiz"
    
    # Sunum kategorisi
    if key in {"sunum", "presentation", "mesaj"}:
        return "Sunum"
    
    # Kapanış kategorisi  
    if key in {"kapanis", "kapanış", "closing"}:
        return "Kapanış"
    
    # Genel kategorisi
    if key in {"genel", "general"}:
        return "Genel"
    
    # Doğrudan tam eşleşme dene
    for c in CATS_ORDER:
        if _norm(c) == key:
            return c
    
    print(f"Bilinmeyen kategori: '{label}' -> '{key}' -> Genel'e atandı")
    return "Genel"

def parse_category_tags(raw: Optional[str]) -> List[Tuple[str, float]]:
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

    agg: Dict[str, Optional[float]] = {}
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

def sql_date_cast(col_expr: str) -> str:
    backend = engine.url.get_backend_name()
    if backend == "sqlite":
        return f"DATE(substr({col_expr}, 1, 10))"
    else:
        return f"CAST(substr({col_expr},1,10) AS DATE)"

# --------------------------- DB Setup ---------------------------
def init_db():
    backend = engine.url.get_backend_name()
    pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if backend == "sqlite" else "SERIAL PRIMARY KEY"
    with engine.begin() as conn:
        conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS questions (
            id {pk},
            text TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            category TEXT
        );"""))
        conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS evaluations (
            id {pk},
            created_at TEXT NOT NULL,
            manager_name TEXT NOT NULL,
            rep_name TEXT NOT NULL,
            region_name TEXT,
            brick_name TEXT,
            pharmacy_name TEXT,
            total_score REAL NOT NULL,
            max_score REAL NOT NULL,
            percentage REAL NOT NULL,
            notes TEXT
        );"""))
        conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS answers (
            id {pk},
            evaluation_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            answer INTEGER NOT NULL,
            weight REAL NOT NULL,
            score REAL NOT NULL,
            question_text TEXT NOT NULL
        );"""))

def migrate_db():
    backend = engine.url.get_backend_name()
    if backend == "sqlite":
        with engine.begin() as conn:
            cols = [r[1] for r in conn.execute(text("PRAGMA table_info(evaluations)")).fetchall()]
            if "brick_name" not in cols:
                conn.execute(text("ALTER TABLE evaluations ADD COLUMN brick_name TEXT"))
            if "pharmacy_name" not in cols:
                conn.execute(text("ALTER TABLE evaluations ADD COLUMN pharmacy_name TEXT"))
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_eval_rep ON evaluations(rep_name, created_at)"))
    except Exception:
        pass

# --------------------------- CSV utils ---------------------------
def read_csv_guess(path: Path):
    if not path.exists(): 
        return None
    
    # Önce encoding seçenekleriyle dene
    encodings = ['utf-8', 'utf-8-sig', 'cp1254', 'latin1', 'iso-8859-9', 'windows-1254']
    separators = [",", ";", "\\t", "|"]
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding)
                if df.shape[1] == 1 and sep != ",":
                    continue
                # Başarılı okuma durumunda bilgi ver
                print(f"CSV okundu: {path.name} (encoding={encoding}, sep='{sep}')")
                return df
            except Exception:
                continue
    
    # Eski yöntemle tekrar dene (fallback)
    for sep in [",",";","\\t","|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] == 1 and sep != ",":
                continue
            return df
        except Exception:
            continue
    
    # Son çare
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def pick_column(df, candidates, default_idx=0):
    low = [c.lower() for c in df.columns]
    for i,c in enumerate(low):
        if c in candidates: return df.columns[i]
    return df.columns[default_idx]

def load_questions_from_csv(path: Path):
    df = read_csv_guess(path)
    if df is None:
        raise RuntimeError("Questions CSV parse failed")

    cols = [c.lower() for c in df.columns]
    text_col = None; weight_col = None; cat_col = None
    
    print(f"CSV kolonları: {df.columns.tolist()}")  # Debug için
    
    # Soru metni kolonu - "Anket Soruları" eklendi
    for i, c in enumerate(cols):
        if c in ["soru", "question", "text", "soru_metni", "soru metni", 
                 "anket soruları", "anket sorulari", "anket soruları"]:
            text_col = df.columns[i]
            break
    
    # Ağırlık kolonu
    for i, c in enumerate(cols):
        if c in ["katsayi", "katsayı", "weight", "puan", "çarpan", "skor"]:
            weight_col = df.columns[i]
            break
    
    # Kategori kolunu - ÖNCELİKLE "örümcek ağı kategori"yi ara
    for i, c in enumerate(cols):
        if c in ["örümcek ağı kategori", "orumcek agi kategori", "örümcek ağı kategori"]:
            cat_col = df.columns[i]
            break

    # Bulunamazsa diğerlerini dene
    if cat_col is None:
        for i, c in enumerate(cols):
            if c in ["kategori", "category", "spider kategori", "radar kategori"]:
                cat_col = df.columns[i]
                break
    
    # Varsayılan değerler
    if text_col is None: 
        text_col = df.columns[0]
    if weight_col is None and len(df.columns) > 1: 
        weight_col = df.columns[1]
    
    print(f"Kullanılan kolonlar - Text: {text_col}, Weight: {weight_col}, Category: {cat_col}")
    
    records = []
    for _, row in df.iterrows():
        # Soru metni
        raw_t = row.get(text_col) if text_col in df.columns else None
        if raw_t is None or (isinstance(raw_t, float) and pd.isna(raw_t)): 
            continue
        t = str(raw_t).strip()
        if not t or t.lower() in {"nan", "none", "nat"}: 
            continue

        # Ağırlık
        w = 1.0
        if weight_col and weight_col in df.columns:
            w_raw = row.get(weight_col)
            if w_raw is not None and not (isinstance(w_raw, float) and pd.isna(w_raw)):
                try: 
                    w = float(str(w_raw).replace(",", "."))  
                except: 
                    w = 1.0

        # Kategori
        c = None
        if cat_col and cat_col in df.columns:
            cat_raw = row.get(cat_col)
            if cat_raw is not None and not (isinstance(cat_raw, float) and pd.isna(cat_raw)):
                c = str(cat_raw).strip() or None

        print(f"Soru eklendi: {t[:50]}... | Ağırlık: {w} | Kategori: {c}")
        records.append((t, w, c))

    print(f"Toplam {len(records)} soru yüklendi")
    
    # Veritabanına kaydet
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM questions"))
        for t, w, c in records:
            conn.execute(text("INSERT INTO questions(text, weight, category) VALUES (:t,:w,:c)"),
                         {"t": t, "w": w, "c": c})

def init_questions():
    preferred = [
        DATA_DIR / "5_Anket_Sorulari.csv",  # Senin dosyan en üstte
        DATA_DIR / "5_Anket_Soruları - Sheet1.csv",
        DATA_DIR / "5_Anket_Soruları.csv",
    ]
    
    chosen = None
    for p in preferred:
        if p.exists():
            chosen = p
            print(f"CSV seçildi: {p.name}")
            break

    if chosen:
        print(f"CSV yükleniyor: {chosen}")
        load_questions_from_csv(chosen)
    else:
        print("CSV bulunamadı, örnek sorular ekleniyor")
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM questions"))
            for t, w, c in [
                ("Ziyaret öncesi hazırlıklar eksiksiz yapıldı mı?", 1.0, "Açılış"),
                ("Ürün mesajı net ve doğru aktarıldı mı?", 1.5, "Sunum"),
                ("Kapanış ve takip adımı planlandı mı?", 1.2, "Kapanış"),
            ]:
                conn.execute(text("INSERT INTO questions(text, weight, category) VALUES (:t,:w,:c)"),
                             {"t": t, "w": w, "c": c})

# --------------------------- Lookups ---------------------------
def build_lookups():
    managers: Set[str] = set()
    regions: Set[str] = set()
    reps: Set[str] = set()
    bricks: Set[str] = set()
    pharmacies: Set[str] = set()

    manager_to_regions: Dict[str, Set[str]] = {}
    region_to_reps: Dict[str, Set[str]] = {}
    rep_to_bricks: Dict[str, Set[str]] = {}
    brick_to_pharmacies: Dict[str, Set[str]] = {}

    def norm_cell(x):
        s = str(x).strip()
        return s if s and s.lower() not in ["nan","none","null"] else ""

    # Manager-Region
    df1 = read_csv_guess(DATA_DIR / "1_BM_Bolge.csv")
    if df1 is not None and not df1.empty:
        region_col = pick_column(df1, ["bölge","bolge","region"])
        manager_col = pick_column(df1, ["müdür","mudur","manager"], default_idx=1 if df1.shape[1]>1 else 0)
        for _,r in df1.iterrows():
            reg = norm_cell(r.get(region_col))
            mgr = norm_cell(r.get(manager_col))
            if reg: regions.add(reg)
            if mgr:
                managers.add(mgr)
                manager_to_regions.setdefault(mgr, set()).add(reg)

    # Region-Rep
    df2 = read_csv_guess(DATA_DIR / "2_Bolge_TTS.csv")
    if df2 is not None and not df2.empty:
        region_col = pick_column(df2, ["bölge","bolge","region"])
        rep_col = pick_column(df2, ["temsilci","sales rep","rep","çalışan"], default_idx=1 if df2.shape[1]>1 else 0)
        for _,r in df2.iterrows():
            reg = norm_cell(r.get(region_col))
            rep = norm_cell(r.get(rep_col))
            if reg: regions.add(reg)
            if rep: reps.add(rep)
            if reg and rep:
                region_to_reps.setdefault(reg, set()).add(rep)

    # Rep-Brick
    df3 = read_csv_guess(DATA_DIR / "3a_TTS_EczBrick.csv")
    if df3 is not None and not df3.empty:
        rep_col = pick_column(df3, ["temsilci","sales rep","rep","çalışan"])
        brick_col = pick_column(df3, ["brick","eczbrick"], default_idx=1 if df3.shape[1]>1 else 0)
        for _,r in df3.iterrows():
            rep = norm_cell(r.get(rep_col))
            brick = norm_cell(r.get(brick_col))
            if rep: reps.add(rep)
            if brick: bricks.add(brick)
            if rep and brick:
                rep_to_bricks.setdefault(rep, set()).add(brick)

    # Brick-Pharmacy
    df4 = read_csv_guess(DATA_DIR / "4a_Brick_Eczane.csv")
    if df4 is not None and not df4.empty:
        brick_col = pick_column(df4, ["brick","eczbrick"])
        ph_col = pick_column(df4, ["eczane","pharmacy"], default_idx=1 if df4.shape[1]>1 else 0)
        for _,r in df4.iterrows():
            brick = norm_cell(r.get(brick_col))
            ph = norm_cell(r.get(ph_col))
            if brick: bricks.add(brick)
            if ph: pharmacies.add(ph)
            if brick and ph:
                brick_to_pharmacies.setdefault(brick, set()).add(ph)

    return {
        "managers": sorted([x for x in managers if x]),
        "regions": sorted([x for x in regions if x]),
        "reps": sorted([x for x in reps if x]),
        "bricks": sorted([x for x in bricks if x]),
        "pharmacies": sorted([x for x in pharmacies if x]),
        "manager_to_regions": {k: sorted([x for x in v if x]) for k,v in manager_to_regions.items()},
        "region_to_reps": {k: sorted([x for x in v if x]) for k,v in region_to_reps.items()},
        "rep_to_bricks": {k: sorted([x for x in v if x]) for k,v in rep_to_bricks.items()},
        "brick_to_pharmacies": {k: sorted([x for x in v if x]) for k,v in brick_to_pharmacies.items()},
    }

class AuthManager:
    def __init__(self, lookups: Dict):
        self.lookups = lookups
        self.passwords = self._load_passwords()
        self.super_users = {"Hakan Öktem"}  # Süper kullanıcılar
        self.temp_codes: Dict[str, Dict] = {}  # manager_name → {"code": ..., "expires": ...}

    # ------------------- CSV Yükleme -------------------
    def _load_passwords(self) -> Dict[str, str]:
        """CSV'den şifreleri yükle"""
        passwords = {}
        csv_path = DATA_DIR / "6_Mudur_Sifreler.csv"
        df = read_csv_guess(csv_path)
        if df is not None:
            print(f"Şifre CSV kolonları: {df.columns.tolist()}")

            manager_col = pick_column(df, ["manager", "müdür", "mudur", "bölge müdürü"])
            password_col = pick_column(df, ["password", "şifre", "sifre", "pass"], default_idx=1)

            for _, row in df.iterrows():
                mgr = str(row.get(manager_col, "")).strip()
                pwd = str(row.get(password_col, "")).strip()
                if mgr and pwd and mgr != "nan" and pwd != "nan":
                    passwords[mgr] = self._hash_password(pwd)
                    print(f"Şifre yüklendi: {mgr}")

        print(f"Toplam {len(passwords)} müdür şifresi yüklendi")
        return passwords

    # ------------------- Hash -------------------
    def _hash_password(self, password: str) -> str:
        """Şifreyi SHA-256 ile hash'le"""
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    # ------------------- Süper Kullanıcı -------------------
    def is_super_user(self, manager_name: str) -> bool:
        """Süper kullanıcı kontrolü"""
        return manager_name in self.super_users

    # ------------------- Kullanılabilir Müdürler -------------------
    def get_available_managers(self) -> List[str]:
        """Giriş yapabilecek müdürleri döndür"""
        available_managers = list(self.super_users)

        if not self.passwords:
            # Şifre dosyası yoksa tüm müdürleri ekle (süper kullanıcılar hariç)
            for mgr in self.lookups.get("managers", []):
                if mgr not in self.super_users:
                    available_managers.append(mgr)
        else:
            # Şifresi olan müdürleri ekle (süper kullanıcılar hariç)
            for mgr in self.lookups.get("managers", []):
                if mgr in self.passwords and mgr not in self.super_users:
                    available_managers.append(mgr)

        return sorted(set(available_managers))

    # ------------------- Giriş Kontrol -------------------
    def authenticate_manager(self, manager_name: str, password: str = None) -> bool:
        """Müdür adı ve şifre kontrolü"""
        # Süper kullanıcı kontrolü
        if self.is_super_user(manager_name):
            if not password:
                return False
            if manager_name in self.passwords:
                hashed_input = self._hash_password(password)
                return hashed_input == self.passwords[manager_name]
            return False

        # Normal kullanıcılar için mevcut sistem
        if not self.passwords:
            return manager_name in self.lookups.get("managers", [])

        if not password:
            return False

        if manager_name not in self.passwords:
            return False

        hashed_input = self._hash_password(password)
        return hashed_input == self.passwords[manager_name]

    # ------------------- Lookup İşlemleri -------------------
    def _empty_lookups(self) -> Dict:
        return {
            "managers": [],
            "regions": [],
            "reps": [],
            "bricks": [],
            "pharmacies": [],
            "manager_to_regions": {},
            "region_to_reps": {},
            "rep_to_bricks": {},
            "brick_to_pharmacies": {},
        }

    def get_user_specific_lookups(self, manager_name: str) -> Dict:
        if not manager_name:
            return self._empty_lookups()

        if self.is_super_user(manager_name):
            return self.lookups.copy()

        manager_regions = self.lookups.get("manager_to_regions", {}).get(manager_name, [])
        region_to_reps = self.lookups.get("region_to_reps", {})
        manager_reps = list(set(rep for region in manager_regions for rep in region_to_reps.get(region, [])))

        rep_to_bricks = self.lookups.get("rep_to_bricks", {})
        manager_bricks = list(set(brick for rep in manager_reps for brick in rep_to_bricks.get(rep, [])))

        brick_to_pharmacies = self.lookups.get("brick_to_pharmacies", {})
        manager_pharmacies = list(
            set(pharmacy for brick in manager_bricks for pharmacy in brick_to_pharmacies.get(brick, []))
        )

        return {
            "managers": [manager_name],
            "regions": sorted(manager_regions),
            "reps": sorted(manager_reps),
            "bricks": sorted(manager_bricks),
            "pharmacies": sorted(manager_pharmacies),
            "manager_to_regions": {manager_name: manager_regions},
            "region_to_reps": {r: region_to_reps.get(r, []) for r in manager_regions},
            "rep_to_bricks": {r: rep_to_bricks.get(r, []) for r in manager_reps},
            "brick_to_pharmacies": {b: brick_to_pharmacies.get(b, []) for b in manager_bricks},
        }

    # ------------------- Şifre Değiştir -------------------
    def change_password(self, manager_name: str, old_password: str, new_password: str) -> bool:
        if not self.authenticate_manager(manager_name, old_password):
            return False
        self.passwords[manager_name] = self._hash_password(new_password)
        return self._save_passwords_to_csv()

    # ------------------- Şifre Sıfırlama -------------------
    def reset_password_request(self, manager_name: str) -> dict:
        """Şifre sıfırlama talebi - geçici kod üret ve gönder"""
        if manager_name not in self.passwords and manager_name not in self.super_users:
            return {"error": "Kullanıcı bulunamadı"}
        
        temp_code = str(random.randint(100000, 999999))
        expiry = datetime.now() + timedelta(minutes=15)
        self.temp_codes[manager_name] = {
            "code": temp_code, 
            "expires": expiry
        }
        
        return {
            "code": temp_code,
            "expires": expiry.strftime("%H:%M"),
            "success": True
        }
    
    def reset_password_with_code(self, manager_name: str, temp_code: str, new_password: str) -> bool:
        """Geçici kod ile şifre sıfırlama"""
        if manager_name not in self.temp_codes:
            return False

        stored = self.temp_codes[manager_name]
        if datetime.now() > stored["expires"]:
            del self.temp_codes[manager_name]
            return False

        if stored["code"] != temp_code:
            return False

        self.passwords[manager_name] = self._hash_password(new_password)
        del self.temp_codes[manager_name]
        return self._save_passwords_to_csv()

    # ------------------- CSV Kaydet -------------------
    def _save_passwords_to_csv(self) -> bool:
        try:
            csv_path = DATA_DIR / "6_Mudur_Sifreler.csv"
            df = read_csv_guess(csv_path)
            if df is None:
                return False

            manager_col = pick_column(df, ["manager", "müdür", "mudur", "bölge müdürü"])
            password_col = pick_column(df, ["password", "şifre", "sifre", "pass"], default_idx=1)

            for idx, row in df.iterrows():
                mgr_name = str(row.get(manager_col, "")).strip()
                if mgr_name in self.passwords:
                    df.loc[idx, password_col] = self.passwords[mgr_name]

            df.to_csv(csv_path, index=False, encoding="utf-8")
            return True
        except Exception as e:
            print(f"CSV kaydetme hatası: {e}")
            return False

# --------------------------- Current User ---------------------------
def get_current_user(session_user: Optional[str] = Cookie(None)) -> str:
    if not session_user:
        print("GET_CURRENT_USER: No session cookie found")
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        decoded_name = urllib.parse.unquote(session_user)
        print(f"GET_CURRENT_USER: Successfully decoded = '{decoded_name}'")
    except Exception as e:
        print(f"GET_CURRENT_USER: Decode error: {e}, using raw = '{session_user}'")
        decoded_name = session_user

    if not decoded_name or not decoded_name.strip():
        print("GET_CURRENT_USER: Empty decoded name")
        raise HTTPException(status_code=401, detail="Invalid session")

    decoded_name = decoded_name.strip()

    if auth_manager.is_super_user(decoded_name):
        print(f"GET_CURRENT_USER: Super user access = '{decoded_name}'")
        return decoded_name

    all_managers = LOOKUPS.get("managers", [])
    if decoded_name not in all_managers:
        print(f"GET_CURRENT_USER: User '{decoded_name}' not in managers: {all_managers}")
        raise HTTPException(status_code=401, detail="Invalid user")

    print(f"GET_CURRENT_USER: Valid user = '{decoded_name}'")
    return decoded_name

# --------------------------- Boot ---------------------------
init_db()
migrate_db()
init_questions()
LOOKUPS = build_lookups()
auth_manager = AuthManager(LOOKUPS)

print(f"Sistem başlatıldı. Müdürler: {LOOKUPS.get('managers', [])}")

# --------------------------- Radar API ---------------------------
def _daterange_defaults(date_from: Optional[str], date_to: Optional[str]) -> Tuple[str, str, int]:
    def _dt(dstr: str) -> date: return date.fromisoformat(dstr)
    if date_from and date_to:
        d1, d2 = _dt(date_from), _dt(date_to)
    else:
        d2 = date.today()
        d1 = d2 - timedelta(days=29)
    days = (d2 - d1).days + 1
    return (d1.isoformat(), d2.isoformat(), days)

def _rep_dominant_region(rep: str, dfrom: str, dto: str) -> Optional[str]:
    with engine.begin() as conn:
        rows = conn.execute(text(f"""
            SELECT COALESCE(region_name,'') AS r, COUNT(*) c
            FROM evaluations
            WHERE rep_name=:rep AND {sql_date_cast('e.created_at')} BETWEEN :d1 AND :d2
            GROUP BY r ORDER BY c DESC
            LIMIT 1
        """), {"rep": rep, "d1": dfrom, "d2": dto}).fetchall()
    if rows and rows[0][0]:
        return rows[0][0]
    return None

def _category_vector_for_query(where_sql: str, params: dict) -> Tuple[List[float], dict]:
    sql = f"""
        SELECT a.answer, a.weight AS w, q.category AS cat
        FROM answers a
        JOIN evaluations e ON e.id = a.evaluation_id
        JOIN questions   q ON q.id = a.question_id
        WHERE {where_sql}
    """
    
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        meta = conn.execute(text(f"""
            SELECT COUNT(DISTINCT e.id) AS cnt, MAX(e.created_at) AS last_at
            FROM evaluations e
            WHERE {where_sql.replace('a.','').replace('q.','')}
        """), params).mappings().first()

    num = defaultdict(float)
    den = defaultdict(float)
    
    for r in rows:
        answer = int(r["answer"])
        w = float(r["w"] or 0.0)
        cats = parse_category_tags(r["cat"])
        for cat, share in cats:
            contrib = w * float(share)
            den[cat] += contrib
            if answer:
                num[cat] += contrib

    vec = []
    for c in CATS_ORDER:
        if den[c] > 0:
            vec.append(round(100.0 * num[c] / den[c], 1))
        else:
            vec.append(0.0)
    
    return vec, {"samples": int(meta["cnt"] or 0), "last_eval_at": meta["last_at"]}

def _calculate_scores(rows):
    """Temel skor hesaplama fonksiyonu"""
    num = defaultdict(float)
    den = defaultdict(float)
    
    for r in rows:
        answer = int(r["answer"])
        w = float(r["w"] or 0.0)
        cats = parse_category_tags(r["cat"])
        
        for cat, share in cats:
            contrib = w * float(share)
            den[cat] += contrib
            if answer:
                num[cat] += contrib
    
    vec = []
    for c in CATS_ORDER:
        if den[c] > 0:
            vec.append(round(100.0 * num[c] / den[c], 1))
        else:
            vec.append(0.0)
    
    return vec

def _calculate_recent_performance(rows, limit_evaluations=7):
    """Son N değerlendirmenin ortalaması"""
    if not rows:
        return [0.0] * len(CATS_ORDER), {"method": "recent", "evaluations_used": 0}
    
    # Evaluation ID'leri için unique created_at'leri al ve son N'ini seç
    eval_dates = sorted(list(set(row["created_at"] for row in rows)), reverse=True)[:limit_evaluations]
    
    # Sadece bu evaluation'ların verilerini kullan
    recent_rows = [row for row in rows if row["created_at"] in eval_dates]
    
    vec = _calculate_scores(recent_rows)
    
    return vec, {
        "method": "recent",
        "evaluations_used": len(eval_dates),
        "date_range": {
            "from": eval_dates[-1][:10] if eval_dates else None,
            "to": eval_dates[0][:10] if eval_dates else None
        }
    }

def _calculate_weighted_performance(rows):
    """Tarih ağırlıklı hesaplama - yeni anketler daha ağır"""
    if not rows:
        return [0.0] * len(CATS_ORDER), {"method": "weighted", "samples": 0}
    
    # En eski ve en yeni tarihleri bul
    dates = [datetime.fromisoformat(row["created_at"][:19]) for row in rows]
    oldest_date = min(dates)
    newest_date = max(dates)
    total_days = (newest_date - oldest_date).days or 1
    
    num = defaultdict(float)
    den = defaultdict(float)
    
    for r in rows:
        answer = int(r["answer"])
        w = float(r["w"] or 0.0)
        row_date = datetime.fromisoformat(r["created_at"][:19])
        
        # Tarih ağırlığı: yeni anketler maksimum %100 daha ağır
        progress = (row_date - oldest_date).days / total_days
        time_weight = 1.0 + progress  # 1.0 - 2.0 arasında
        
        cats = parse_category_tags(r["cat"])
        for cat, share in cats:
            weighted_contribution = w * float(share) * time_weight
            den[cat] += weighted_contribution
            if answer:
                num[cat] += weighted_contribution
    
    vec = []
    for c in CATS_ORDER:
        if den[c] > 0:
            vec.append(round(100.0 * num[c] / den[c], 1))
        else:
            vec.append(0.0)
    
    return vec, {
        "method": "weighted", 
        "samples": len(set(r["created_at"] for r in rows)),
        "time_span_days": total_days
    }

def _calculate_trend_analysis(rows):
    """Gelişim trendi analizi"""
    if not rows:
        return [0.0] * len(CATS_ORDER), {"method": "trend", "insufficient_data": True}
    
    eval_dates = sorted(list(set(row["created_at"] for row in rows)))
    
    if len(eval_dates) < 2:
        # Tek anket varsa sadece mevcut skoru göster
        vec = _calculate_scores(rows)
        return vec, {
            "method": "trend",
            "current_scores": vec,
            "early_scores": vec,
            "improvement": [0.0] * len(CATS_ORDER),
            "insufficient_data": True
        }
    
    # İlk yarı vs son yarı karşılaştırması
    mid_point = len(eval_dates) // 2
    early_dates = eval_dates[:mid_point] if mid_point > 0 else [eval_dates[0]]
    recent_dates = eval_dates[mid_point:]
    
    early_rows = [row for row in rows if row["created_at"] in early_dates]
    recent_rows = [row for row in rows if row["created_at"] in recent_dates]
    
    early_scores = _calculate_scores(early_rows)
    recent_scores = _calculate_scores(recent_rows)
    
    # Gelişim hesaplama
    improvement = []
    for early, recent in zip(early_scores, recent_scores):
        if early > 0:
            change = recent - early  # Basit fark
            improvement.append(round(change, 1))
        else:
            improvement.append(recent)  # Eğer başlangıçta 0 idiyse, şimdiki değer gelişim
    
    return recent_scores, {
        "method": "trend",
        "current_scores": recent_scores,
        "early_scores": early_scores,
        "improvement": improvement,
        "early_period": {"from": early_dates[0][:10], "to": early_dates[-1][:10]},
        "recent_period": {"from": recent_dates[0][:10], "to": recent_dates[-1][:10]},
        "total_evaluations": len(eval_dates)
    }

def _calculate_target_based(rows, targets=None):
    """Hedefe yakınlık bazlı hesaplama"""
    if targets is None:
        # Her kategori için hedef % - isteğe göre ayarlanabilir
        targets = {
            "Açılış": 80,
            "İhtiyaç Belirleme": 80, 
            "Analiz": 80,
            "Sunum": 80,
            "Kapanış": 80,
            "Genel": 80
        }
    
    current_scores = _calculate_scores(rows)
    
    # Hedefe yakınlık yüzdesini hesapla
    target_achievement = []
    target_values = []
    
    for i, cat in enumerate(CATS_ORDER):
        current = current_scores[i]
        target = targets.get(cat, 80)  # Varsayılan hedef %80
        target_values.append(target)
        
        if target > 0:
            achievement = min(100, (current / target) * 100)
            target_achievement.append(round(achievement, 1))
        else:
            target_achievement.append(100.0)
    
    return current_scores, {
        "method": "target",
        "targets": target_values,
        "achievement": target_achievement,
        "average_achievement": round(sum(target_achievement) / len(target_achievement), 1),
        "categories_at_target": sum(1 for a in target_achievement if a >= 100)
    }

def _category_vector_for_query_enhanced(where_sql: str, params: dict, method: str = "traditional") -> Tuple[List[float], dict]:
    """Gelişmiş radar hesaplama - farklı metodlar"""
    sql = f"""
        SELECT a.answer, a.weight AS w, q.category AS cat, e.created_at
        FROM answers a
        JOIN evaluations e ON e.id = a.evaluation_id
        JOIN questions   q ON q.id = a.question_id
        WHERE {where_sql}
        ORDER BY e.created_at DESC
    """
    
    print(f"ENHANCED CATEGORY VECTOR - SQL: {sql}")
    print(f"ENHANCED CATEGORY VECTOR - PARAMS: {params}")
    print(f"ENHANCED CATEGORY VECTOR - METHOD: {method}")
    
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        meta_basic = conn.execute(text(f"""
            SELECT COUNT(DISTINCT e.id) AS cnt, MAX(e.created_at) AS last_at
            FROM evaluations e
            WHERE {where_sql.replace('a.','').replace('q.','')}
        """), params).mappings().first()
    
    print(f"ENHANCED CATEGORY VECTOR - Total rows: {len(rows)}")
    
    # Temel meta bilgileri
    base_meta = {
        "samples": int(meta_basic["cnt"] or 0),
        "last_eval_at": meta_basic["last_at"]
    }
    
    if method == "recent":
        vec, method_meta = _calculate_recent_performance(rows, 7)
    elif method == "weighted":
        vec, method_meta = _calculate_weighted_performance(rows)
    elif method == "trend":
        vec, method_meta = _calculate_trend_analysis(rows)
    elif method == "target":
        vec, method_meta = _calculate_target_based(rows)
    else:  # traditional
        vec, method_meta = _category_vector_for_query(where_sql, params)
        return vec, {**base_meta, "method": "traditional"}
    
    # Meta bilgileri birleştir
    combined_meta = {**base_meta, **method_meta}
    
    print(f"ENHANCED CATEGORY VECTOR - Final vector: {vec}")
    print(f"ENHANCED CATEGORY VECTOR - Meta: {combined_meta}")
    
    return vec, combined_meta

# --------------------------- API ENDPOINTS ---------------------------

@app.get("/api/rep-radar")
def api_rep_radar(rep: str, manager: Optional[str] = None,
                  date_from: Optional[str] = None, date_to: Optional[str] = None,
                  method: str = "traditional", prev: int = 0,
                  current_user: str = Depends(get_current_user)):
    """Radar API - Süper kullanıcı desteği eklendi"""
    if not rep:
        return JSONResponse({"error": "rep required"}, status_code=400)

    d1, d2, span_days = _daterange_defaults(date_from, date_to)
    
    print(f"RADAR API - User: {current_user}, Rep: {rep}, Method: {method}")
    
    where = f"e.rep_name = :rep AND {sql_date_cast('e.created_at')} BETWEEN :d1 AND :d2"
    params = {"rep": rep, "d1": d1, "d2": d2}
    
    # Süper kullanıcı değilse manager kontrolü ekle
    if not auth_manager.is_super_user(current_user):
        if manager:
            where += " AND e.manager_name = :mgr"
            params["mgr"] = manager
        else:
            where += " AND e.manager_name = :current_user"
            params["current_user"] = current_user

    curr_vec, meta = _category_vector_for_query_enhanced(where, params, method)
    
    payload = {
        "labels": CATS_ORDER,
        "data": curr_vec,
        "method": method,
        "range": {"from": d1, "to": d2},
        **meta
    }

    # Önceki dönem karşılaştırması
    if int(prev) == 1 and method in ["traditional", "recent"]:
        prev_to = (date.fromisoformat(d1) - timedelta(days=1)).isoformat()
        prev_from = (date.fromisoformat(prev_to) - timedelta(days=span_days-1)).isoformat()
        
        prev_where = where.replace(":d1", ":pd1").replace(":d2", ":pd2")
        prev_params = {"rep": rep, "pd1": prev_from, "pd2": prev_to}
        
        if not auth_manager.is_super_user(current_user):
            if manager:
                prev_params["mgr"] = manager
            else:
                prev_params["current_user"] = current_user
        
        p_vec, p_meta = _category_vector_for_query_enhanced(prev_where, prev_params, method)
        payload["prev"] = p_vec
        payload["prevRange"] = {"from": prev_from, "to": prev_to}
        payload["prevMeta"] = p_meta

    return JSONResponse(payload)

@app.get("/api/rep-radar-csv")
def api_rep_radar_csv(rep: str, date_from: Optional[str] = None, date_to: Optional[str] = None):
    d1, d2, _ = _daterange_defaults(date_from, date_to)
    where = f"e.rep_name = :rep AND {sql_date_cast('e.created_at')} BETWEEN :d1 AND :d2"
    vec, _ = _category_vector_for_query(where, {"rep": rep, "d1": d1, "d2": d2})
    rows = ["Kategori,Yuzde"] + [f"{cat},{val}" for cat, val in zip(CATS_ORDER, vec)]
    csv_data = "\n".join(rows)
    return HTMLResponse(content=csv_data, media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="radar_{rep}_{d1}_{d2}.csv"'})

@app.get("/api/radar-methods")
def get_radar_methods():
    """Radar hesaplama yöntemlerinin açıklamaları"""
    return JSONResponse({
        "methods": {
            "traditional": {
                "name": "Geleneksel",
                "description": "Tüm geçmiş anketlerin ağırlıklı ortalaması",
                "icon": "📊",
                "best_for": "Genel performans görünümü"
            },
            "recent": {
                "name": "Mevcut Durum",
                "description": "Son 7 anketin ortalaması",
                "icon": "📈",
                "best_for": "Güncel performans değerlendirmesi"
            },
            "weighted": {
                "name": "Tarih Ağırlıklı",
                "description": "Yeni anketler daha ağır sayılır",
                "icon": "⚖️",
                "best_for": "Gelişimi ödüllendiren değerlendirme"
            },
            "trend": {
                "name": "Gelişim Trendi",
                "description": "İlk yarı vs son yarı karşılaştırması",
                "icon": "📉📈",
                "best_for": "İlerleme analizi ve coaching"
            },
            "target": {
                "name": "Hedefe Yakınlık",
                "description": "Belirlenen hedeflere ne kadar yakın",
                "icon": "🎯",
                "best_for": "Hedef odaklı performans ölçümü"
            }
        },
        "default": "recent",
        "recommended": ["recent", "trend"]
    })

@app.get("/lookups")
def get_lookups(current_user: str = Depends(get_current_user)):
    return JSONResponse(auth_manager.get_user_specific_lookups(current_user))

@app.get("/api/pharmacy-info/{pharmacy_name}")
def get_pharmacy_info(pharmacy_name: str, current_user: str = Depends(get_current_user)):
    """Eczane seçildiğinde brick bilgisini döndür"""
    user_lookups = auth_manager.get_user_specific_lookups(current_user)
    
    # Eczanenin hangi brick'te olduğunu bul
    for brick, pharmacies in user_lookups["brick_to_pharmacies"].items():
        if pharmacy_name in pharmacies:
            return JSONResponse({
                "brick_name": brick,
                "success": True
            })
    
    return JSONResponse({
        "error": "Brick bulunamadı",
        "success": False
    }, status_code=404)

# --------------------------- AUTH ROUTES ---------------------------

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse("login.html", {
        "request": request, 
        "error": error,
        "managers": auth_manager.get_available_managers(),
        "has_passwords": bool(auth_manager.passwords)
    })

@app.post("/login")
async def login_submit(request: Request, 
                      manager_name: str = Form(...),
                      password: str = Form(None)):
    
    print(f"LOGIN ATTEMPT: Manager='{manager_name}', Password={'*'*len(password) if password else 'None'}")
    
    # Form validation - boş değerler kontrolü
    if not manager_name or not manager_name.strip():
        print("LOGIN FAILED: Empty manager name")
        return RedirectResponse(url="/login?error=Lütfen bir bölge müdürü seçin", status_code=303)
    
    manager_name = manager_name.strip()
    
    # Şifre sistemi aktifse şifre kontrolü yap
    if auth_manager.passwords:
        if not password or not password.strip():
            print("LOGIN FAILED: Password required but not provided")
            return RedirectResponse(url="/login?error=Şifre gerekli", status_code=303)
        
        password = password.strip()
        
        try:
            if auth_manager.authenticate_manager(manager_name, password):
                print(f"LOGIN SUCCESS: {manager_name}")
                response = RedirectResponse(url="/", status_code=303)
                # URL encoding ile Türkçe karakter desteği
                encoded_name = urllib.parse.quote(manager_name, safe='')
                response.set_cookie("session_user", encoded_name, 
                                  httponly=True, max_age=3600*8, 
                                  samesite="lax", secure=False)
                return response
            else:
                print(f"LOGIN FAILED: Invalid credentials for {manager_name}")
                # Hata mesajını URL encode et
                error_msg = urllib.parse.quote("Geçersiz kullanıcı adı veya şifre")
                return RedirectResponse(url=f"/login?error={error_msg}", status_code=303)
                
        except Exception as e:
            print(f"LOGIN ERROR: Authentication exception: {str(e)}")
            error_msg = urllib.parse.quote("Giriş işlemi sırasında hata oluştu")
            return RedirectResponse(url=f"/login?error={error_msg}", status_code=303)
    
    # Şifre sistemi aktif değilse eski sistem
    else:
        try:
            if auth_manager.authenticate_manager(manager_name):
                print(f"LOGIN SUCCESS (no password system): {manager_name}")
                response = RedirectResponse(url="/", status_code=303)
                # URL encoding ile Türkçe karakter desteği
                encoded_name = urllib.parse.quote(manager_name, safe='')
                response.set_cookie("session_user", encoded_name, 
                                  httponly=True, max_age=3600*8,
                                  samesite="lax", secure=False)
                return response
            else:
                print(f"LOGIN FAILED: Invalid manager {manager_name}")
                error_msg = urllib.parse.quote("Geçersiz müdür")
                return RedirectResponse(url=f"/login?error={error_msg}", status_code=303)
                
        except Exception as e:
            print(f"LOGIN ERROR: Manager authentication exception: {str(e)}")
            error_msg = urllib.parse.quote("Giriş işlemi sırasında hata oluştu")
            return RedirectResponse(url=f"/login?error={error_msg}", status_code=303)

# --- Şifre Yönetimi Route'ları ---
@app.get("/change-password")
def change_password_form(request: Request):
    current_user = request.query_params.get("manager", "")
    return templates.TemplateResponse(
        "change_password.html", 
        {"request": request, "current_user": current_user}
    )

@app.post("/change-password") 
def change_password_post(
    request: Request,
    current_user: str = Form(...),
    old_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...)
):
    # Şifre eşleşme kontrolü
    if new_password != confirm_password:
        error_msg = "Şifreler eşleşmiyor"
        return RedirectResponse(url=f"/change-password?manager={current_user}&error={urllib.parse.quote(error_msg)}", status_code=303)
    
    # Şifre değiştirme
    if not auth_manager.change_password(current_user, old_password, new_password):
        error_msg = "Mevcut şifre yanlış"
        return RedirectResponse(url=f"/change-password?manager={current_user}&error={urllib.parse.quote(error_msg)}", status_code=303)
    
    return RedirectResponse(url="/login?message=Şifre başarıyla değiştirildi", status_code=303)

@app.get("/forgot-password")
def forgot_password_form(request: Request):
    return templates.TemplateResponse(
        "forgot_password.html",
        {
            "request": request, 
            "step": "request", 
            "managers": auth_manager.get_available_managers()
        }
    )

@app.post("/forgot-password-request")
def forgot_password_request_post(request: Request, manager_name: str = Form(...)):
    result = auth_manager.reset_password_request(manager_name)
    
    if "error" in result:
        error_msg = urllib.parse.quote(result["error"])
        return RedirectResponse(url=f"/forgot-password?error={error_msg}", status_code=303)
    
    # Başarılı - kod oluşturuldu
    return templates.TemplateResponse(
        "forgot_password.html",
        {
            "request": request,
            "step": "code", 
            "manager_name": manager_name,
            "temp_code": result["code"],
            "expires": result["expires"]
        }
    )

@app.post("/forgot-password-reset")
def forgot_password_reset_post(
    request: Request,
    manager_name: str = Form(...),
    temp_code: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...)
):
    # Şifre eşleşme kontrolü
    if new_password != confirm_password:
        return templates.TemplateResponse(
            "forgot_password.html",
            {
                "request": request,
                "step": "code",
                "manager_name": manager_name,
                "error": "Şifreler eşleşmiyor"
            }
        )
    
    # Şifre sıfırlama
    if not auth_manager.reset_password_with_code(manager_name, temp_code, new_password):
        return templates.TemplateResponse(
            "forgot_password.html", 
            {
                "request": request,
                "step": "code",
                "manager_name": manager_name,
                "error": "Geçici kod geçersiz veya süresi dolmuş"
            }
        )
    
    return RedirectResponse(url="/login?message=Şifre başarıyla sıfırlandı", status_code=303)

@app.get("/logout")
def logout(request: Request):
    print("LOGOUT: User logging out, redirecting to login page")
    response = RedirectResponse(url="/login?message=Başarıyla çıkış yapıldı", status_code=303)
    response.delete_cookie("session_user", path="/")
    return response

# --------------------------- MAIN PAGES ---------------------------

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request, session_user: Optional[str] = Cookie(None)):
    if not session_user:
        print("FORM PAGE: No session cookie, redirecting to login")
        return RedirectResponse(url="/login", status_code=303)
    
    # URL decode
    try:
        decoded_name = urllib.parse.unquote(session_user)
        print(f"FORM PAGE: Decoded user = '{decoded_name}'")
    except:
        decoded_name = session_user
        print(f"FORM PAGE: Could not decode, using = '{decoded_name}'")
    
    # DÜZELTME: Tüm manager listesinden kontrol et (sadece şifresi olanlar değil)
    all_managers = LOOKUPS.get('managers', [])
    if decoded_name not in all_managers:
        print(f"FORM PAGE: User '{decoded_name}' not in all managers: {all_managers}")
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie("session_user")
        return response
    
    print(f"FORM PAGE: User '{decoded_name}' is valid, showing form")
    
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, text, weight, COALESCE(category,'') AS category FROM questions ORDER BY category, id"
        )).mappings().all()
    
    user_lookups = auth_manager.get_user_specific_lookups(decoded_name)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "questions": rows,
        "current_user": decoded_name,
        "has_passwords": bool(auth_manager.passwords),  # YENİ: Şifre sistemi aktif mi?
        **user_lookups
    })

@app.get("/evaluations", response_class=HTMLResponse)
def evaluations_page(request: Request, 
                    current_user: str = Depends(get_current_user),
                    q: Optional[str] = None,
                    date_from: Optional[str] = None, 
                    date_to: Optional[str] = None):
    """Değerlendirme kayıtları sayfası"""
    
    # Tarih filtresi varsayılanları
    if not date_from and not date_to:
        date_to = date.today().isoformat()
        date_from = (date.today() - timedelta(days=30)).isoformat()
    
    # Base query
    where_conditions = []
    params = {}
    
    # Süper kullanıcı değilse sadece kendi kayıtlarını göster
    if not auth_manager.is_super_user(current_user):
        where_conditions.append("e.manager_name = :current_user")
        params["current_user"] = current_user
    
    # Tarih filtreleri
    if date_from:
        where_conditions.append(f"{sql_date_cast('e.created_at')} >= :date_from")
        params["date_from"] = date_from
    if date_to:
        where_conditions.append(f"{sql_date_cast('e.created_at')} <= :date_to")
        params["date_to"] = date_to
    
    # Metin arama
    if q and q.strip():
        search_term = f"%{q.strip()}%"
        where_conditions.append("""
            (e.manager_name LIKE :search OR 
             e.rep_name LIKE :search OR 
             e.region_name LIKE :search OR 
             e.pharmacy_name LIKE :search)
        """)
        params["search"] = search_term
    
    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
    # Sorguyu çalıştır
    with engine.begin() as conn:
        evaluations = conn.execute(text(f"""
            SELECT e.id, e.created_at, e.manager_name, e.rep_name, 
                   e.region_name, e.brick_name, e.pharmacy_name, 
                   e.total_score, e.max_score, e.percentage, e.notes
            FROM evaluations e
            WHERE {where_clause}
            ORDER BY e.created_at DESC
            LIMIT 500
        """), params).mappings().all()
    
    return templates.TemplateResponse("evaluations.html", {
        "request": request,
        "evaluations": evaluations,
        "current_user": current_user,
        "is_super_user": auth_manager.is_super_user(current_user)
    })

@app.get("/evaluation/{eval_id}", response_class=HTMLResponse)
def evaluation_detail(eval_id: int, request: Request, 
                     current_user: str = Depends(get_current_user)):
    """Değerlendirme detay sayfası"""
    
    with engine.begin() as conn:
        # Evaluation bilgilerini al
        eval_query = """
            SELECT e.id, e.created_at, e.manager_name, e.rep_name, 
                   e.region_name, e.brick_name, e.pharmacy_name,
                   e.total_score, e.max_score, e.percentage, e.notes
            FROM evaluations e 
            WHERE e.id = :eval_id
        """
        
        # Süper kullanıcı değilse sadece kendi kayıtlarını görebilir
        if not auth_manager.is_super_user(current_user):
            eval_query += " AND e.manager_name = :current_user"
        
        evaluation = conn.execute(text(eval_query), {
            "eval_id": eval_id,
            "current_user": current_user
        }).mappings().first()
        
        if not evaluation:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        # Cevapları al
        answers = conn.execute(text("""
            SELECT a.question_id, a.question_text, a.answer, a.weight, a.score
            FROM answers a
            WHERE a.evaluation_id = :eval_id
            ORDER BY a.question_id
        """), {"eval_id": eval_id}).mappings().all()
    
    return templates.TemplateResponse("detail.html", {
        "request": request,
        "eval": evaluation,
        "answers": answers,
        "current_user": current_user
    })

@app.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request, current_user: str = Depends(get_current_user)):
    
    if auth_manager.is_super_user(current_user):
        # Süper kullanıcıysa tüm verileri göster
        user_lookups = LOOKUPS.copy()
        mgr_to_reps = {}
        
        # DÜZELTME: Tüm müdürler için temsilcileri hesapla
        region_to_reps = LOOKUPS.get("region_to_reps", {})
        
        for mgr in LOOKUPS.get("managers", []):
            mgr_regions = LOOKUPS.get("manager_to_regions", {}).get(mgr, [])
            mgr_reps = []
            for region in mgr_regions:
                mgr_reps.extend(region_to_reps.get(region, []))
            mgr_to_reps[mgr] = sorted(list(set(mgr_reps)))
        
        # Süper kullanıcının kendisini de ekle ve tüm temsilcileri ver
        all_reps = []
        for reps_list in region_to_reps.values():
            all_reps.extend(reps_list)
        mgr_to_reps[current_user] = sorted(list(set(all_reps)))
        
        print(f"Süper kullanıcı {current_user} - mgr_to_reps: {mgr_to_reps}")
        
    else:
        # Normal kullanıcı
        user_lookups = auth_manager.get_user_specific_lookups(current_user)
        mgr_to_reps = {current_user: user_lookups["reps"]}
    
    return templates.TemplateResponse("reports.html", {
        "request": request,
        "current_user": current_user,
        "current_manager": current_user,
        "managers": user_lookups["managers"],
        "mgr_to_reps": mgr_to_reps,
        "radar_order": CATS_ORDER,
        "is_super_user": auth_manager.is_super_user(current_user)
    })

# --------------------------- FORM SUBMISSION ---------------------------

@app.post("/submit")
async def submit_form(
    request: Request,
    current_user: str = Depends(get_current_user),
    rep_name: str = Form(...),
    region_name: Optional[str] = Form(None),
    brick_name: Optional[str] = Form(None),
    pharmacy_name: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    form = await request.form()
    with engine.begin() as conn:
        qs = conn.execute(text("SELECT id, text, weight FROM questions ORDER BY id")).mappings().all()
    
    for q in qs:
        answer_field = f"answer_{q['id']}"
        if form.get(answer_field) is None:
            return RedirectResponse(url="/", status_code=303)

    total = 0.0
    max_total = 0.0
    answers_payload = []
    
    for q in qs:
        qid = q["id"]
        w = float(q["weight"])
        answer_field = f"answer_{qid}"
        ans = 1 if form.get(answer_field) == "1" else 0
        score = ans * w
        total += score
        max_total += w
        
        answers_payload.append({
            "question_id": qid, 
            "question_text": q["text"], 
            "answer": ans, 
            "weight": w, 
            "score": score
        })

    pct = (total / max_total * 100.0) if max_total > 0 else 0.0
    
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO evaluations(created_at, manager_name, rep_name, region_name, brick_name, pharmacy_name,
                                        total_score, max_score, percentage, notes)
                VALUES (:created_at,:manager,:rep,:region,:brick,:pharmacy,:total,:max,:pct,:notes)
            """), {
                "created_at": datetime.utcnow().isoformat(timespec="seconds"),
                "manager": current_user,
                "rep": rep_name.strip(),
                "region": (region_name or "").strip() or None,
                "brick": (brick_name or "").strip() or None,
                "pharmacy": (pharmacy_name or "").strip() or None,
                "total": total, 
                "max": max_total, 
                "pct": pct, 
                "notes": (notes or "").strip() or None,
            })
            
            eval_id = result.lastrowid
            
            for a in answers_payload:
                conn.execute(text("""
                    INSERT INTO answers(evaluation_id, question_id, answer, weight, score, question_text)
                    VALUES (:e,:q,:ans,:w,:s,:qt)
                """), {
                    "e": eval_id, 
                    "q": a["question_id"], 
                    "ans": a["answer"], 
                    "w": a["weight"], 
                    "s": a["score"], 
                    "qt": a["question_text"]
                })
        
        return RedirectResponse(url=f"/evaluation/{eval_id}", status_code=303)
        
    except Exception as e:
        print(f"Submit hatası: {str(e)}")
        return RedirectResponse(url="/", status_code=303)

# --------------------------- DEBUG ENDPOINTS ---------------------------

@app.get("/debug/questions-detail")
def debug_questions_detail():
    """Soruların kategori detaylarını göster"""
    with engine.begin() as conn:
        questions = conn.execute(text("""
            SELECT id, text, weight, category 
            FROM questions 
            ORDER BY id
        """)).mappings().all()
        
        parsed_questions = []
        for q in questions:
            cats = parse_category_tags(q["category"])
            parsed_questions.append({
                "id": q["id"],
                "text": q["text"][:50] + "...",
                "weight": q["weight"],
                "raw_category": q["category"],
                "parsed_categories": cats
            })
        
        return JSONResponse({
            "questions_count": len(questions),
            "questions": parsed_questions,
            "expected_categories": CATS_ORDER
        })

@app.get("/debug/check")
def debug_check():
    with engine.begin() as conn:
        evals = conn.execute(text("""
            SELECT id, rep_name, manager_name, created_at, total_score
            FROM evaluations 
            ORDER BY id DESC
            LIMIT 10
        """)).mappings().all()
        
        questions = conn.execute(text("""
            SELECT id, text, weight, category 
            FROM questions 
            ORDER BY id
            LIMIT 5
        """)).mappings().all()
        
        return JSONResponse({
            "evaluations_count": len(evals),
            "recent_evaluations": [dict(e) for e in evals],
            "questions_count": len(questions),  
            "sample_questions": [dict(q) for q in questions],
            "available_managers": LOOKUPS.get("managers", [])
        })

# --------------------------- MAIN ENTRY POINT ---------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)