# main.py - Eczane/Doktor anket sistemi
from collections import defaultdict
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Set, List, Tuple
from pathlib import Path
import os
import unicodedata
import urllib.parse
import hashlib
import random
import tempfile
import json

import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, Depends, Cookie
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# PDF kütüphanesi
from reportlab.pdfgen import canvas

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

# --------------------------- Anket Türü ve Kategoriler ---------------------------
SURVEY_TYPES = {
    "eczane": {
        "name": "Eczane Ziyareti",
        "categories": ["Müşteri Odaklılık", "Öğrenme Çevikliği", "Sonuç Çevikliği"],
        "requires_doctor_name": False,
        "segment_field": "pharmacy_name"
    },
    "doktor": {
        "name": "Doktor Ziyareti", 
        "categories": ["Müşteri Odaklılık", "Öğrenme Çevikliği", "Sonuç Çevikliği", "Kişisel Farkındalık", "Değişim Çevikliği"],
        "requires_doctor_name": True,
        "segment_field": "doctor_name"
    }
}

# CSV'den gelen kategoriler (backward compatibility için)
CATS_ORDER: List[str] = ["Müşteri Odaklılık", "Öğrenme Çevikliği", "Sonuç Çevikliği", "Kişisel Farkındalık", "Değişim Çevikliği", "Genel"]
CATS_SET = {c.lower() for c in CATS_ORDER}

def _norm(s: Optional[str]) -> str:
    if not s: return ""
    s = str(s).strip().lower()
    s = (s.replace("ı","i").replace("ğ","g").replace("ü","u")
           .replace("ş","s").replace("ö","o").replace("ç","c"))
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def get_survey_categories(survey_type: str) -> List[str]:
    """Anket türüne göre kategori listesi döndür"""
    return SURVEY_TYPES.get(survey_type, {}).get("categories", CATS_ORDER)

def canon_cat(label: str, survey_type: str = None) -> str:
    """Kategori eşleştirme - CSV'den gelen kategorilere göre"""
    key = _norm(label)
    
    # Ana kategoriler (CSV'den gelenler)
    if key in {"musteri odaklilik", "müşteri odaklılık", "customer focus"}:
        return "Müşteri Odaklılık"
    elif key in {"ogrenme cevikligi", "öğrenme çevikliği", "learning agility"}:
        return "Öğrenme Çevikliği"
    elif key in {"sonuc cevikligi", "sonuç çevikliği", "result agility"}:
        return "Sonuç Çevikliği"
    elif key in {"kisisel farkindalik", "kişisel farkındalık", "personal awareness"}:
        return "Kişisel Farkındalık"
    elif key in {"degisim cevikligi", "değişim çevikliği", "change agility"}:
        return "Değişim Çevikliği"
    elif key in {"genel", "general"}:
        return "Genel"
    
    # Eski sistem için backward compatibility
    elif key in {"acilis", "açılış", "hazirlik", "ziyaret oncesi", "ziyaret öncesi"}: 
        return "Müşteri Odaklılık"  # Açılış -> Müşteri Odaklılık
    elif key in {"ihtiyac belirleme", "ihtiyaç belirleme", "ihtiyac", "ihtiyac-belirleme"}:
        return "Öğrenme Çevikliği"  # İhtiyaç Belirleme -> Öğrenme Çevikliği
    elif key in {"analiz", "analysis", "analys", "veri analiz"}:
        return "Öğrenme Çevikliği"  # Analiz -> Öğrenme Çevikliği
    elif key in {"sunum", "presentation", "mesaj"}:
        return "Sonuç Çevikliği"  # Sunum -> Sonuç Çevikliği
    elif key in {"kapanis", "kapanış", "closing"}:
        return "Sonuç Çevikliği"  # Kapanış -> Sonuç Çevikliği
    
    # Doğrudan tam eşleşme dene
    for c in CATS_ORDER:
        if _norm(c) == key:
            return c
    
    print(f"Bilinmeyen kategori: '{label}' -> '{key}' -> Genel'e atandı")
    return "Genel"

def parse_category_tags(raw: Optional[str], survey_type: str = None) -> List[Tuple[str, float]]:
    """Kategori tag'lerini parse et - virgül ile ayrılan kategoriler"""
    if not raw or str(raw).strip() == "":
        return [("Genel", 1.0)]

    # Virgül ile ayrılmış kategorileri işle
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    
    if len(parts) == 1:
        # Tek kategori
        return [(canon_cat(parts[0], survey_type), 1.0)]
    
    # Çoklu kategori - eşit ağırlık
    share = 1.0 / len(parts)
    result = []
    for p in parts:
        cat = canon_cat(p.strip(), survey_type)
        result.append((cat, share))
    
    return result

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
            category TEXT,
            survey_type TEXT DEFAULT 'eczane'
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
            doctor_name TEXT,
            survey_type TEXT DEFAULT 'eczane',
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
    """Veritabanını yeni sütunlarla güncelle"""
    backend = engine.url.get_backend_name()
    if backend == "sqlite":
        with engine.begin() as conn:
            # Evaluations tablosu için yeni sütunlar
            cols = [r[1] for r in conn.execute(text("PRAGMA table_info(evaluations)")).fetchall()]
            if "doctor_name" not in cols:
                conn.execute(text("ALTER TABLE evaluations ADD COLUMN doctor_name TEXT"))
            if "survey_type" not in cols:
                conn.execute(text("ALTER TABLE evaluations ADD COLUMN survey_type TEXT DEFAULT 'eczane'"))
                
            # Questions tablosu için survey_type sütunu
            q_cols = [r[1] for r in conn.execute(text("PRAGMA table_info(questions)")).fetchall()]
            if "survey_type" not in q_cols:
                conn.execute(text("ALTER TABLE questions ADD COLUMN survey_type TEXT DEFAULT 'eczane'"))
    
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_eval_rep_type ON evaluations(rep_name, survey_type, created_at)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_eval_survey_type ON evaluations(survey_type)"))
    except Exception:
        pass

# --------------------------- CSV utils ---------------------------
def read_csv_guess(path: Path):
    if not path.exists(): 
        return None
    
    encodings = ['utf-8', 'utf-8-sig', 'cp1254', 'latin1', 'iso-8859-9', 'windows-1254']
    separators = [",", ";", "\\t", "|"]
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding)
                if df.shape[1] == 1 and sep != ",":
                    continue
                print(f"CSV okundu: {path.name} (encoding={encoding}, sep='{sep}')")
                return df
            except Exception:
                continue
    
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
    print(f"CSV kolonları: {df.columns.tolist()}")
    
    # Soru metni kolonu
    text_col = None
    for i, c in enumerate(cols):
        if c in ["soru", "question", "text", "soru_metni", "soru metni", 
                 "anket soruları", "anket sorulari"]:
            text_col = df.columns[i]
            break
    
    # Ağırlık kolonu (Skor sütunu)
    weight_col = None
    for i, c in enumerate(cols):
        if c in ["katsayi", "katsayı", "weight", "puan", "çarpan", "skor"]:
            weight_col = df.columns[i]
            break
    
    # Kategori kolonu
    cat_col = None
    for i, c in enumerate(cols):
        if c in ["örümcek ağı kategori", "orumcek agi kategori"]:
            cat_col = df.columns[i]
            break
    
    # Segment kolonu (eczane/doktor)
    segment_col = None
    for i, c in enumerate(cols):
        if c in ["segment", "tip", "type"]:
            segment_col = df.columns[i]
            break
    
    # Varsayılan değerler
    if text_col is None: 
        text_col = df.columns[0]
    if weight_col is None and len(df.columns) > 1: 
        weight_col = df.columns[1]
    
    print(f"Kullanılan kolonlar - Text: {text_col}, Weight: {weight_col}, Category: {cat_col}, Segment: {segment_col}")
    
    # Mevcut soruları temizle
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM questions"))
    
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
                    w_str = str(w_raw).replace(",", ".")
                    w = float(w_str)
                except: 
                    w = 1.0

        # Kategori
        c = None
        if cat_col and cat_col in df.columns:
            cat_raw = row.get(cat_col)
            if cat_raw is not None and not (isinstance(cat_raw, float) and pd.isna(cat_raw)):
                c = str(cat_raw).strip() or None

        # Survey type (segment)
        survey_type = "eczane"  # Default
        if segment_col and segment_col in df.columns:
            seg_raw = row.get(segment_col)
            if seg_raw is not None and not (isinstance(seg_raw, float) and pd.isna(seg_raw)):
                seg_str = str(seg_raw).strip().lower()
                if seg_str == "doktor":
                    survey_type = "doktor"

        print(f"Soru eklendi: {t[:50]}... | Ağırlık: {w} | Kategori: {c} | Tip: {survey_type}")
        records.append((t, w, c, survey_type))

    print(f"Toplam {len(records)} soru yüklendi")
    
    # Veritabanına kaydet
    with engine.begin() as conn:
        for t, w, c, survey_type in records:
            conn.execute(text("INSERT INTO questions(text, weight, category, survey_type) VALUES (:t,:w,:c,:st)"),
                         {"t": t, "w": w, "c": c, "st": survey_type})

def init_questions():
    preferred = [
        DATA_DIR / "5_Anket_Sorulari.csv",
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
            # Eczane soruları
            for t, w, c in [
                ("Ziyaret öncesi hazırlıklar eksiksiz yapıldı mı?", 1.0, "Müşteri Odaklılık"),
                ("Ürün mesajı net ve doğru aktarıldı mı?", 1.5, "Sonuç Çevikliği"),
                ("Kapanış ve takip adımı planlandı mı?", 1.2, "Sonuç Çevikliği"),
            ]:
                conn.execute(text("INSERT INTO questions(text, weight, category, survey_type) VALUES (:t,:w,:c,:st)"),
                             {"t": t, "w": w, "c": c, "st": "eczane"})
            # Doktor soruları
            for t, w, c in [
                ("HCP'nin segmentini dikkate alır", 1.0, "Müşteri Odaklılık"),
                ("Tüm portföy için medikal bilgiye hakimdir", 1.5, "Öğrenme Çevikliği"),
                ("Sosyal stil farkındalığını gösterir", 1.2, "Kişisel Farkındalık"),
            ]:
                conn.execute(text("INSERT INTO questions(text, weight, category, survey_type) VALUES (:t,:w,:c,:st)"),
                             {"t": t, "w": w, "c": c, "st": "doktor"})

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
        self.user_emails = self._load_user_emails()
        self.super_users = {"Hakan Öktem", "Ceyhan Gırça"}
        self.temp_codes: Dict[str, Dict] = {}

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

    def _load_user_emails(self) -> Dict[str, str]:
        """CSV'den kullanıcı email'lerini yükle"""
        emails = {}
        csv_path = DATA_DIR / "6_Mudur_Sifreler.csv"
        df = read_csv_guess(csv_path)
        
        if df is not None:
            manager_col = pick_column(df, ["manager", "müdür", "mudur", "bölge müdürü"])
            email_col = pick_column(df, ["email", "e-mail", "mail", "e_mail"], default_idx=2)
            
            for _, row in df.iterrows():
                mgr = str(row.get(manager_col, "")).strip()
                email = str(row.get(email_col, "")).strip()
                
                if mgr and email and mgr != "nan" and email != "nan":
                    emails[mgr] = email
                    print(f"Email yüklendi: {mgr} -> {email}")
        
        print(f"Toplam {len(emails)} kullanıcı email'i yüklendi")
        return emails

    def get_user_email(self, manager_name: str) -> str:
        """Kullanıcının email adresini döndür"""
        return self.user_emails.get(manager_name, "")

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def is_super_user(self, manager_name: str) -> bool:
        return manager_name in self.super_users

    def get_available_managers(self) -> List[str]:
        available_managers = list(self.super_users)

        if not self.passwords:
            for mgr in self.lookups.get("managers", []):
                if mgr not in self.super_users:
                    available_managers.append(mgr)
        else:
            for mgr in self.lookups.get("managers", []):
                if mgr in self.passwords and mgr not in self.super_users:
                    available_managers.append(mgr)

        return sorted(set(available_managers))

    def authenticate_manager(self, manager_name: str, password: str = None) -> bool:
        if self.is_super_user(manager_name):
            if not password:
                return False
            if manager_name in self.passwords:
                hashed_input = self._hash_password(password)
                return hashed_input == self.passwords[manager_name]
            return False

        if not self.passwords:
            return manager_name in self.lookups.get("managers", [])

        if not password:
            return False

        if manager_name not in self.passwords:
            return False

        hashed_input = self._hash_password(password)
        return hashed_input == self.passwords[manager_name]

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

    def change_password(self, manager_name: str, old_password: str, new_password: str) -> bool:
        if not self.authenticate_manager(manager_name, old_password):
            return False
        self.passwords[manager_name] = self._hash_password(new_password)
        return self._save_passwords_to_csv()

    def reset_password_request(self, manager_name: str) -> dict:
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

def _category_vector_for_query(where_sql: str, params: dict, survey_type: str = "eczane") -> Tuple[List[float], dict]:
    """Kategori vektörü hesaplama - anket türüne göre"""
    sql = f"""
        SELECT a.answer, a.weight AS w, q.category AS cat
        FROM answers a
        JOIN evaluations e ON e.id = a.evaluation_id
        JOIN questions   q ON q.id = a.question_id
        WHERE {where_sql} AND COALESCE(e.survey_type, 'eczane') = :survey_type
    """
    
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {**params, "survey_type": survey_type}).mappings().all()
        meta = conn.execute(text(f"""
            SELECT COUNT(DISTINCT e.id) AS cnt, MAX(e.created_at) AS last_at
            FROM evaluations e
            WHERE {where_sql.replace('a.','').replace('q.','')} 
            AND COALESCE(e.survey_type, 'eczane') = :survey_type
        """), {**params, "survey_type": survey_type}).mappings().first()

    # Anket türüne göre kategori listesi al
    categories = get_survey_categories(survey_type)
    
    num = defaultdict(float)
    den = defaultdict(float)
    
    for r in rows:
        answer = int(r["answer"])
        w = float(r["w"] or 0.0)
        cats = parse_category_tags(r["cat"], survey_type)
        for cat, share in cats:
            contrib = w * float(share)
            den[cat] += contrib
            if answer:
                num[cat] += contrib

    vec = []
    for c in categories:
        if den[c] > 0:
            vec.append(round(100.0 * num[c] / den[c], 1))
        else:
            vec.append(0.0)
    
    return vec, {
        "samples": int(meta["cnt"] or 0), 
        "last_eval_at": meta["last_at"],
        "categories": categories
    }

def _calculate_scores(rows, survey_type: str = "eczane"):
    """Temel skor hesaplama fonksiyonu - anket türüne göre"""
    categories = get_survey_categories(survey_type)
    
    num = defaultdict(float)
    den = defaultdict(float)
    
    for r in rows:
        answer = int(r["answer"])
        w = float(r["w"] or 0.0)
        cats = parse_category_tags(r["cat"], survey_type)
        
        for cat, share in cats:
            contrib = w * float(share)
            den[cat] += contrib
            if answer:
                num[cat] += contrib
    
    vec = []
    for c in categories:
        if den[c] > 0:
            vec.append(round(100.0 * num[c] / den[c], 1))
        else:
            vec.append(0.0)
    
    return vec

def _calculate_recent_performance(rows, survey_type: str = "eczane", limit_evaluations=7):
    """Son N değerlendirmenin ortalaması"""
    if not rows:
        categories = get_survey_categories(survey_type)
        return [0.0] * len(categories), {"method": "recent", "evaluations_used": 0}
    
    eval_dates = sorted(list(set(row["created_at"] for row in rows)), reverse=True)[:limit_evaluations]
    recent_rows = [row for row in rows if row["created_at"] in eval_dates]
    
    vec = _calculate_scores(recent_rows, survey_type)
    
    return vec, {
        "method": "recent",
        "evaluations_used": len(eval_dates),
        "date_range": {
            "from": eval_dates[-1][:10] if eval_dates else None,
            "to": eval_dates[0][:10] if eval_dates else None
        }
    }

def _calculate_weighted_performance(rows, survey_type: str = "eczane"):
    """Tarih ağırlıklı hesaplama"""
    if not rows:
        categories = get_survey_categories(survey_type)
        return [0.0] * len(categories), {"method": "weighted", "samples": 0}
    
    dates = [datetime.fromisoformat(row["created_at"][:19]) for row in rows]
    oldest_date = min(dates)
    newest_date = max(dates)
    total_days = (newest_date - oldest_date).days or 1
    
    categories = get_survey_categories(survey_type)
    num = defaultdict(float)
    den = defaultdict(float)
    
    for r in rows:
        answer = int(r["answer"])
        w = float(r["w"] or 0.0)
        row_date = datetime.fromisoformat(r["created_at"][:19])
        
        progress = (row_date - oldest_date).days / total_days
        time_weight = 1.0 + progress
        
        cats = parse_category_tags(r["cat"], survey_type)
        for cat, share in cats:
            weighted_contribution = w * float(share) * time_weight
            den[cat] += weighted_contribution
            if answer:
                num[cat] += weighted_contribution
    
    vec = []
    for c in categories:
        if den[c] > 0:
            vec.append(round(100.0 * num[c] / den[c], 1))
        else:
            vec.append(0.0)
    
    return vec, {
        "method": "weighted", 
        "samples": len(set(r["created_at"] for r in rows)),
        "time_span_days": total_days
    }

def _calculate_trend_analysis(rows, survey_type: str = "eczane"):
    """Gelişim trendi analizi"""
    categories = get_survey_categories(survey_type)
    
    if not rows:
        return [0.0] * len(categories), {"method": "trend", "insufficient_data": True}
    
    eval_dates = sorted(list(set(row["created_at"] for row in rows)))
    
    if len(eval_dates) < 2:
        vec = _calculate_scores(rows, survey_type)
        return vec, {
            "method": "trend",
            "current_scores": vec,
            "early_scores": vec,
            "improvement": [0.0] * len(categories),
            "insufficient_data": True
        }
    
    mid_point = len(eval_dates) // 2
    early_dates = eval_dates[:mid_point] if mid_point > 0 else [eval_dates[0]]
    recent_dates = eval_dates[mid_point:]
    
    early_rows = [row for row in rows if row["created_at"] in early_dates]
    recent_rows = [row for row in rows if row["created_at"] in recent_dates]
    
    early_scores = _calculate_scores(early_rows, survey_type)
    recent_scores = _calculate_scores(recent_rows, survey_type)
    
    improvement = []
    for early, recent in zip(early_scores, recent_scores):
        if early > 0:
            change = recent - early
            improvement.append(round(change, 1))
        else:
            improvement.append(recent)
    
    return recent_scores, {
        "method": "trend",
        "current_scores": recent_scores,
        "early_scores": early_scores,
        "improvement": improvement,
        "early_period": {"from": early_dates[0][:10], "to": early_dates[-1][:10]},
        "recent_period": {"from": recent_dates[0][:10], "to": recent_dates[-1][:10]},
        "total_evaluations": len(eval_dates)
    }

def _calculate_target_based(rows, survey_type: str = "eczane", targets=None):
    """Hedefe yakınlık bazlı hesaplama"""
    categories = get_survey_categories(survey_type)
    
    if targets is None:
        targets = {cat: 80 for cat in categories}
    
    current_scores = _calculate_scores(rows, survey_type)
    
    target_achievement = []
    target_values = []
    
    for i, cat in enumerate(categories):
        current = current_scores[i]
        target = targets.get(cat, 80)
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

def _category_vector_for_query_enhanced(where_sql: str, params: dict, method: str = "traditional", survey_type: str = "eczane") -> Tuple[List[float], dict]:
    """Gelişmiş radar hesaplama - anket türü desteği ile"""
    sql = f"""
        SELECT a.answer, a.weight AS w, q.category AS cat, e.created_at
        FROM answers a
        JOIN evaluations e ON e.id = a.evaluation_id
        JOIN questions   q ON q.id = a.question_id
        WHERE {where_sql} AND COALESCE(e.survey_type, 'eczane') = :survey_type
        ORDER BY e.created_at DESC
    """
    
    print(f"ENHANCED CATEGORY VECTOR - SQL: {sql}")
    print(f"ENHANCED CATEGORY VECTOR - PARAMS: {params}")
    print(f"ENHANCED CATEGORY VECTOR - METHOD: {method}, SURVEY_TYPE: {survey_type}")
    
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {**params, "survey_type": survey_type}).mappings().all()
        meta_basic = conn.execute(text(f"""
            SELECT COUNT(DISTINCT e.id) AS cnt, MAX(e.created_at) AS last_at
            FROM evaluations e
            WHERE {where_sql.replace('a.','').replace('q.','')} 
            AND COALESCE(e.survey_type, 'eczane') = :survey_type
        """), {**params, "survey_type": survey_type}).mappings().first()
    
    print(f"ENHANCED CATEGORY VECTOR - Total rows: {len(rows)}")
    
    base_meta = {
        "samples": int(meta_basic["cnt"] or 0),
        "last_eval_at": meta_basic["last_at"],
        "survey_type": survey_type,
        "categories": get_survey_categories(survey_type)
    }
    
    if method == "recent":
        vec, method_meta = _calculate_recent_performance(rows, survey_type, 7)
    elif method == "weighted":
        vec, method_meta = _calculate_weighted_performance(rows, survey_type)
    elif method == "trend":
        vec, method_meta = _calculate_trend_analysis(rows, survey_type)
    elif method == "target":
        vec, method_meta = _calculate_target_based(rows, survey_type)
    else:  # traditional
        vec, method_meta = _category_vector_for_query(where_sql, {**params, "survey_type": survey_type}, survey_type)
        return vec, {**base_meta, "method": "traditional"}
    
    combined_meta = {**base_meta, **method_meta}
    
    print(f"ENHANCED CATEGORY VECTOR - Final vector: {vec}")
    print(f"ENHANCED CATEGORY VECTOR - Meta: {combined_meta}")
    
    return vec, combined_meta

# --------------------------- API ENDPOINTS ---------------------------

@app.get("/api/rep-radar")
def api_rep_radar(rep: str, manager: Optional[str] = None,
                  date_from: Optional[str] = None, date_to: Optional[str] = None,
                  method: str = "traditional", prev: int = 0,
                  survey_type: str = "eczane",
                  current_user: str = Depends(get_current_user)):
    """Radar API - Anket türü desteği eklendi"""
    if not rep:
        return JSONResponse({"error": "rep required"}, status_code=400)

    # Survey type validation
    if survey_type not in SURVEY_TYPES:
        survey_type = "eczane"

    d1, d2, span_days = _daterange_defaults(date_from, date_to)
    
    print(f"RADAR API - User: {current_user}, Rep: {rep}, Method: {method}, Survey Type: {survey_type}")
    
    where = f"e.rep_name = :rep AND {sql_date_cast('e.created_at')} BETWEEN :d1 AND :d2"
    params = {"rep": rep, "d1": d1, "d2": d2}
    
    if not auth_manager.is_super_user(current_user):
        if manager:
            where += " AND e.manager_name = :mgr"
            params["mgr"] = manager
        else:
            where += " AND e.manager_name = :current_user"
            params["current_user"] = current_user

    curr_vec, meta = _category_vector_for_query_enhanced(where, params, method, survey_type)
    
    payload = {
        "labels": get_survey_categories(survey_type),
        "data": curr_vec,
        "method": method,
        "survey_type": survey_type,
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
        
        p_vec, p_meta = _category_vector_for_query_enhanced(prev_where, prev_params, method, survey_type)
        payload["prev"] = p_vec
        payload["prevRange"] = {"from": prev_from, "to": prev_to}
        payload["prevMeta"] = p_meta

    return JSONResponse(payload)

@app.get("/api/survey-types")
def get_survey_types():
    """Anket türlerini döndür"""
    return JSONResponse({
        "types": SURVEY_TYPES,
        "default": "eczane"
    })

@app.get("/api/questions/{survey_type}")
def get_questions_by_type(survey_type: str, current_user: str = Depends(get_current_user)):
    """Anket türüne göre soruları döndür"""
    if survey_type not in SURVEY_TYPES:
        return JSONResponse({"error": "Invalid survey type"}, status_code=400)
    
    with engine.begin() as conn:
        questions = conn.execute(text("""
            SELECT id, text, weight, category
            FROM questions 
            WHERE COALESCE(survey_type, 'eczane') = :survey_type
            ORDER BY id
        """), {"survey_type": survey_type}).mappings().all()
    
    return JSONResponse({
        "questions": [dict(q) for q in questions],
        "survey_info": SURVEY_TYPES[survey_type],
        "categories": get_survey_categories(survey_type)
    })

@app.get("/lookups")
def get_lookups(current_user: str = Depends(get_current_user)):
    return JSONResponse(auth_manager.get_user_specific_lookups(current_user))

@app.get("/api/pharmacy-info/{pharmacy_name}")
def get_pharmacy_info(pharmacy_name: str, current_user: str = Depends(get_current_user)):
    """Eczane seçildiğinde brick bilgisini döndür"""
    user_lookups = auth_manager.get_user_specific_lookups(current_user)
    
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

# --------------------------- DASHBOARD API ENDPOINTS ---------------------------

@app.get("/api/dashboard-stats")
def dashboard_stats(current_user: str = Depends(get_current_user),
                   time_range: int = 30,
                   rep_filter: Optional[str] = None,
                   brick_filter: Optional[str] = None,
                   survey_type_filter: Optional[str] = None):
    """Dashboard için temel istatistikler - anket türü filtreleme eklendi"""
    
    end_date = date.today()
    start_date = end_date - timedelta(days=time_range)
    
    where_conditions = [f"{sql_date_cast('e.created_at')} BETWEEN :start_date AND :end_date"]
    params = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
    
    if not auth_manager.is_super_user(current_user):
        where_conditions.append("e.manager_name = :current_user")
        params["current_user"] = current_user
    
    if rep_filter and rep_filter.strip():
        where_conditions.append("e.rep_name = :rep_filter")
        params["rep_filter"] = rep_filter.strip()
    
    if brick_filter and brick_filter.strip():
        where_conditions.append("e.brick_name = :brick_filter")
        params["brick_filter"] = brick_filter.strip()
    
    # Anket türü filtresi
    if survey_type_filter and survey_type_filter in SURVEY_TYPES:
        where_conditions.append("COALESCE(e.survey_type, 'eczane') = :survey_type_filter")
        params["survey_type_filter"] = survey_type_filter
    
    where_clause = " AND ".join(where_conditions)
    print(f"DASHBOARD STATS - WHERE: {where_clause}")
    print(f"DASHBOARD STATS - PARAMS: {params}")
    
    with engine.begin() as conn:
        # Temel istatistikler
        basic_stats = conn.execute(text(f"""
            SELECT 
                COUNT(*) as total_evaluations,
                COUNT(DISTINCT e.rep_name) as active_reps,
                COUNT(DISTINCT CASE WHEN COALESCE(e.survey_type, 'eczane') = 'eczane' 
                      THEN e.pharmacy_name ELSE e.doctor_name END) as active_targets,
                AVG(e.percentage) as avg_performance,
                MAX(e.created_at) as last_evaluation
            FROM evaluations e
            WHERE {where_clause}
        """), params).mappings().first()
        
        # Anket türü dağılımı
        survey_distribution = conn.execute(text(f"""
            SELECT 
                COALESCE(e.survey_type, 'eczane') as survey_type,
                COUNT(*) as count
            FROM evaluations e
            WHERE {where_clause}
            GROUP BY COALESCE(e.survey_type, 'eczane')
        """), params).mappings().all()
        
        # Performans dağılımı
        performance_distribution = conn.execute(text(f"""
            SELECT 
                CASE 
                    WHEN e.percentage >= 90 THEN 'Mükemmel'
                    WHEN e.percentage >= 80 THEN 'Çok İyi'
                    WHEN e.percentage >= 70 THEN 'İyi'
                    ELSE 'Gelişmeli'
                END as performance_level,
                COUNT(*) as count
            FROM evaluations e
            WHERE {where_clause}
            GROUP BY performance_level
        """), params).mappings().all()
        
        # Günlük aktivite
        daily_activity = conn.execute(text(f"""
            SELECT 
                {sql_date_cast('e.created_at')} as eval_date,
                COUNT(*) as daily_count,
                COALESCE(e.survey_type, 'eczane') as survey_type
            FROM evaluations e
            WHERE {where_clause} 
                AND {sql_date_cast('e.created_at')} >= :recent_date
            GROUP BY {sql_date_cast('e.created_at')}, COALESCE(e.survey_type, 'eczane')
            ORDER BY eval_date DESC
            LIMIT 14
        """), {**params, "recent_date": (end_date - timedelta(days=7)).isoformat()}).mappings().all()
    
    return JSONResponse({
        "total_evaluations": int(basic_stats["total_evaluations"] or 0),
        "active_reps": int(basic_stats["active_reps"] or 0),
        "active_targets": int(basic_stats["active_targets"] or 0),  # eczane veya doktor sayısı
        "avg_performance": round(basic_stats["avg_performance"] or 0, 1),
        "last_evaluation": basic_stats["last_evaluation"],
        "survey_distribution": [dict(row) for row in survey_distribution],
        "performance_distribution": [dict(row) for row in performance_distribution],
        "daily_activity": [dict(row) for row in daily_activity],
        "time_range": time_range,
        "filters": {
            "rep_filter": rep_filter,
            "brick_filter": brick_filter,
            "survey_type_filter": survey_type_filter
        }
    })

# Dashboard chart endpoint'ını düzelt - "Tümü" seçeneği için veri birleştirme
@app.get("/api/dashboard-charts")
def dashboard_charts(current_user: str = Depends(get_current_user),
                    time_range: int = 30,
                    chart_type: str = "performance_trend",
                    method: str = "traditional",
                    rep_filter: Optional[str] = None,
                    brick_filter: Optional[str] = None,
                    survey_type_filter: Optional[str] = None):
    """Dashboard grafik verileri - Tümü seçeneği için düzeltilmiş veri birleştirme"""
    
    end_date = date.today()
    start_date = end_date - timedelta(days=time_range)
    
    where_conditions = [f"{sql_date_cast('e.created_at')} BETWEEN :start_date AND :end_date"]
    params = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
    
    if not auth_manager.is_super_user(current_user):
        where_conditions.append("e.manager_name = :current_user")
        params["current_user"] = current_user
    
    if rep_filter and rep_filter.strip():
        where_conditions.append("e.rep_name = :rep_filter")
        params["rep_filter"] = rep_filter.strip()
    
    if brick_filter and brick_filter.strip():
        where_conditions.append("e.brick_name = :brick_filter") 
        params["brick_filter"] = brick_filter.strip()
    
    # Survey type filtresi - Tümü için boş bırak
    if survey_type_filter and survey_type_filter in SURVEY_TYPES:
        where_conditions.append("COALESCE(e.survey_type, 'eczane') = :survey_type_filter")
        params["survey_type_filter"] = survey_type_filter
    
    where_clause = " AND ".join(where_conditions)
    
    with engine.begin() as conn:
        if chart_type == "performance_trend":
            chart_data = conn.execute(text(f"""
                SELECT 
                    {sql_date_cast('e.created_at')} as eval_date,
                    AVG(e.percentage) as avg_performance,
                    COUNT(*) as count,
                    COALESCE(e.survey_type, 'eczane') as survey_type
                FROM evaluations e
                WHERE {where_clause}
                GROUP BY {sql_date_cast('e.created_at')}, COALESCE(e.survey_type, 'eczane')
                ORDER BY eval_date
            """), params).mappings().all()
            
            # Anket türüne göre dataset'leri ayır
            eczane_data = [row for row in chart_data if row["survey_type"] == "eczane"]
            doktor_data = [row for row in chart_data if row["survey_type"] == "doktor"]
            
            # Tüm tarihleri birleştir ve sırala
            all_dates = sorted(list(set(row["eval_date"] for row in chart_data)))
            
            datasets = []
            
            if eczane_data:
                # Eczane verilerini tarih sırasına göre düzenle
                eczane_dict = {row["eval_date"]: row["avg_performance"] for row in eczane_data}
                eczane_values = [round(eczane_dict.get(date, 0), 1) for date in all_dates]
                
                datasets.append({
                    "label": "Eczane Performansı (%)",
                    "data": eczane_values,
                    "borderColor": "#3b82f6",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                    "tension": 0.4,
                    "fill": False
                })
            
            if doktor_data:
                # Doktor verilerini tarih sırasına göre düzenle
                doktor_dict = {row["eval_date"]: row["avg_performance"] for row in doktor_data}
                doktor_values = [round(doktor_dict.get(date, 0), 1) for date in all_dates]
                
                datasets.append({
                    "label": "Doktor Performansı (%)",
                    "data": doktor_values,
                    "borderColor": "#10b981",
                    "backgroundColor": "rgba(16, 185, 129, 0.1)",
                    "tension": 0.4,
                    "fill": False
                })
            
            return JSONResponse({
                "labels": all_dates,
                "datasets": datasets,
                "type": "line"
            })
            
        elif chart_type == "rep_performance":
            chart_data = conn.execute(text(f"""
                SELECT 
                    e.rep_name,
                    COALESCE(e.survey_type, 'eczane') as survey_type,
                    COUNT(*) as evaluation_count,
                    AVG(e.percentage) as avg_performance
                FROM evaluations e
                WHERE {where_clause}
                GROUP BY e.rep_name, COALESCE(e.survey_type, 'eczane')
                HAVING COUNT(*) >= 1
                ORDER BY avg_performance DESC
                LIMIT 15
            """), params).mappings().all()
            
            return JSONResponse({
                "labels": [f"{row['rep_name']} ({row['survey_type']})" for row in chart_data],
                "datasets": [{
                    "label": "Ortalama Performans (%)",
                    "data": [round(row["avg_performance"], 1) for row in chart_data],
                    "backgroundColor": [
                        "#10b981" if row["avg_performance"] >= 85 else
                        "#f59e0b" if row["avg_performance"] >= 70 else
                        "#ef4444" for row in chart_data
                    ]
                }],
                "type": "bar"
            })
            
        elif chart_type == "category_performance":
            # *** BU KISIM DÜZELTİLDİ ***
            
            # "Tümü" seçildiğinde birleştirilmiş kategori performansı hesapla
            if not survey_type_filter or survey_type_filter.strip() == "":
                print("CATEGORY PERFORMANCE: Calculating combined data for 'Tümü'")
                
                # Doktor kategorilerini kullan (5 kategori)
                target_categories = get_survey_categories("doktor")
                
                # Her kategori için hem eczane hem doktor verilerini birleştir
                category_scores = []
                for cat in target_categories:
                    print(f"Processing category: {cat}")
                    
                    # Bu kategori için tüm cevapları al (hem eczane hem doktor)
                    cat_result = conn.execute(text(f"""
                        SELECT 
                            AVG(CASE WHEN a.answer = 1 THEN 100.0 ELSE 0.0 END) as avg_score,
                            COUNT(*) as answer_count,
                            COALESCE(e.survey_type, 'eczane') as survey_type
                        FROM answers a
                        JOIN evaluations e ON e.id = a.evaluation_id
                        JOIN questions q ON q.id = a.question_id
                        WHERE {where_clause}
                        AND (q.category LIKE :cat_pattern OR q.category IS NULL)
                        GROUP BY COALESCE(e.survey_type, 'eczane')
                    """), {**params, "cat_pattern": f"%{cat}%"}).mappings().all()
                    
                    # Eczane ve doktor verilerini birleştir (ağırlıklı ortalama)
                    total_count = 0
                    total_score = 0.0
                    
                    for result in cat_result:
                        count = result["answer_count"] or 0
                        score = result["avg_score"] or 0.0
                        total_count += count
                        total_score += score * count
                        print(f"  {result['survey_type']}: {score}% ({count} cevap)")
                    
                    final_score = (total_score / total_count) if total_count > 0 else 0.0
                    category_scores.append(round(final_score, 1))
                    
                    print(f"  Final score for {cat}: {final_score}% (total: {total_count} cevap)")
                
                chart_title = "Kategori Performansı - Tüm Anketler (Birleşik)"
                
            else:
                # Belirli bir anket türü seçildi
                target_survey_type = survey_type_filter
                target_categories = get_survey_categories(target_survey_type)
                
                print(f"CATEGORY PERFORMANCE: Calculating data for {target_survey_type}")
                
                category_scores = []
                for cat in target_categories:
                    cat_result = conn.execute(text(f"""
                        SELECT 
                            AVG(CASE WHEN a.answer = 1 THEN 100.0 ELSE 0.0 END) as avg_score,
                            COUNT(*) as answer_count
                        FROM answers a
                        JOIN evaluations e ON e.id = a.evaluation_id
                        JOIN questions q ON q.id = a.question_id
                        WHERE {where_clause} 
                        AND COALESCE(e.survey_type, 'eczane') = :target_survey_type
                        AND (q.category LIKE :cat_pattern OR q.category IS NULL)
                    """), {**params, "target_survey_type": target_survey_type, "cat_pattern": f"%{cat}%"}).scalar()
                    
                    category_scores.append(round(cat_result or 0, 1))
                
                chart_title = f"Kategori Performansı - {SURVEY_TYPES[target_survey_type]['name']}"
            
            return JSONResponse({
                "labels": target_categories,
                "datasets": [{
                    "label": f"{chart_title} (%)",
                    "data": category_scores,
                    "backgroundColor": "rgba(59, 130, 246, 0.2)",
                    "borderColor": "#3b82f6", 
                    "pointBackgroundColor": "#3b82f6",
                    "pointBorderColor": "#fff",
                    "borderWidth": 2
                }],
                "type": "radar",
                "method": method,
                "survey_type": survey_type_filter or "combined"
            })
    
    return JSONResponse({"error": "Invalid chart type"}, status_code=400)

@app.get("/api/dashboard-recent-activities")
def dashboard_recent_activities(current_user: str = Depends(get_current_user),
                               limit: int = 10,
                               survey_type_filter: Optional[str] = None):
    """Son aktiviteler listesi - anket türü filtreleme eklendi"""
    
    where_conditions = []
    params = {}
    
    if not auth_manager.is_super_user(current_user):
        where_conditions.append("e.manager_name = :current_user")
        params["current_user"] = current_user
    
    if survey_type_filter and survey_type_filter in SURVEY_TYPES:
        where_conditions.append("COALESCE(e.survey_type, 'eczane') = :survey_type_filter")
        params["survey_type_filter"] = survey_type_filter
    
    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
    with engine.begin() as conn:
        activities = conn.execute(text(f"""
            SELECT 
                e.id,
                e.created_at,
                e.rep_name,
                e.pharmacy_name,
                e.doctor_name,
                e.brick_name,
                COALESCE(e.survey_type, 'eczane') as survey_type,
                e.percentage,
                CASE 
                    WHEN e.percentage >= 90 THEN 'Mükemmel'
                    WHEN e.percentage >= 80 THEN 'Çok İyi'
                    WHEN e.percentage >= 70 THEN 'İyi'
                    ELSE 'Gelişmeli'
                END as status
            FROM evaluations e
            WHERE {where_clause}
            ORDER BY e.created_at DESC
            LIMIT :limit
        """), {**params, "limit": limit}).mappings().all()
    
    return JSONResponse([{
        "id": row["id"],
        "date": row["created_at"][:16].replace('T', ' '),
        "rep_name": row["rep_name"],
        "target_name": row["pharmacy_name"] if row["survey_type"] == "eczane" else row["doctor_name"],
        "target_type": "Eczane" if row["survey_type"] == "eczane" else "Doktor",
        "brick_name": row["brick_name"] or "-",
        "percentage": round(row["percentage"], 0),
        "status": row["status"],
        "survey_type": row["survey_type"]
    } for row in activities])

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
    
    if not manager_name or not manager_name.strip():
        print("LOGIN FAILED: Empty manager name")
        return RedirectResponse(url="/login?error=Lütfen bir bölge müdürü seçin", status_code=303)
    
    manager_name = manager_name.strip()
    
    if auth_manager.passwords:
        if not password or not password.strip():
            print("LOGIN FAILED: Password required but not provided")
            return RedirectResponse(url="/login?error=Şifre gerekli", status_code=303)
        
        password = password.strip()
        
        try:
            if auth_manager.authenticate_manager(manager_name, password):
                print(f"LOGIN SUCCESS: {manager_name}")
                response = RedirectResponse(url="/dashboard", status_code=303)
                encoded_name = urllib.parse.quote(manager_name, safe='')
                response.set_cookie("session_user", encoded_name, 
                                  httponly=True, max_age=3600*8, 
                                  samesite="lax", secure=False)
                return response
            else:
                print(f"LOGIN FAILED: Invalid credentials for {manager_name}")
                error_msg = urllib.parse.quote("Geçersiz kullanıcı adı veya şifre")
                return RedirectResponse(url=f"/login?error={error_msg}", status_code=303)
                
        except Exception as e:
            print(f"LOGIN ERROR: Authentication exception: {str(e)}")
            error_msg = urllib.parse.quote("Giriş işlemi sırasında hata oluştu")
            return RedirectResponse(url=f"/login?error={error_msg}", status_code=303)
    
    else:
        try:
            if auth_manager.authenticate_manager(manager_name):
                print(f"LOGIN SUCCESS (no password system): {manager_name}")
                response = RedirectResponse(url="/", status_code=303)
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
    if new_password != confirm_password:
        error_msg = "Şifreler eşleşmiyor"
        return RedirectResponse(url=f"/change-password?manager={current_user}&error={urllib.parse.quote(error_msg)}", status_code=303)
    
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
    
    try:
        decoded_name = urllib.parse.unquote(session_user)
        print(f"FORM PAGE: Decoded user = '{decoded_name}'")
    except:
        decoded_name = session_user
        print(f"FORM PAGE: Could not decode, using = '{decoded_name}'")
    
    all_managers = LOOKUPS.get('managers', [])
    if decoded_name not in all_managers and not auth_manager.is_super_user(decoded_name):
        print(f"FORM PAGE: User '{decoded_name}' not in all managers: {all_managers}")
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie("session_user")
        return response
    
    print(f"FORM PAGE: User '{decoded_name}' is valid, showing form")
    
    user_lookups = auth_manager.get_user_specific_lookups(decoded_name)
    
    # Kullanıcının bölgesini belirle
    user_region = ""
    if auth_manager.is_super_user(decoded_name):
        user_region = "Tüm Bölgeler"
    else:
        user_regions = user_lookups.get("regions", [])
        if user_regions:
            user_region = user_regions[0]  # İlk bölgeyi varsayılan yap
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "current_user": decoded_name,
        "user_region": user_region,  # Bu satır eksikti
        "has_passwords": bool(auth_manager.passwords),
        "survey_types": SURVEY_TYPES,
        **user_lookups
    })

@app.get("/api/dashboard-recent-activities")
def dashboard_recent_activities(current_user: str = Depends(get_current_user),
                               limit: int = 10,
                               survey_type_filter: Optional[str] = None):
    """Son aktiviteler listesi - düzeltilmiş versiyon"""
    
    where_conditions = []
    params = {}
    
    if not auth_manager.is_super_user(current_user):
        where_conditions.append("e.manager_name = :current_user")
        params["current_user"] = current_user
    
    if survey_type_filter and survey_type_filter in SURVEY_TYPES:
        where_conditions.append("COALESCE(e.survey_type, 'eczane') = :survey_type_filter")
        params["survey_type_filter"] = survey_type_filter
    
    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
    with engine.begin() as conn:
        activities = conn.execute(text(f"""
            SELECT 
                e.id,
                e.created_at,
                e.rep_name,
                e.pharmacy_name,
                e.doctor_name,
                e.brick_name,
                COALESCE(e.survey_type, 'eczane') as survey_type,
                e.percentage,
                CASE 
                    WHEN e.percentage >= 90 THEN 'Mükemmel'
                    WHEN e.percentage >= 80 THEN 'Çok İyi'
                    WHEN e.percentage >= 70 THEN 'İyi'
                    ELSE 'Gelişmeli'
                END as status
            FROM evaluations e
            WHERE {where_clause}
            ORDER BY e.created_at DESC
            LIMIT :limit
        """), {**params, "limit": limit}).mappings().all()
    
    return JSONResponse([{
        "id": row["id"],
        "date": row["created_at"][:16].replace('T', ' ') if row["created_at"] else "",
        "rep_name": row["rep_name"],
        "target_name": row["pharmacy_name"] if row["survey_type"] == "eczane" else row["doctor_name"],
        "target_type": "Eczane" if row["survey_type"] == "eczane" else "Doktor",
        "brick_name": row["brick_name"] or "-",
        "percentage": round(row["percentage"], 0) if row["percentage"] else 0,
        "status": row["status"],
        "survey_type": row["survey_type"]
    } for row in activities])

@app.get("/api/dashboard-top-performers")
def dashboard_top_performers(current_user: str = Depends(get_current_user),
                           time_range: int = 30,
                           rep_filter: Optional[str] = None,
                           brick_filter: Optional[str] = None,
                           survey_type_filter: Optional[str] = None,
                           limit: int = 10):
    """En iyi performans gösteren temsilciler"""
    
    end_date = date.today()
    start_date = end_date - timedelta(days=time_range)
    
    where_conditions = [f"{sql_date_cast('e.created_at')} BETWEEN :start_date AND :end_date"]
    params = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
    
    if not auth_manager.is_super_user(current_user):
        where_conditions.append("e.manager_name = :current_user")
        params["current_user"] = current_user
    
    if rep_filter and rep_filter.strip():
        where_conditions.append("e.rep_name = :rep_filter")
        params["rep_filter"] = rep_filter.strip()
    
    if brick_filter and brick_filter.strip():
        where_conditions.append("e.brick_name = :brick_filter")
        params["brick_filter"] = brick_filter.strip()
    
    if survey_type_filter and survey_type_filter in SURVEY_TYPES:
        where_conditions.append("COALESCE(e.survey_type, 'eczane') = :survey_type_filter")
        params["survey_type_filter"] = survey_type_filter
    
    where_clause = " AND ".join(where_conditions)
    
    with engine.begin() as conn:
        performers = conn.execute(text(f"""
            SELECT 
                e.rep_name,
                COALESCE(e.survey_type, 'eczane') as survey_type,
                COUNT(*) as evaluation_count,
                AVG(e.percentage) as avg_performance,
                MAX(e.created_at) as last_evaluation
            FROM evaluations e
            WHERE {where_clause}
            GROUP BY e.rep_name, COALESCE(e.survey_type, 'eczane')
            HAVING COUNT(*) >= 2
            ORDER BY avg_performance DESC, evaluation_count DESC
            LIMIT :limit
        """), {**params, "limit": limit}).mappings().all()
    
    return JSONResponse([{
        "rep_name": row["rep_name"],
        "survey_type": row["survey_type"],
        "evaluation_count": int(row["evaluation_count"]),
        "avg_performance": round(row["avg_performance"], 1),
        "last_evaluation": row["last_evaluation"][:10] if row["last_evaluation"] else None
    } for row in performers])

@app.get("/evaluations", response_class=HTMLResponse)
def evaluations_page(request: Request, 
                    current_user: str = Depends(get_current_user),
                    q: Optional[str] = None,
                    date_from: Optional[str] = None, 
                    date_to: Optional[str] = None,
                    survey_type_filter: Optional[str] = None):
    """Değerlendirme kayıtları sayfası - anket türü filtreleme eklendi"""
    
    if not date_from and not date_to:
        date_to = date.today().isoformat()
        date_from = (date.today() - timedelta(days=30)).isoformat()
    
    where_conditions = []
    params = {}
    
    if not auth_manager.is_super_user(current_user):
        where_conditions.append("e.manager_name = :current_user")
        params["current_user"] = current_user
    
    if date_from:
        where_conditions.append(f"{sql_date_cast('e.created_at')} >= :date_from")
        params["date_from"] = date_from
    if date_to:
        where_conditions.append(f"{sql_date_cast('e.created_at')} <= :date_to")
        params["date_to"] = date_to
    
    if survey_type_filter and survey_type_filter in SURVEY_TYPES:
        where_conditions.append("COALESCE(e.survey_type, 'eczane') = :survey_type_filter")
        params["survey_type_filter"] = survey_type_filter
    
    if q and q.strip():
        search_term = f"%{q.strip()}%"
        where_conditions.append("""
            (e.manager_name LIKE :search OR 
             e.rep_name LIKE :search OR 
             e.region_name LIKE :search OR 
             e.pharmacy_name LIKE :search OR
             e.doctor_name LIKE :search)
        """)
        params["search"] = search_term
    
    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
    with engine.begin() as conn:
        evaluations = conn.execute(text(f"""
            SELECT e.id, e.created_at, e.manager_name, e.rep_name, 
                   e.region_name, e.brick_name, e.pharmacy_name, e.doctor_name,
                   COALESCE(e.survey_type, 'eczane') as survey_type,
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
        "is_super_user": auth_manager.is_super_user(current_user),
        "survey_types": SURVEY_TYPES
    })

@app.get("/evaluation/{eval_id}", response_class=HTMLResponse)
def evaluation_detail(eval_id: int, request: Request, 
                     current_user: str = Depends(get_current_user)):
    """Değerlendirme detay sayfası"""
    
    with engine.begin() as conn:
        eval_query = """
            SELECT e.id, e.created_at, e.manager_name, e.rep_name, 
                   e.region_name, e.brick_name, e.pharmacy_name, e.doctor_name,
                   COALESCE(e.survey_type, 'eczane') as survey_type,
                   e.total_score, e.max_score, e.percentage, e.notes
            FROM evaluations e 
            WHERE e.id = :eval_id
        """
        
        if not auth_manager.is_super_user(current_user):
            eval_query += " AND e.manager_name = :current_user"
        
        evaluation = conn.execute(text(eval_query), {
            "eval_id": eval_id,
            "current_user": current_user
        }).mappings().first()
        
        if not evaluation:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
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
        "current_user": current_user,
        "survey_types": SURVEY_TYPES
    })

@app.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request, current_user: str = Depends(get_current_user)):
    
    if auth_manager.is_super_user(current_user):
        user_lookups = LOOKUPS.copy()
        mgr_to_reps = {}
        
        region_to_reps = LOOKUPS.get("region_to_reps", {})
        
        for mgr in LOOKUPS.get("managers", []):
            mgr_regions = LOOKUPS.get("manager_to_regions", {}).get(mgr, [])
            mgr_reps = []
            for region in mgr_regions:
                mgr_reps.extend(region_to_reps.get(region, []))
            mgr_to_reps[mgr] = sorted(list(set(mgr_reps)))
        
        all_reps = []
        for reps_list in region_to_reps.values():
            all_reps.extend(reps_list)
        mgr_to_reps[current_user] = sorted(list(set(all_reps)))
        
    else:
        user_lookups = auth_manager.get_user_specific_lookups(current_user)
        mgr_to_reps = {current_user: user_lookups["reps"]}
    
    return templates.TemplateResponse("reports.html", {
        "request": request,
        "current_user": current_user,
        "current_manager": current_user,
        "managers": user_lookups["managers"],
        "mgr_to_reps": mgr_to_reps,
        "radar_order": CATS_ORDER,  # Backward compatibility
        "survey_types": SURVEY_TYPES,
        "is_super_user": auth_manager.is_super_user(current_user)
    })

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request, current_user: str = Depends(get_current_user)):
    """Dashboard sayfası"""
    user_email = auth_manager.get_user_email(current_user)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "current_user": current_user,
        "user_email": user_email,
        "is_super_user": auth_manager.is_super_user(current_user),
        "survey_types": SURVEY_TYPES
    })

@app.get("/detayli-rapor", response_class=HTMLResponse)
async def detayli_rapor(rep: str,
                        date_from: Optional[str] = None,
                        date_to: Optional[str] = None,
                        survey_type: Optional[str] = None,
                        current_user: str = Depends(get_current_user),
                        request: Request = None):
    """Detaylı rapor ekranı (HTML)"""

    report_data = api_generate_report(
        rep=rep,
        date_from=date_from,
        date_to=date_to,
        survey_type=survey_type,
        current_user=current_user
    )

    if isinstance(report_data, JSONResponse):
        import json
        report_dict = json.loads(report_data.body.decode())
    else:
        raise HTTPException(status_code=500, detail="Rapor alınamadı")

    return templates.TemplateResponse("detayli_rapor.html", {
        "request": request,
        **report_dict
    })


@app.get("/api/generate-report")
def api_generate_report(rep: str, 
                       date_from: Optional[str] = None,
                       date_to: Optional[str] = None,
                       survey_type: Optional[str] = None,
                       current_user: str = Depends(get_current_user)):
    """Detaylı rapor verisi oluştur - Düzeltilmiş veritabanı bağlantı yönetimi"""
    if not rep:
        return JSONResponse({"error": "Temsilci gerekli"}, status_code=400)
    
    d1, d2, span_days = _daterange_defaults(date_from, date_to)
    
    where_conditions = ["e.rep_name = :rep", f"{sql_date_cast('e.created_at')} BETWEEN :d1 AND :d2"]
    params = {"rep": rep, "d1": d1, "d2": d2}
    
    if not auth_manager.is_super_user(current_user):
        where_conditions.append("e.manager_name = :current_user")
        params["current_user"] = current_user
    
    # Survey type filtresi - boş ise tümü
    if survey_type and survey_type.strip() and survey_type in SURVEY_TYPES:
        where_conditions.append("COALESCE(e.survey_type, 'eczane') = :survey_type")
        params["survey_type"] = survey_type
    
    where_clause = " AND ".join(where_conditions)
    
    try:
        # HER SORGU İÇİN AYRI CONNECTION KULLAN
        
        # 1. Temel istatistikler
        with engine.begin() as conn1:
            basic_stats = conn1.execute(text(f"""
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(e.percentage) as avg_performance,
                    MAX(e.percentage) as max_performance,
                    MIN(e.percentage) as min_performance,
                    COUNT(DISTINCT CASE WHEN COALESCE(e.survey_type, 'eczane') = 'eczane' 
                          THEN e.pharmacy_name ELSE e.doctor_name END) as unique_visits
                FROM evaluations e
                WHERE {where_clause}
            """), params).mappings().first()
        
        # 2. Son ziyaretler
        with engine.begin() as conn2:
            recent_visits = conn2.execute(text(f"""
                SELECT e.created_at, e.pharmacy_name, e.doctor_name, e.brick_name,
                       COALESCE(e.survey_type, 'eczane') as survey_type, e.percentage
                FROM evaluations e
                WHERE {where_clause}
                ORDER BY e.created_at DESC
                LIMIT 20
            """), params).mappings().all()
        
        # 3. Trend analizi için tüm değerlendirmeler
        with engine.begin() as conn3:
            all_evals = conn3.execute(text(f"""
                SELECT e.created_at, e.percentage
                FROM evaluations e
                WHERE {where_clause}
                ORDER BY e.created_at
            """), params).mappings().all()
        
        # Trend analizi hesaplama
        trend_analysis = None
        if len(all_evals) >= 4:
            mid = len(all_evals) // 2
            first_half = all_evals[:mid]
            second_half = all_evals[mid:]
            
            first_avg = sum(e["percentage"] for e in first_half) / len(first_half)
            second_avg = sum(e["percentage"] for e in second_half) / len(second_half)
            improvement = second_avg - first_avg
            
            trend_analysis = {
                "first_half_avg": round(first_avg, 1),
                "second_half_avg": round(second_avg, 1),
                "improvement": round(improvement, 1),
                "trend": "pozitif" if improvement > 2 else "negatif" if improvement < -2 else "sabit"
            }
        
        # 4. Kategori performansları
        category_performance = {}
        for st in ["eczane", "doktor"]:
            where_with_type = where_clause + f" AND COALESCE(e.survey_type, 'eczane') = '{st}'"
            
            with engine.begin() as conn4:
                cat_data = conn4.execute(text(f"""
                    SELECT q.category, AVG(CASE WHEN a.answer = 1 THEN 100.0 ELSE 0.0 END) as avg_score
                    FROM answers a
                    JOIN evaluations e ON e.id = a.evaluation_id
                    JOIN questions q ON q.id = a.question_id
                    WHERE {where_with_type}
                    GROUP BY q.category
                    HAVING COUNT(*) > 0
                """), params).mappings().all()
            
            if cat_data:
                categories = []
                for cat in get_survey_categories(st):
                    score = 0
                    for row in cat_data:
                        if row["category"] and cat.lower() in row["category"].lower():
                            score = row["avg_score"] or 0
                            break
                    categories.append({"category": cat, "score": round(score, 1)})
                
                if any(c["score"] > 0 for c in categories):
                    category_performance[st] = categories
        
        return JSONResponse({
            "rep_name": rep,
            "date_range": {"from": d1, "to": d2},
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "basic_stats": {
                "total_evaluations": int(basic_stats["total_evaluations"] or 0),
                "avg_performance": round(basic_stats["avg_performance"] or 0, 1),
                "max_performance": round(basic_stats["max_performance"] or 0, 1),
                "min_performance": round(basic_stats["min_performance"] or 0, 1),
                "unique_visits": int(basic_stats["unique_visits"] or 0)
            },
            "trend_analysis": trend_analysis,
            "category_performance": category_performance,
            "recent_visits": [dict(visit) for visit in recent_visits]
        })
        
    except Exception as e:
        print(f"Rapor oluşturma hatası: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return JSONResponse({"error": f"Rapor oluşturulamadı: {str(e)}"}, status_code=500)

from fastapi.responses import FileResponse
import pandas as pd
from reportlab.pdfgen import canvas
import tempfile
import os

@app.get("/api/download-report")
async def api_download_report(rep: str,
                              date_from: Optional[str] = None,
                              date_to: Optional[str] = None,
                              survey_type: Optional[str] = None,
                              format: str = "pdf",
                              current_user: str = Depends(get_current_user)):
    """Rapor indir - PDF veya Excel"""

    # Önce JSON verisini alalım
    report_data = api_generate_report(
        rep=rep,
        date_from=date_from,
        date_to=date_to,
        survey_type=survey_type,
        current_user=current_user
    )

    if isinstance(report_data, JSONResponse):
        report_dict = report_data.body.decode()
        import json
        report_dict = json.loads(report_dict)
    else:
        return JSONResponse({"error": "Rapor alınamadı"}, status_code=500)

    # Geçici dosya üretelim
    tmpfile = tempfile.NamedTemporaryFile(delete=False)

    if format == "excel":
        df = pd.DataFrame(report_dict["recent_visits"])
        excel_path = tmpfile.name + ".xlsx"
        df.to_excel(excel_path, index=False)
        return FileResponse(
            excel_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"{rep}_rapor.xlsx"
        )

    elif format == "pdf":
        pdf_path = tmpfile.name + ".pdf"
        c = canvas.Canvas(pdf_path)
        c.setFont("Helvetica", 12)
        c.drawString(50, 800, f"Performans Raporu - {rep}")
        c.drawString(50, 780, f"Tarih Aralığı: {date_from} - {date_to}")
        c.drawString(50, 760, f"Toplam Değerlendirme: {report_dict['basic_stats']['total_evaluations']}")
        c.drawString(50, 740, f"Ortalama Performans: {report_dict['basic_stats']['avg_performance']}")
        c.save()

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{rep}_rapor.pdf"
        )

    else:
        return JSONResponse({"error": "Geçersiz format seçildi (pdf/excel)"}, status_code=400)
    

# --------------------------- FORM SUBMISSION ---------------------------

@app.post("/submit")
async def submit_form(
    request: Request,
    current_user: str = Depends(get_current_user),
    survey_type: str = Form(...),
    rep_name: str = Form(...),
    region_name: Optional[str] = Form(None),
    brick_name: Optional[str] = Form(None),
    pharmacy_name: Optional[str] = Form(None),
    doctor_name: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    """Form gönderimi - anket türü desteği eklendi"""
    
    # Survey type validation
    if survey_type not in SURVEY_TYPES:
        return RedirectResponse(url="/?error=Geçersiz anket türü", status_code=303)
    
    # Doktor anketi için doktor adı zorunlu
    if survey_type == "doktor":
        if not doctor_name or not doctor_name.strip():
            return RedirectResponse(url="/?error=Doktor anketi için doktor adı gereklidir", status_code=303)
    
    form = await request.form()
    
    # Bu anket türü için soruları al
    with engine.begin() as conn:
        qs = conn.execute(text("""
            SELECT id, text, weight 
            FROM questions 
            WHERE COALESCE(survey_type, 'eczane') = :survey_type
            ORDER BY id
        """), {"survey_type": survey_type}).mappings().all()
    
    # Tüm sorular cevaplandı mı kontrol et
    for q in qs:
        answer_field = f"answer_{q['id']}"
        if form.get(answer_field) is None:
            return RedirectResponse(url=f"/?error=Tüm soruları cevaplayın&survey_type={survey_type}", status_code=303)

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
    
    # Doktor anketi için not oluştur
    final_notes = (notes or "").strip()
    if survey_type == "doktor" and doctor_name and doctor_name.strip():
        doctor_note = f"Doktor: {doctor_name.strip()}"
        if final_notes:
            final_notes = f"{doctor_note}\n{final_notes}"
        else:
            final_notes = doctor_note
    
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO evaluations(created_at, manager_name, rep_name, region_name, brick_name, 
                                        pharmacy_name, doctor_name, survey_type,
                                        total_score, max_score, percentage, notes)
                VALUES (:created_at,:manager,:rep,:region,:brick,:pharmacy,:doctor,:survey_type,
                        :total,:max,:pct,:notes)
            """), {
                "created_at": datetime.utcnow().isoformat(timespec="seconds"),
                "manager": current_user,
                "rep": rep_name.strip(),
                "region": (region_name or "").strip() or None,
                "brick": (brick_name or "").strip() or None,
                "pharmacy": (pharmacy_name or "").strip() or None,
                "doctor": (doctor_name or "").strip() if survey_type == "doktor" else None,
                "survey_type": survey_type,
                "total": total, 
                "max": max_total, 
                "pct": pct, 
                "notes": final_notes or None,
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
        error_msg = urllib.parse.quote("Form gönderiminde hata oluştu")
        return RedirectResponse(url=f"/?error={error_msg}&survey_type={survey_type}", status_code=303)

# --------------------------- DEBUG ENDPOINTS ---------------------------

@app.get("/debug/questions-detail")
def debug_questions_detail():
    """Soruların kategori detaylarını göster"""
    with engine.begin() as conn:
        questions = conn.execute(text("""
            SELECT id, text, weight, category, COALESCE(survey_type, 'eczane') as survey_type
            FROM questions 
            ORDER BY survey_type, id
        """)).mappings().all()
        
        parsed_questions = []
        for q in questions:
            cats = parse_category_tags(q["category"], q["survey_type"])
            parsed_questions.append({
                "id": q["id"],
                "text": q["text"][:50] + "...",
                "weight": q["weight"],
                "survey_type": q["survey_type"],
                "raw_category": q["category"],
                "parsed_categories": cats
            })
        
        return JSONResponse({
            "questions_count": len(questions),
            "questions": parsed_questions,
            "survey_types": SURVEY_TYPES,
            "eczane_categories": get_survey_categories("eczane"),
            "doktor_categories": get_survey_categories("doktor")
        })

@app.get("/debug/check")
def debug_check():
    with engine.begin() as conn:
        evals = conn.execute(text("""
            SELECT id, rep_name, manager_name, created_at, total_score, 
                   COALESCE(survey_type, 'eczane') as survey_type,
                   pharmacy_name, doctor_name
            FROM evaluations 
            ORDER BY id DESC
            LIMIT 10
        """)).mappings().all()
        
        questions = conn.execute(text("""
            SELECT id, text, weight, category, COALESCE(survey_type, 'eczane') as survey_type
            FROM questions 
            ORDER BY survey_type, id
            LIMIT 10
        """)).mappings().all()
        
        return JSONResponse({
            "evaluations_count": len(evals),
            "recent_evaluations": [dict(e) for e in evals],
            "questions_count": len(questions),  
            "sample_questions": [dict(q) for q in questions],
            "available_managers": LOOKUPS.get("managers", []),
            "survey_types": SURVEY_TYPES
        })

# --------------------------- MAIN ENTRY POINT ---------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)