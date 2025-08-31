 
# core/database.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./field_eval.db")
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)

if DB_URL.startswith("sqlite"):
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DB_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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