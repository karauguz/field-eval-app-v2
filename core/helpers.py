from core.database import engine

def sql_date_cast(col_expr: str) -> str:
    """SQLite ve PostgreSQL uyumlu tarih formatı"""
    backend = engine.url.get_backend_name()
    if backend == "sqlite":
        return f"DATE(substr({col_expr}, 1, 10))"
    else:
        return f"CAST(substr({col_expr},1,10) AS DATE)"