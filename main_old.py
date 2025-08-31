# main.py - Modüler ve Auth'lu versiyon
from collections import defaultdict
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Set, List, Tuple
from pathlib import Path
import os
import unicodedata

import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, Depends, Cookie
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import modüller
from core.database import init_db, migrate_db, SessionLocal, engine
from core.config import BASE_DIR, DATA_DIR, CATS_ORDER, CATS_SET
from core.lookups import build_lookups, LOOKUPS
from core.questions import init_questions
from core.categories import parse_category_tags, canon_cat
from core.auth import AuthManager, get_current_user, login_required
from api.radar import router as radar_router
from utils.helpers import sql_date_cast

# --------------------------- App Setup ---------------------------
templates = Jinja2Templates(directory=str((BASE_DIR / "templates").as_posix()))

app = FastAPI(title="Saha Gözlem Formu")
app.mount("/static", StaticFiles(directory=str((BASE_DIR / "static").as_posix())), name="static")

# Auth manager
auth_manager = AuthManager()

# API router'ları dahil et
app.include_router(radar_router, prefix="/api")

# --------------------------- Boot ---------------------------
init_db()
migrate_db()
init_questions()

# --------------------------- Auth Routes ---------------------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse("login.html", {
        "request": request, 
        "error": error,
        "managers": auth_manager.get_available_managers()
    })

@app.post("/login")
async def login_submit(request: Request, manager_name: str = Form(...)):
    if auth_manager.authenticate_manager(manager_name):
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie("session_user", manager_name, httponly=True, max_age=3600*8)
        return response
    else:
        return RedirectResponse(url="/login?error=Geçersiz müdür", status_code=303)

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_user")
    return response

# --------------------------- Protected Pages ---------------------------
@app.get("/", response_class=HTMLResponse)
def form_page(request: Request, current_user: str = Depends(get_current_user)):
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, text, weight, COALESCE(category,'') AS category FROM questions ORDER BY category, id"
        )).mappings().all()
    
    # Kullanıcıya özel lookup'ları hazırla
    user_lookups = auth_manager.get_user_specific_lookups(current_user)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "questions": rows,
        "current_user": current_user,
        **user_lookups
    })

@app.get("/evaluations", response_class=HTMLResponse)
def evaluations_list(request: Request, current_user: str = Depends(get_current_user),
                     q: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None):
    where = ["manager_name = :current_user"]  # Sadece kendi kayıtları
    params = {"current_user": current_user}
    
    if q:
        where.append("(rep_name LIKE :kw OR COALESCE(region_name,'') LIKE :kw OR COALESCE(pharmacy_name,'') LIKE :kw)")
        params["kw"] = f"%{q}%"
    if date_from:
        where.append(f"{sql_date_cast('created_at')} >= :fromd")
        params["fromd"] = date_from
    if date_to:
        where.append(f"{sql_date_cast('created_at')} <= :tod")
        params["tod"] = date_to
    
    sql = "SELECT * FROM evaluations WHERE " + " AND ".join(where) + " ORDER BY id DESC"
    
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    
    return templates.TemplateResponse("evaluations.html", {
        "request": request, 
        "evaluations": rows,
        "current_user": current_user
    })

@app.get("/evaluation/{eval_id}", response_class=HTMLResponse)
def eval_detail(request: Request, eval_id: int, current_user: str = Depends(get_current_user)):
    with engine.begin() as conn:
        # Sadece kendi kayıtlarına erişim
        eval_row = conn.execute(text(
            "SELECT * FROM evaluations WHERE id=:i AND manager_name=:mgr"
        ), {"i": eval_id, "mgr": current_user}).mappings().first()
        
        if not eval_row:
            return RedirectResponse(url="/evaluations", status_code=303)
            
        ans = conn.execute(text(
            "SELECT * FROM answers WHERE evaluation_id=:i ORDER BY id"
        ), {"i": eval_id}).mappings().all()
    
    return templates.TemplateResponse("detail.html", {
        "request": request, 
        "eval": eval_row, 
        "answers": ans,
        "current_user": current_user
    })

@app.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request, current_user: str = Depends(get_current_user)):
    # Kullanıcıya özel temsilciler
    user_lookups = auth_manager.get_user_specific_lookups(current_user)
    
    return templates.TemplateResponse("reports.html", {
        "request": request,
        "current_user": current_user,
        "managers": [current_user],  # Sadece kendi adı
        "mgr_to_reps": {current_user: user_lookups["reps"]},
        "radar_order": CATS_ORDER
    })

# --------------------------- Submit ---------------------------
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
    print(f"Form submit başladı - Rep: {rep_name}, Manager: {current_user}")
    
    form = await request.form()
    with engine.begin() as conn:
        qs = conn.execute(text("SELECT id, text, weight FROM questions ORDER BY id")).mappings().all()
    
    # Tüm soruların cevaplandırılıp cevaplandırılmadığını kontrol et
    for q in qs:
        answer_field = f"answer_{q['id']}"
        if form.get(answer_field) is None:
            print(f"Eksik cevap: {answer_field}")
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
            # manager_name otomatik olarak current_user
            result = conn.execute(text("""
                INSERT INTO evaluations(created_at, manager_name, rep_name, region_name, brick_name, pharmacy_name,
                                        total_score, max_score, percentage, notes)
                VALUES (:created_at,:manager,:rep,:region,:brick,:pharmacy,:total,:max,:pct,:notes)
            """), {
                "created_at": datetime.utcnow().isoformat(timespec="seconds"),
                "manager": current_user,  # Auth'dan gelir
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

# --------------------------- API Endpoints (Debug için) ---------------------------
@app.get("/lookups")
def get_lookups(current_user: str = Depends(get_current_user)):
    return JSONResponse(auth_manager.get_user_specific_lookups(current_user))

# Debug endpoints (sadece development için)
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
            "sample_questions": [dict(q) for q in questions]
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)