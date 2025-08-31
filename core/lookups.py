from pathlib import Path
from typing import Set, Dict
import pandas as pd
from core.config import DATA_DIR
import unicodedata

def _norm_csv(x):
    s = str(x).strip()
    return s if s and s.lower() not in ["nan","none","null"] else ""

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

def pick_column(df, candidates, default_idx=0):
    low = [c.lower() for c in df.columns]
    for i,c in enumerate(low):
        if c in candidates: return df.columns[i]
    return df.columns[default_idx]

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

    # 1) Manager-Region
    df1 = read_csv_guess(DATA_DIR / "1_BM_Bolge.csv")
    if df1 is not None and not df1.empty:
        region_col = pick_column(df1, ["bölge","bolge","region","bölge adı","bolge adi"])
        manager_col = pick_column(df1, ["müdür","mudur","manager","bölge müdürü","bolge muduru","rm","regional manager"],
                                  default_idx=1 if df1.shape[1]>1 else 0)
        for _,r in df1.iterrows():
            reg = _norm_csv(r.get(region_col))
            mgr = _norm_csv(r.get(manager_col))
            if reg: regions.add(reg)
            if mgr:
                managers.add(mgr)
                manager_to_regions.setdefault(mgr, set()).add(reg)

    # 2) Region-Rep
    df2 = read_csv_guess(DATA_DIR / "2_Bolge_TTS.csv")
    if df2 is not None and not df2.empty:
        region_col = pick_column(df2, ["bölge","bolge","region"])
        rep_col = pick_column(df2, ["temsilci","sales rep","rep","çalışan","calisan","personel","ziyaret edilen çalışan"],
                              default_idx=1 if df2.shape[1]>1 else 0)
        for _,r in df2.iterrows():
            reg = _norm_csv(r.get(region_col))
            rep = _norm_csv(r.get(rep_col))
            if reg: regions.add(reg)
            if rep: reps.add(rep)
            if reg and rep:
                region_to_reps.setdefault(reg, set()).add(rep)

    # 3) Rep-Brick
    df3 = read_csv_guess(DATA_DIR / "3a_TTS_EczBrick.csv")
    if df3 is not None and not df3.empty:
        rep_col = pick_column(df3, ["temsilci","sales rep","rep","çalışan","calisan","personel"])
        brick_col = pick_column(df3, ["brick","eczbrick","ecz_brick","eczane brick","eczanebrick","tts_eczbrick","eczane_brick"],
                                default_idx=1 if df3.shape[1]>1 else 0)
        for _,r in df3.iterrows():
            rep = _norm_csv(r.get(rep_col))
            brick = _norm_csv(r.get(brick_col))
            if rep: reps.add(rep)
            if brick: bricks.add(brick)
            if rep and brick:
                rep_to_bricks.setdefault(rep, set()).add(brick)

    # 4) Brick-Pharmacy
    df4 = read_csv_guess(DATA_DIR / "4a_Brick_Eczane.csv")
    if df4 is not None and not df4.empty:
        brick_col = pick_column(df4, ["brick","eczbrick","ecz_brick","eczane brick","eczanebrick"])
        ph_col = pick_column(df4, ["eczane","pharmacy","pharmacy name","eczane adi","eczane adı","apteka"],
                             default_idx=1 if df4.shape[1]>1 else 0)
        for _,r in df4.iterrows():
            brick = _norm_csv(r.get(brick_col))
            ph = _norm_csv(r.get(ph_col))
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

# Global lookups
LOOKUPS = build_lookups()

# ======================================