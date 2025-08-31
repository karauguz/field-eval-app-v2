 
# core/auth.py
from typing import Optional, Dict, List, Set
from fastapi import HTTPException, Cookie, Depends
from core.lookups import LOOKUPS

class AuthManager:
    def __init__(self):
        self.lookups = LOOKUPS
    
    def get_available_managers(self) -> List[str]:
        """Tüm mevcut müdürlerin listesi"""
        return self.lookups.get("managers", [])
    
    def authenticate_manager(self, manager_name: str) -> bool:
        """Müdür isminin geçerliliğini kontrol et"""
        return manager_name in self.get_available_managers()
    
    def get_user_specific_lookups(self, manager_name: str) -> Dict:
        """Belirli bir müdüre ait alt birimleri getir"""
        if not manager_name:
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
        
        manager_regions = self.lookups.get("manager_to_regions", {}).get(manager_name, [])
        region_to_reps = self.lookups.get("region_to_reps", {})
        manager_reps = []
        for region in manager_regions:
            manager_reps.extend(region_to_reps.get(region, []))
        manager_reps = list(set(manager_reps))
        
        rep_to_bricks = self.lookups.get("rep_to_bricks", {})
        manager_bricks = []
        for rep in manager_reps:
            manager_bricks.extend(rep_to_bricks.get(rep, []))
        manager_bricks = list(set(manager_bricks))
        
        brick_to_pharmacies = self.lookups.get("brick_to_pharmacies", {})
        manager_pharmacies = []
        for brick in manager_bricks:
            manager_pharmacies.extend(brick_to_pharmacies.get(brick, []))
        manager_pharmacies = list(set(manager_pharmacies))
        
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

def get_current_user(session_user: Optional[str] = Cookie(None)) -> str:
    """Current user'ı session'dan al, yoksa login'e yönlendir"""
    if not session_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    auth_manager = AuthManager()
    if not auth_manager.authenticate_manager(session_user):
        raise HTTPException(status_code=401, detail="Invalid user")
    
    return session_user

def login_required(current_user: str = Depends(get_current_user)) -> str:
    """Login zorunlu decorator'ı"""
    return current_user