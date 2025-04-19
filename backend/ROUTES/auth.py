from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Dict
from auth import (
    create_access_token,
    verify_password,
    get_password_hash,
    decode_token
)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# In-memory user store (use DB in prod)
user_db: Dict[str, Dict] = {}

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    role: str  # e.g., "clinician", "admin"

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/auth/register")
async def register_user(user: UserRegister):
    if user.email in user_db:
        raise HTTPException(status_code=400, detail="User already exists")
    
    user_db[user.email] = {
        "email": user.email,
        "hashed_password": get_password_hash(user.password),
        "role": user.role
    }
    return {"message": "User registered successfully", "user": {"email": user.email, "role": user.role}}

@router.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = user_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = create_access_token({
        "sub": user["email"],
        "role": user["role"]
    })
    return {"access_token": access_token, "token_type": "bearer"}

# Dependency to get the current user from token
def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload

# Role-based check
def require_role(role: str):
    def dependency(user=Depends(get_current_user)):
        if user["role"] != role:
            raise HTTPException(status_code=403, detail="Not authorized")
        return user
    return dependency
