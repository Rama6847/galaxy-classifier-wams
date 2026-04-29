from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import uuid

import consul
import socket

# ========== CONFIGURATION ==========
SECRET_KEY = "mon_secret_super_long_pour_les_tokens_jwt_123456789"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ========== INITIALISATION ==========
app = FastAPI(title="Auth Service")


def register_to_consul(service_name: str, port: int):
    try:
        c = consul.Consul(host='localhost', port=8500)
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        c.agent.service.register(
            name=service_name,
            service_id=f"{service_name}-{port}",
            address=ip_address,
            port=port,
            check=consul.Check.http(f"http://localhost:{port}/health", interval="10s")
        )
        print(f"✅ Service {service_name} enregistré dans Consul sur le port {port}")
    except Exception as e:
        print(f"⚠️ Impossible de s'enregistrer dans Consul: {e}")

register_to_consul("auth-service", 8001)

# ========== CORS (pour que le navigateur puisse appeler ce service) ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
security = HTTPBearer()

# ========== BASE DE DONNEES EN MEMOIRE ==========
users_db = {}

# ========== MODELES ==========
class UserRegister(BaseModel):
    email: str
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# ========== FONCTIONS ==========
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_token(username: str, user_id: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data = {
        "sub": username,
        "user_id": user_id,
        "role": role,
        "exp": expire
    }
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def find_user_by_username(username: str) -> Optional[Dict]:
    for user_id, user in users_db.items():
        if user["username"] == username:
            return user
    return None

def find_user_by_email(email: str) -> Optional[Dict]:
    for user_id, user in users_db.items():
        if user["email"] == email:
            return user
    return None

# ========== ENDPOINTS ==========
@app.get("/")
def root():
    return {"service": "Auth Service", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/register")
async def register(user: UserRegister):
    if find_user_by_username(user.username):
        raise HTTPException(status_code=400, detail="Ce nom d'utilisateur existe déjà")
    
    if find_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Cet email existe déjà")
    
    user_id = str(uuid.uuid4())
    hashed_pw = hash_password(user.password)
    
    role = "admin" if len(users_db) == 0 else "user"
    
    users_db[user_id] = {
        "user_id": user_id,
        "email": user.email,
        "username": user.username,
        "hashed_password": hashed_pw,
        "role": role
    }
    
    token = create_token(user.username, user_id, role)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id,
        "username": user.username,
        "role": role
    }

@app.post("/login")
async def login(user: UserLogin):
    db_user = find_user_by_username(user.username)
    
    if not db_user:
        raise HTTPException(status_code=401, detail="Nom d'utilisateur ou mot de passe incorrect")
    
    if not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Nom d'utilisateur ou mot de passe incorrect")
    
    token = create_token(db_user["username"], db_user["user_id"], db_user["role"])
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": db_user["user_id"],
        "username": db_user["username"],
        "role": db_user["role"]
    }

@app.get("/verify")
async def verify(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "valid": True,
            "username": payload.get("sub"),
            "user_id": payload.get("user_id"),
            "role": payload.get("role")
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré")

@app.get("/users")
async def list_users():
    safe_list = []
    for user_id, user in users_db.items():
        safe_list.append({
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        })
    return {"users": safe_list}