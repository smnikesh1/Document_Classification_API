from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

from app.image_classifier import classify_image_bytes

# --------------------------------------------------
# JWT & Security Configuration
# --------------------------------------------------
SECRET_KEY = "super-secret-key-change-in-production-replace-with-env"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Fixed user database: Key matches the username ("beejaai")
FAKE_USERS_DB = {
    "beejaai": {
        "username": "beejaai",
        "password": "sairam123",
    }
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --------------------------------------------------
# FastAPI App Initialization
# --------------------------------------------------
app = FastAPI(
    title="Document Classification API with JWT",
    description="Classifies identity documents with OAuth2/JWT security.",
    version="1.1.0",
)

# --------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str


class ClassificationResponse(BaseModel):
    filename: str
    classified: str
    predicted_category: str
    confidence: float
    all_scores: Optional[Dict[str, float]] = None
    message: str


# --------------------------------------------------
# Authentication Helper Functions
# --------------------------------------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in FAKE_USERS_DB:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "service": "document-classification-api"}


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/classify", response_model=ClassificationResponse)
async def classify_document(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),  # Protected endpoint
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    try:
        result = classify_image_bytes(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return {
        "filename": file.filename,
        "classified": result["classified"],
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
        "all_scores": result.get("all_scores"),
        "message": result["message"],
    }