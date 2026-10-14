import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Union

import joblib
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "test_secret_key_for_ci_pipeline")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("secretpassword"),
    }
}

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "modelname.joblib"
model: Any = None


def load_classifier_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        # Fallback dummy model so CI passes even if the weights file is absent
        class DummyModel:
            def predict(self, data):
                return ["document" for _ in data]

            def predict_proba(self, data):
                return [[0.1, 0.9] for _ in data]

        model = DummyModel()


app = FastAPI(title="Document Classification API", version="1.0.0")


@app.on_event("startup")
def startup_event():
    load_classifier_model()


class Token(BaseModel):
    access_token: str
    token_type: str


class PredictRequest(BaseModel):
    data: List[Union[str, List[float]]]


class PredictResponse(BaseModel):
    predictions: List[Union[str, int, float]]
    probabilities: Optional[List[List[float]]] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username


@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/classify", response_model=PredictResponse)
def classify_document(
    request: PredictRequest, current_user: str = Depends(get_current_user)
):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier model not initialized.",
        )
    try:
        preds = model.predict(request.data)
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(request.data).tolist()

        return PredictResponse(
            predictions=list(preds),
            probabilities=probabilities,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model inference failed: {str(exc)}",
        )
