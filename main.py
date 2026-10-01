"""
API de Reconhecimento Facial - FastAPI + InsightFace (512d)
Gera embeddings faciais modernos (ArcFace 512 dimensões) sem persistência.
O backend principal decide quando persistir biometria.
"""

import asyncio
import hashlib
import io
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from insightface.app import FaceAnalysis


# ==============================
# Configurações
# ==============================

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))  # 12 MB
MIN_FACE_CONFIDENCE = float(os.getenv("MIN_FACE_CONFIDENCE", "0.70"))
MIN_FACE_AREA_RATIO = float(os.getenv("MIN_FACE_AREA_RATIO", "0.05"))  # 5% da imagem
API_KEY = os.getenv("BIOMETRIA_API_KEY", "").strip()
SSO_PROD_BASE_URL = "https://ssows.ssp.go.gov.br/"
SSO_HOMO_BASE_URL = "https://ssows-h.ssp.go.gov.br/"
SSO_VALIDATE_URLS = (SSO_PROD_BASE_URL, SSO_HOMO_BASE_URL)
SSO_VALIDATE_TIMEOUT = float(os.getenv("SSO_VALIDATE_TIMEOUT", "8"))
SSO_TOKEN_CACHE_TTL = float(os.getenv("SSO_TOKEN_CACHE_TTL", "30"))


def parse_cors_origins(raw: Optional[str] = None) -> list[str]:
    source = os.getenv("CORS_ORIGINS", "") if raw is None else raw
    origins: list[str] = []
    for origin in source.split(","):
        value = origin.strip()
        if not value or value == "*":
            # ConfigMap global às vezes traz '*'; não derruba o processo.
            continue
        origins.append(value.rstrip("/"))
    return origins

ERROR_CODE_NO_FACE = "NO_FACE_DETECTED"
ERROR_CODE_MULTIPLE_FACES = "MULTIPLE_FACES"
ERROR_CODE_INVALID_IMAGE = "INVALID_IMAGE"
ERROR_CODE_INVALID_FILE = "INVALID_FILE"
ERROR_CODE_LOW_FACE_CONFIDENCE = "LOW_FACE_CONFIDENCE"
ERROR_CODE_FACE_TOO_SMALL = "FACE_TOO_SMALL"
ERROR_CODE_UNAUTHORIZED = "UNAUTHORIZED"
ERROR_CODE_RATE_LIMITED = "RATE_LIMITED"

RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_IP = int(os.getenv("RATE_LIMIT_IP", "30"))
RATE_LIMIT_TOKEN = int(os.getenv("RATE_LIMIT_TOKEN", "20"))
AUTH_FAILURE_LIMIT = int(os.getenv("AUTH_FAILURE_LIMIT", "10"))
AUTH_FAILURE_BLOCK_SECONDS = float(os.getenv("AUTH_FAILURE_BLOCK_SECONDS", "60"))
RATE_LIMITED_PATHS = {
    "/generate-embedding",
    "/validate-photo-quality",
    "/validate-detainee-photo",
}
_rate_hits: dict[str, list[float]] = {}
_rate_lock = threading.Lock()
_auth_failures: dict[str, tuple[int, float]] = {}
_auth_lock = threading.Lock()
ERROR_CODE_MODEL_NOT_READY = "MODEL_NOT_READY"
ERROR_CODE_TOO_DARK = "TOO_DARK"
ERROR_CODE_TOO_BRIGHT = "TOO_BRIGHT"
ERROR_CODE_TOO_BLURRY = "TOO_BLURRY"
ERROR_CODE_WRONG_POSE = "WRONG_POSE"
ERROR_CODE_INVALID_POSE = "INVALID_POSE"

MIN_BRIGHTNESS = float(os.getenv("MIN_BRIGHTNESS", "45"))   # 0–255
MAX_BRIGHTNESS = float(os.getenv("MAX_BRIGHTNESS", "220"))  # 0–255
MIN_BLUR_VARIANCE = float(os.getenv("MIN_BLUR_VARIANCE", "35"))
BLUR_EVAL_MAX_SIDE = int(os.getenv("BLUR_EVAL_MAX_SIDE", "640"))
MIN_FACE_WIDTH_RATIO = float(os.getenv("MIN_FACE_WIDTH_RATIO", "0.15"))
MIN_FACE_WIDTH_RATIO_PROFILE = float(os.getenv("MIN_FACE_WIDTH_RATIO_PROFILE", "0.12"))
MIN_FACE_CONFIDENCE_PROFILE = float(os.getenv("MIN_FACE_CONFIDENCE_PROFILE", "0.55"))
YAW_FRONT_MAX = float(os.getenv("YAW_FRONT_MAX", "20"))       # |yaw| <= isto = frente
YAW_PROFILE_MIN = float(os.getenv("YAW_PROFILE_MIN", "30"))   # |yaw| >= isto = perfil

POSE_FRONT = "front"
POSE_LEFT = "left_profile"
POSE_RIGHT = "right_profile"
POSE_ALIASES = {
    "front": POSE_FRONT,
    "frente": POSE_FRONT,
    "left": POSE_LEFT,
    "left_profile": POSE_LEFT,
    "perfil_esquerdo": POSE_LEFT,
    "right": POSE_RIGHT,
    "right_profile": POSE_RIGHT,
    "perfil_direito": POSE_RIGHT,
}

MODEL_NAME = "ArcFace"
MODEL_VERSION = "buffalo_l"
EMBEDDING_SIZE = 512


# ==============================
# Inicialização do modelo
# ==============================

face_app: Optional[FaceAnalysis] = None
model_ready = False


def load_face_app() -> FaceAnalysis:
    app_instance = FaceAnalysis(
        name=MODEL_VERSION,
        providers=["CPUExecutionProvider"],  # Trocar para CUDAExecutionProvider se tiver GPU
    )
    app_instance.prepare(ctx_id=0, det_size=(640, 640))
    return app_instance


def get_face_app() -> FaceAnalysis:
    if face_app is None or not model_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "code": ERROR_CODE_MODEL_NOT_READY,
                "message": "Modelo não pronto",
                "detail": "O modelo de reconhecimento facial ainda está carregando.",
            },
        )
    return face_app


def _load_face_app_in_background() -> None:
    global face_app, model_ready
    try:
        loaded = load_face_app()
        face_app = loaded
        model_ready = True
    except Exception:
        model_ready = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global face_app, model_ready
    model_ready = False
    loader = threading.Thread(target=_load_face_app_in_background, daemon=True)
    loader.start()
    yield
    model_ready = False
    face_app = None


# ==============================
# FastAPI Setup
# ==============================

def api_docs_enabled() -> bool:
    return os.getenv("API_DOCS_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


_docs_enabled = api_docs_enabled()
app = FastAPI(
    title="API Reconhecimento Facial",
    description="Geração de embeddings faciais 512d com InsightFace (ArcFace)",
    version="2.2.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

def configure_cors(application: FastAPI, origins: list[str]) -> None:
    if not origins:
        return
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "token"],
    )


configure_cors(app, parse_cors_origins())


# ==============================
# Auth
# ==============================

_sso_token_cache: dict[str, float] = {}


def sso_required() -> bool:
    return os.getenv("REQUIRE_SSO", "true").strip().lower() not in {"0", "false", "no", "off"}


def extract_request_token(authorization: Optional[str], token_header: Optional[str]) -> str:
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    return (token_header or "").strip()


def _token_cache_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def client_ip(request: Request) -> str:
    real_ip = (request.headers.get("x-real-ip") or "").strip()
    if real_ip:
        return real_ip
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def rate_limit_allows(key: str, limit: int, now: Optional[float] = None) -> bool:
    if limit <= 0:
        return True
    current = time.monotonic() if now is None else now
    with _rate_lock:
        hits = [
            stamp
            for stamp in _rate_hits.get(key, [])
            if current - stamp < RATE_LIMIT_WINDOW_SECONDS
        ]
        if len(hits) >= limit:
            _rate_hits[key] = hits
            return False
        hits.append(current)
        _rate_hits[key] = hits
        return True


def auth_block_remaining(ip: str, now: Optional[float] = None) -> int:
    if not ip:
        return 0
    current = time.monotonic() if now is None else now
    with _auth_lock:
        _count, until = _auth_failures.get(ip, (0, 0.0))
        if until > current:
            return max(1, int(until - current))
        return 0


def register_auth_failure(ip: str, now: Optional[float] = None) -> None:
    if AUTH_FAILURE_LIMIT <= 0 or not ip:
        return
    current = time.monotonic() if now is None else now
    with _auth_lock:
        count, until = _auth_failures.get(ip, (0, 0.0))
        if until > current:
            return
        count += 1
        if count % AUTH_FAILURE_LIMIT == 0:
            step = count // AUTH_FAILURE_LIMIT
            delay = min(AUTH_FAILURE_BLOCK_SECONDS * (2 ** (step - 1)), 900)
            until = current + delay
        _auth_failures[ip] = (count, until)


def clear_auth_failures(ip: str) -> None:
    if not ip:
        return
    with _auth_lock:
        _auth_failures.pop(ip, None)


def rate_limit_response() -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": {
                "code": ERROR_CODE_RATE_LIMITED,
                "message": "Muitas requisições",
                "detail": "Aguarde antes de tentar novamente.",
            }
        },
        headers={"Retry-After": str(max(1, int(RATE_LIMIT_WINDOW_SECONDS)))},
    )


@app.middleware("http")
async def limit_biometric_requests(request: Request, call_next):
    if request.method == "POST" and request.url.path in RATE_LIMITED_PATHS:
        if not rate_limit_allows(f"ip:{client_ip(request)}", RATE_LIMIT_IP):
            return rate_limit_response()
        raw = extract_request_token(
            request.headers.get("authorization"),
            request.headers.get("token"),
        )
        if raw and not rate_limit_allows(f"token:{_token_cache_key(raw)}", RATE_LIMIT_TOKEN):
            return rate_limit_response()
    return await call_next(request)


def _validate_against_sso(base_url: str, raw_token: str) -> bool:
    url = f"{base_url}validate?token={urllib.parse.quote(raw_token, safe='-._~')}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=SSO_VALIDATE_TIMEOUT) as response:
            if int(getattr(response, "status", 200)) != 200:
                return False
            content_type = str(response.headers.get("Content-Type") or "")
            if "json" not in content_type.lower():
                return False
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
        return False
    if not isinstance(payload, dict):
        return False
    servidor = payload.get("servidor")
    if isinstance(servidor, dict) and (servidor.get("cpf") or servidor.get("nome")):
        return True
    return bool(payload.get("cpf") or payload.get("nome"))


def call_sso_validate(raw_token: str) -> bool:
    """Aceita token válido no SSO de produção ou no de homologação."""
    for base in SSO_VALIDATE_URLS:
        if _validate_against_sso(base, raw_token):
            return True
    return False


def sso_token_is_valid(raw_token: str) -> bool:
    key = _token_cache_key(raw_token)
    now = time.monotonic()
    expires = _sso_token_cache.get(key)
    if expires is not None and expires > now:
        return True
    if not call_sso_validate(raw_token):
        _sso_token_cache.pop(key, None)
        return False
    _sso_token_cache[key] = now + SSO_TOKEN_CACHE_TTL
    return True


def raise_unauthorized() -> None:
    raise HTTPException(
        status_code=401,
        detail={
            "code": ERROR_CODE_UNAUTHORIZED,
            "message": "Acesso não autorizado. Token inválido ou ausente.",
            "detail": "Faça login novamente para continuar.",
        },
    )


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "code": ERROR_CODE_UNAUTHORIZED,
                "message": "Não autorizado",
                "detail": "Informe um X-API-Key válido.",
            },
        )


async def enforce_sso_token(
    authorization: Optional[str],
    token_header: Optional[str],
    ip: str = "",
) -> None:
    if not sso_required():
        return
    remaining = auth_block_remaining(ip)
    if remaining:
        raise HTTPException(
            status_code=429,
            detail={
                "code": ERROR_CODE_RATE_LIMITED,
                "message": "Muitas tentativas de autenticação",
                "detail": "Aguarde antes de tentar novamente.",
            },
            headers={"Retry-After": str(remaining)},
        )
    raw = extract_request_token(authorization, token_header)
    if not raw or not await asyncio.to_thread(sso_token_is_valid, raw):
        register_auth_failure(ip)
        raise_unauthorized()
    clear_auth_failures(ip)


async def require_sso_token(request: Request) -> None:
    await enforce_sso_token(
        request.headers.get("authorization"),
        request.headers.get("token"),
        client_ip(request),
    )


# ==============================
# Utilitários
# ==============================

def validate_image_upload(file: UploadFile) -> None:
    filename = file.filename or "face.jpg"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_INVALID_FILE,
                "message": "Formato de imagem não suportado",
                "detail": f"Use: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
            },
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_INVALID_FILE,
                "message": "Arquivo inválido",
                "detail": "O arquivo enviado não parece ser uma imagem.",
            },
        )


async def read_image_bytes(file: UploadFile) -> bytes:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_INVALID_FILE,
                "message": "Arquivo vazio",
                "detail": "O arquivo enviado está vazio.",
            },
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_INVALID_FILE,
                "message": "Arquivo muito grande",
                "detail": f"Tamanho máximo permitido: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            },
        )
    return content


def load_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_INVALID_IMAGE,
                "message": "Imagem inválida",
                "detail": str(e),
            },
        )


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_INVALID_IMAGE,
                "message": "Embedding inválido",
                "detail": "Não foi possível normalizar o embedding facial.",
            },
        )
    return embedding / norm


def face_area_ratio(face, image_shape: tuple[int, ...]) -> float:
    bbox = getattr(face, "bbox", None)
    if bbox is None or len(bbox) < 4:
        return 1.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    face_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = float(image_shape[0] * image_shape[1])
    if image_area <= 0:
        return 1.0
    return face_area / image_area


def face_width_ratio(face, image_shape: tuple[int, ...]) -> float:
    bbox = getattr(face, "bbox", None)
    if bbox is None or len(bbox) < 4:
        return 1.0
    x1, x2 = float(bbox[0]), float(bbox[2])
    img_w = float(image_shape[1])
    if img_w <= 0:
        return 1.0
    return abs(x2 - x1) / img_w


def crop_face_region(img_np: np.ndarray, bbox, pad_ratio: float = 0.25) -> np.ndarray:
    """Recorte do rosto com margem; cai para a imagem inteira se o bbox for inválido."""
    if bbox is None or len(bbox) < 4:
        return img_np
    h, w = img_np.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 = max(0, int(x1 - bw * pad_ratio))
    y1 = max(0, int(y1 - bh * pad_ratio))
    x2 = min(w, int(x2 + bw * pad_ratio))
    y2 = min(h, int(y2 + bh * pad_ratio))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return img_np
    return img_np[y1:y2, x1:x2]


def _resize_for_blur(img_np: np.ndarray) -> np.ndarray:
    """Normaliza resolução: Laplaciano 3x3 em foto 12MP subestima nitidez (pixels vizinhos iguais)."""
    h, w = img_np.shape[:2]
    max_side = max(h, w)
    if max_side <= BLUR_EVAL_MAX_SIDE or max_side < 3:
        return img_np
    scale = BLUR_EVAL_MAX_SIDE / float(max_side)
    new_w = max(3, int(round(w * scale)))
    new_h = max(3, int(round(h * scale)))
    return np.array(Image.fromarray(img_np).resize((new_w, new_h), Image.BILINEAR))


def blur_variance(img_np: np.ndarray) -> float:
    """Variância do Laplaciano (nitidez), em tamanho canônico. Valores baixos = borrado."""
    sample = _resize_for_blur(img_np)
    gray = (
        0.299 * sample[:, :, 0].astype(np.float64)
        + 0.587 * sample[:, :, 1].astype(np.float64)
        + 0.114 * sample[:, :, 2].astype(np.float64)
    )
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
    )
    return float(np.var(lap))


def estimate_yaw_degrees(face) -> Optional[float]:
    """
    Yaw aproximado via landmarks 5 pontos (olho esq, olho dir, nariz, ...).
    Positivo: nariz à direita do meio dos olhos (rosto virado para a esquerda do sujeito).
    """
    kps = getattr(face, "kps", None)
    if kps is None:
        return None
    pts = np.asarray(kps, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return None
    left_eye, right_eye, nose = pts[0], pts[1], pts[2]
    interocular = abs(float(right_eye[0] - left_eye[0]))
    if interocular < 1e-3:
        return None
    eye_mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
    offset = (float(nose[0]) - eye_mid_x) / interocular
    return float(np.clip(offset, -1.5, 1.5) * 60.0)


def normalize_pose(pose: str) -> str:
    key = (pose or "").strip().lower()
    normalized = POSE_ALIASES.get(key)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_INVALID_POSE,
                "message": "Pose inválida",
                "detail": "Use pose=front|left_profile|right_profile (ou left|right).",
            },
        )
    return normalized


def mean_brightness(img_np: np.ndarray) -> float:
    """Luminância média aproximada (0–255) em RGB."""
    # ITU-R BT.601
    r = img_np[:, :, 0].astype(np.float64)
    g = img_np[:, :, 1].astype(np.float64)
    b = img_np[:, :, 2].astype(np.float64)
    return float(np.mean(0.299 * r + 0.587 * g + 0.114 * b))


def assert_illumination_ok(img_np: np.ndarray) -> float:
    brightness = mean_brightness(img_np)
    if brightness < MIN_BRIGHTNESS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_TOO_DARK,
                "message": "Imagem muito escura",
                "detail": (
                    f"Brilho médio: {brightness:.1f}. "
                    f"Mínimo: {MIN_BRIGHTNESS:.0f}. Melhore a iluminação."
                ),
            },
        )
    if brightness > MAX_BRIGHTNESS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_TOO_BRIGHT,
                "message": "Imagem muito clara / estourada",
                "detail": (
                    f"Brilho médio: {brightness:.1f}. "
                    f"Máximo: {MAX_BRIGHTNESS:.0f}. Reduza a luz ou evite reflexo."
                ),
            },
        )
    return brightness


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def collect_pose_errors(expected_pose: str, yaw: Optional[float]) -> list[str]:
    if yaw is None:
        return [ERROR_CODE_WRONG_POSE]
    if expected_pose == POSE_FRONT:
        if abs(yaw) > YAW_FRONT_MAX:
            return [ERROR_CODE_WRONG_POSE]
    elif expected_pose == POSE_LEFT:
        if yaw > -YAW_PROFILE_MIN:
            return [ERROR_CODE_WRONG_POSE]
    elif expected_pose == POSE_RIGHT:
        if yaw < YAW_PROFILE_MIN:
            return [ERROR_CODE_WRONG_POSE]
    return []


class InspectedFace(BaseModel):
    embedding: list[float]
    yaw: Optional[float] = None
    brightness: float
    blur: float
    face_count: int
    face_width_ratio: Optional[float] = None
    face_confidence: Optional[float] = None
    pose: str
    errors: list[str]


def inspect_face_image(img_np: np.ndarray, expected_pose: str) -> InspectedFace:
    brightness = mean_brightness(img_np)
    errors: list[str] = []

    if brightness < MIN_BRIGHTNESS:
        errors.append(ERROR_CODE_TOO_DARK)
    elif brightness > MAX_BRIGHTNESS:
        errors.append(ERROR_CODE_TOO_BRIGHT)

    analysis = get_face_app()
    faces = analysis.get(img_np)
    face_count = len(faces) if faces else 0
    yaw: Optional[float] = None
    width_ratio: Optional[float] = None
    confidence: Optional[float] = None
    embedding: list[float] = []
    blur = 0.0

    if face_count == 0:
        errors.append(ERROR_CODE_NO_FACE)
        blur = blur_variance(img_np)
    elif face_count > 1:
        errors.append(ERROR_CODE_MULTIPLE_FACES)
        blur = blur_variance(img_np)
    else:
        face = faces[0]
        blur = blur_variance(crop_face_region(img_np, getattr(face, "bbox", None)))
        if blur < MIN_BLUR_VARIANCE:
            errors.append(ERROR_CODE_TOO_BLURRY)
        confidence = float(face.det_score)
        width_ratio = face_width_ratio(face, img_np.shape)
        yaw = estimate_yaw_degrees(face)
        min_conf = MIN_FACE_CONFIDENCE if expected_pose == POSE_FRONT else MIN_FACE_CONFIDENCE_PROFILE
        min_width = MIN_FACE_WIDTH_RATIO if expected_pose == POSE_FRONT else MIN_FACE_WIDTH_RATIO_PROFILE
        if confidence < min_conf:
            errors.append(ERROR_CODE_LOW_FACE_CONFIDENCE)
        if width_ratio < min_width:
            errors.append(ERROR_CODE_FACE_TOO_SMALL)
        errors.extend(collect_pose_errors(expected_pose, yaw))
        if not errors:
            embedding = l2_normalize(np.asarray(face.embedding, dtype=np.float64)).tolist()

    deduped: list[str] = []
    for code in errors:
        if code not in deduped:
            deduped.append(code)

    return InspectedFace(
        embedding=embedding,
        yaw=yaw,
        brightness=round(brightness, 2),
        blur=round(blur, 2),
        face_count=face_count,
        face_width_ratio=round(width_ratio, 4) if width_ratio is not None else None,
        face_confidence=round(confidence, 4) if confidence is not None else None,
        pose=expected_pose,
        errors=deduped,
    )


async def inspect_upload(file: UploadFile, expected_pose: str) -> InspectedFace:
    validate_image_upload(file)
    image_bytes = await read_image_bytes(file)
    img_np = load_image(image_bytes)
    return await asyncio.to_thread(inspect_face_image, img_np, expected_pose)


def raise_inspect_error(step: str, inspected: InspectedFace) -> None:
    if not inspected.errors:
        return
    code = inspected.errors[0]
    raise HTTPException(
        status_code=400,
        detail={
            "code": code,
            "message": f"Falha na captura ({step})",
            "detail": f"Etapa {step}: {', '.join(inspected.errors)}",
            "step": step,
            "errors": inspected.errors,
            "metrics": inspected.model_dump(exclude={"embedding"}),
        },
    )


# ==============================
# Response Model
# ==============================

class GenerateEmbeddingResponse(BaseModel):
    embedding: list[float]
    modelName: str
    modelVersion: str
    embeddingSize: int
    faceConfidence: float


class ValidatePhotoQualityResponse(BaseModel):
    ok: bool
    brightness: float
    minBrightness: float
    maxBrightness: float


class DetaineePhotoMetrics(BaseModel):
    face_count: int
    yaw: Optional[float] = None
    brightness: float
    blur: float
    face_width_ratio: Optional[float] = None
    face_confidence: Optional[float] = None
    pose: str


class ValidateDetaineePhotoResponse(BaseModel):
    valid: bool
    errors: list[str]
    metrics: DetaineePhotoMetrics


# ==============================
# Endpoint principal
# ==============================

@app.post(
    "/generate-embedding",
    response_model=GenerateEmbeddingResponse,
    dependencies=[Depends(require_api_key), Depends(require_sso_token)],
)
async def generate_embedding(
    file: UploadFile = File(..., description="Imagem do rosto para gerar embedding"),
):
    validate_image_upload(file)
    image_bytes = await read_image_bytes(file)
    img_np = load_image(image_bytes)
    assert_illumination_ok(img_np)

    analysis = get_face_app()
    faces = await asyncio.to_thread(analysis.get, img_np)

    if not faces:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_NO_FACE,
                "message": "Nenhum rosto detectado",
                "detail": "Envie uma foto frontal, bem iluminada e com apenas um rosto.",
            },
        )

    if len(faces) > 1:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_MULTIPLE_FACES,
                "message": "Mais de um rosto detectado",
                "detail": "Envie uma imagem contendo apenas um rosto.",
            },
        )

    face = faces[0]
    confidence = float(face.det_score)

    if confidence < MIN_FACE_CONFIDENCE:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_LOW_FACE_CONFIDENCE,
                "message": "Confiança do rosto abaixo do mínimo",
                "detail": (
                    f"Confiança detectada: {confidence:.3f}. "
                    f"Mínimo exigido: {MIN_FACE_CONFIDENCE:.2f}."
                ),
            },
        )

    ratio = face_area_ratio(face, img_np.shape)
    if ratio < MIN_FACE_AREA_RATIO:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_FACE_TOO_SMALL,
                "message": "Rosto muito pequeno na imagem",
                "detail": (
                    f"Área do rosto: {ratio:.1%} da imagem. "
                    f"Mínimo exigido: {MIN_FACE_AREA_RATIO:.0%}."
                ),
            },
        )

    embedding = l2_normalize(np.asarray(face.embedding, dtype=np.float64))

    return GenerateEmbeddingResponse(
        embedding=embedding.tolist(),
        modelName=MODEL_NAME,
        modelVersion=MODEL_VERSION,
        embeddingSize=len(embedding),
        faceConfidence=confidence,
    )


@app.post(
    "/validate-photo-quality",
    response_model=ValidatePhotoQualityResponse,
    dependencies=[Depends(require_api_key), Depends(require_sso_token)],
)
async def validate_photo_quality(
    file: UploadFile = File(..., description="Imagem para validar iluminação (sem exigir rosto)"),
):
    """Valida só iluminação — legado; preferir /validate-detainee-photo para fotos de busto."""
    validate_image_upload(file)
    image_bytes = await read_image_bytes(file)
    img_np = load_image(image_bytes)
    brightness = assert_illumination_ok(img_np)
    return ValidatePhotoQualityResponse(
        ok=True,
        brightness=brightness,
        minBrightness=MIN_BRIGHTNESS,
        maxBrightness=MAX_BRIGHTNESS,
    )


@app.post(
    "/validate-detainee-photo",
    response_model=ValidateDetaineePhotoResponse,
    dependencies=[Depends(require_api_key), Depends(require_sso_token)],
)
async def validate_detainee_photo(
    file: UploadFile = File(..., description="Foto de identificação (frente ou perfil)"),
    pose: str = Query(
        ...,
        description="Pose esperada: front | left_profile | right_profile (aliases: left, right)",
    ),
):
    """
    Valida foto de identificação sem gerar embedding:
    iluminação, nitidez, 1 rosto, tamanho mínimo e orientação (yaw) conforme a pose.
    """
    expected_pose = normalize_pose(pose)
    validate_image_upload(file)
    image_bytes = await read_image_bytes(file)
    img_np = load_image(image_bytes)

    brightness = mean_brightness(img_np)
    errors: list[str] = []

    if brightness < MIN_BRIGHTNESS:
        errors.append(ERROR_CODE_TOO_DARK)
    elif brightness > MAX_BRIGHTNESS:
        errors.append(ERROR_CODE_TOO_BRIGHT)

    analysis = get_face_app()
    faces = await asyncio.to_thread(analysis.get, img_np)
    face_count = len(faces) if faces else 0

    yaw: Optional[float] = None
    width_ratio: Optional[float] = None
    confidence: Optional[float] = None
    blur = 0.0

    if face_count == 0:
        errors.append(ERROR_CODE_NO_FACE)
        blur = blur_variance(img_np)
    elif face_count > 1:
        errors.append(ERROR_CODE_MULTIPLE_FACES)
        blur = blur_variance(img_np)
    else:
        face = faces[0]
        blur = blur_variance(crop_face_region(img_np, getattr(face, "bbox", None)))
        if blur < MIN_BLUR_VARIANCE:
            errors.append(ERROR_CODE_TOO_BLURRY)
        confidence = float(face.det_score)
        width_ratio = face_width_ratio(face, img_np.shape)
        yaw = estimate_yaw_degrees(face)

        min_conf = (
            MIN_FACE_CONFIDENCE
            if expected_pose == POSE_FRONT
            else MIN_FACE_CONFIDENCE_PROFILE
        )
        min_width = (
            MIN_FACE_WIDTH_RATIO
            if expected_pose == POSE_FRONT
            else MIN_FACE_WIDTH_RATIO_PROFILE
        )

        if confidence < min_conf:
            errors.append(ERROR_CODE_LOW_FACE_CONFIDENCE)
        if width_ratio < min_width:
            errors.append(ERROR_CODE_FACE_TOO_SMALL)

        if yaw is None:
            errors.append(ERROR_CODE_WRONG_POSE)
        elif expected_pose == POSE_FRONT:
            if abs(yaw) > YAW_FRONT_MAX:
                errors.append(ERROR_CODE_WRONG_POSE)
        elif expected_pose == POSE_LEFT:
            # Perfil esquerdo: yaw negativo (nariz à esquerda na imagem)
            if yaw > -YAW_PROFILE_MIN:
                errors.append(ERROR_CODE_WRONG_POSE)
        elif expected_pose == POSE_RIGHT:
            if yaw < YAW_PROFILE_MIN:
                errors.append(ERROR_CODE_WRONG_POSE)

    # remove duplicatas preservando ordem
    deduped: list[str] = []
    for code in errors:
        if code not in deduped:
            deduped.append(code)

    return ValidateDetaineePhotoResponse(
        valid=len(deduped) == 0,
        errors=deduped,
        metrics=DetaineePhotoMetrics(
            face_count=face_count,
            yaw=yaw,
            brightness=round(brightness, 2),
            blur=round(blur, 2),
            face_width_ratio=round(width_ratio, 4) if width_ratio is not None else None,
            face_confidence=round(confidence, 4) if confidence is not None else None,
            pose=expected_pose,
        ),
    )


def build_openapi() -> dict:
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    components = schema.setdefault("components", {})
    schemes = components.setdefault("securitySchemes", {})
    schemes["BearerAuth"] = {"type": "http", "scheme": "bearer"}
    schemes["TokenHeader"] = {"type": "apiKey", "in": "header", "name": "token"}
    for path, methods in schema.get("paths", {}).items():
        if path not in RATE_LIMITED_PATHS:
            continue
        for operation in methods.values():
            if not isinstance(operation, dict):
                continue
            operation["security"] = [{"BearerAuth": []}, {"TokenHeader": []}]
    return schema


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    app.openapi_schema = build_openapi()
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/health")
async def health():
    return {"status": "ok" if model_ready else "degraded"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
