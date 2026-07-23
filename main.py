"""
API de Reconhecimento Facial - FastAPI + InsightFace (512d)
Gera embeddings faciais modernos (ArcFace 512 dimensões) sem persistência.
O backend principal decide quando persistir biometria.
"""

import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from insightface.app import FaceAnalysis


# ==============================
# Configurações
# ==============================

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))  # 8 MB
MIN_FACE_CONFIDENCE = float(os.getenv("MIN_FACE_CONFIDENCE", "0.70"))
MIN_FACE_AREA_RATIO = float(os.getenv("MIN_FACE_AREA_RATIO", "0.05"))  # 5% da imagem
API_KEY = os.getenv("BIOMETRIA_API_KEY", "").strip()
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

ERROR_CODE_NO_FACE = "NO_FACE_DETECTED"
ERROR_CODE_MULTIPLE_FACES = "MULTIPLE_FACES"
ERROR_CODE_INVALID_IMAGE = "INVALID_IMAGE"
ERROR_CODE_INVALID_FILE = "INVALID_FILE"
ERROR_CODE_LOW_FACE_CONFIDENCE = "LOW_FACE_CONFIDENCE"
ERROR_CODE_FACE_TOO_SMALL = "FACE_TOO_SMALL"
ERROR_CODE_UNAUTHORIZED = "UNAUTHORIZED"
ERROR_CODE_MODEL_NOT_READY = "MODEL_NOT_READY"
ERROR_CODE_TOO_DARK = "TOO_DARK"
ERROR_CODE_TOO_BRIGHT = "TOO_BRIGHT"

MIN_BRIGHTNESS = float(os.getenv("MIN_BRIGHTNESS", "45"))   # 0–255
MAX_BRIGHTNESS = float(os.getenv("MAX_BRIGHTNESS", "220"))  # 0–255

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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global face_app, model_ready
    face_app = await asyncio.to_thread(load_face_app)
    model_ready = True
    yield
    model_ready = False
    face_app = None


# ==============================
# FastAPI Setup
# ==============================

app = FastAPI(
    title="API Reconhecimento Facial",
    description="Geração de embeddings faciais 512d com InsightFace (ArcFace)",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# Auth
# ==============================

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


# ==============================
# Endpoint principal
# ==============================

@app.post(
    "/generate-embedding",
    response_model=GenerateEmbeddingResponse,
    dependencies=[Depends(require_api_key)],
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
    dependencies=[Depends(require_api_key)],
)
async def validate_photo_quality(
    file: UploadFile = File(..., description="Imagem para validar iluminação (sem exigir rosto)"),
):
    """Valida só iluminação — útil para fotos de perfil onde a face pode não ser detectada."""
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


@app.get("/health")
async def health():
    return {
        "status": "ok" if model_ready else "starting",
        "model": f"{MODEL_VERSION} ({EMBEDDING_SIZE}d)",
        "modelReady": model_ready,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
