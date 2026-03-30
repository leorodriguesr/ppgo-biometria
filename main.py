"""
API de Reconhecimento Facial - FastAPI + InsightFace (512d)
Gera embeddings faciais modernos (ArcFace 512 dimensões) sem persistência.
O backend principal decide quando persistir biometria.
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import insightface
from insightface.app import FaceAnalysis


# ==============================
# Configurações
# ==============================

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ERROR_CODE_NO_FACE = "NO_FACE_DETECTED"
ERROR_CODE_MULTIPLE_FACES = "MULTIPLE_FACES"
ERROR_CODE_INVALID_IMAGE = "INVALID_IMAGE"
ERROR_CODE_INVALID_FILE = "INVALID_FILE"


# ==============================
# Inicialização do modelo
# ==============================

face_app: Optional[FaceAnalysis] = None


def get_face_app() -> FaceAnalysis:
    global face_app
    if face_app is None:
        face_app = FaceAnalysis(
            name="buffalo_l",  # 512 dimensões
            providers=["CPUExecutionProvider"],  # Trocar para CUDAExecutionProvider se tiver GPU
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app


# ==============================
# FastAPI Setup
# ==============================

app = FastAPI(
    title="API Reconhecimento Facial",
    description="Geração de embeddings faciais 512d com InsightFace (ArcFace)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
                "detail": f"Use: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
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
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_INVALID_FILE,
                "message": "Arquivo vazio",
                "detail": "O arquivo enviado está vazio.",
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


# ==============================
# Response Model
# ==============================

class GenerateEmbeddingResponse(BaseModel):
    embedding: list[float]
    modelName: str
    modelVersion: str
    embeddingSize: int
    faceConfidence: float


# ==============================
# Endpoint principal
# ==============================

@app.post("/generate-embedding", response_model=GenerateEmbeddingResponse)
async def generate_embedding(
    file: UploadFile = File(..., description="Imagem do rosto para gerar embedding"),
):
    validate_image_upload(file)
    image_bytes = await read_image_bytes(file)
    img_np = load_image(image_bytes)

    face_app = get_face_app()
    faces = face_app.get(img_np)

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

    embedding = face.embedding  # 512 dimensões
    confidence = float(face.det_score)

    return GenerateEmbeddingResponse(
        embedding=embedding.tolist(),
        modelName="ArcFace",
        modelVersion="buffalo_l",
        embeddingSize=len(embedding),
        faceConfidence=confidence,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": "buffalo_l (512d)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)





# """
# API de Reconhecimento Facial - FastAPI + face_recognition
# Gera embeddings faciais sem persistência. O backend principal decide quando persistir biometria.
# """

# import io
# from pathlib import Path

# import numpy as np
# from PIL import Image, ImageOps
# from fastapi import FastAPI, File, HTTPException, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# def get_face_lib():
#     try:
#         import face_recognition
#         return face_recognition
#     except ImportError as e:
#         raise HTTPException(
#             status_code=503,
#             detail="Biblioteca face_recognition não disponível. Instale: pip install face_recognition (requer cmake e dlib)."
#         ) from e


# app = FastAPI(
#     title="API Reconhecimento Facial",
#     description="Geração de embeddings faciais com face_recognition (sem persistência)",
#     version="1.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Códigos de erro para o frontend tratar
# ERROR_CODE_NO_FACE = "NO_FACE_DETECTED"
# ERROR_CODE_INVALID_IMAGE = "INVALID_IMAGE"
# ERROR_CODE_ENCODING_FAILED = "ENCODING_FAILED"
# ERROR_CODE_INVALID_FILE = "INVALID_FILE"


# def _error_detail(code: str, message: str, detail: str) -> dict:
#     """Resposta de erro padronizada: code, message (curto), detail (motivo completo)."""
#     return {"code": code, "message": message, "detail": detail}


# def validate_image_upload(file: UploadFile) -> None:
#     # React Native / Expo às vezes não envia filename; usar fallback
#     filename = file.filename or "face.jpg"
#     ext = Path(filename).suffix.lower()
#     if ext not in ALLOWED_IMAGE_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=_error_detail(
#                 ERROR_CODE_INVALID_FILE,
#                 "Formato de imagem não suportado",
#                 f"Use um dos formatos: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
#             ),
#         )
#     if file.content_type and not file.content_type.startswith("image/"):
#         raise HTTPException(
#             status_code=400,
#             detail=_error_detail(
#                 ERROR_CODE_INVALID_FILE,
#                 "Arquivo inválido",
#                 "O arquivo não parece ser uma imagem.",
#             ),
#         )


# async def read_image_bytes(file: UploadFile) -> bytes:
#     try:
#         content = await file.read()
#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=_error_detail(
#                 ERROR_CODE_INVALID_FILE,
#                 "Erro ao ler o arquivo",
#                 str(e),
#             ),
#         ) from e
#     if not content:
#         raise HTTPException(
#             status_code=400,
#             detail=_error_detail(
#                 ERROR_CODE_INVALID_FILE,
#                 "Arquivo vazio",
#                 "O arquivo enviado está vazio.",
#             ),
#         )
#     return content


# def _load_image_correct_orientation(image_bytes: bytes) -> np.ndarray:
#     """Carrega a imagem e aplica correção de orientação EXIF (comum em fotos de celular)."""
#     img = Image.open(io.BytesIO(image_bytes))
#     img = ImageOps.exif_transpose(img)  # corrige rotação de fotos de celular
#     if img.mode != "RGB":
#         img = img.convert("RGB")
#     return np.array(img)


# def load_image_and_detect_face(image_bytes: bytes):
#     """Retorna (encoding, None, None) em sucesso ou (None, mensagem, codigo) em erro."""
#     import face_recognition
#     try:
#         img_np = _load_image_correct_orientation(image_bytes)
#     except Exception as e:
#         return None, f"Imagem inválida ou corrompida: {str(e)}", ERROR_CODE_INVALID_IMAGE

#     face_locations = face_recognition.face_locations(
#         img_np, model="hog", number_of_times_to_upsample=2
#     )
#     if not face_locations:
#         return (
#             None,
#             "Nenhum rosto detectado na imagem. Envie uma foto com o rosto visível, de frente e bem iluminado.",
#             ERROR_CODE_NO_FACE,
#         )
#     if len(face_locations) > 1:
#         def area(box):
#             top, right, bottom, left = box
#             return (bottom - top) * (right - left)
#         face_locations = [max(face_locations, key=area)]

#     encodings = face_recognition.face_encodings(img_np, known_face_locations=face_locations)
#     if not encodings:
#         return None, "Não foi possível gerar o encoding do rosto.", ERROR_CODE_ENCODING_FAILED
#     return encodings[0], None, None


# class GenerateEmbeddingResponse(BaseModel):
#     embedding: list[float]
#     modelVersion: str
#     embeddingSize: int


# def _embedding_error_message(code: str) -> str:
#     """Mensagens curtas para o endpoint /generate-embedding."""
#     if code == ERROR_CODE_NO_FACE:
#         return "Nenhum rosto detectado na imagem"
#     if code == ERROR_CODE_ENCODING_FAILED:
#         return "Não foi possível gerar o embedding"
#     if code == ERROR_CODE_INVALID_IMAGE:
#         return "Imagem inválida ou corrompida"
#     return "Erro ao processar a imagem"


# @app.post("/generate-embedding", response_model=GenerateEmbeddingResponse)
# async def generate_embedding(
#     file: UploadFile = File(..., description="Imagem do rosto para gerar embedding"),
# ):
#     """
#     Gera o embedding facial da imagem sem persistir nada.
#     Usado antes do cadastro definitivo; o backend principal decide quando persistir biometria.
#     """
#     validate_image_upload(file)
#     image_bytes = await read_image_bytes(file)

#     get_face_lib()
#     encoding, error_msg, error_code = load_image_and_detect_face(image_bytes)
#     if error_msg is not None:
#         raise HTTPException(
#             status_code=400,
#             detail={
#                 "code": error_code,
#                 "message": _embedding_error_message(error_code),
#                 "detail": error_msg,
#             },
#         )

#     embedding_list = encoding.tolist()
#     return GenerateEmbeddingResponse(
#         embedding=embedding_list,
#         modelVersion="hog",
#         embeddingSize=len(embedding_list),
#     )


# @app.get("/health")
# async def health():
#     return {"status": "ok"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
