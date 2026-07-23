# Instalação – Reconhecimento Facial API

API FastAPI + **InsightFace** (`buffalo_l` / ArcFace 512d) com **onnxruntime**.
Não usa mais `face_recognition` / dlib.

---

## Requisitos

- Python 3.10+
- ~500 MB de espaço (modelo baixado na primeira execução)

---

## Instalação

```bash
cd /Users/leorodrigues/projects/ppgo-biometria

python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

Se houver erro de SSL no pip:

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

## Variáveis de ambiente (opcional)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `BIOMETRIA_API_KEY` | (vazio) | Se definido, exige header `X-API-Key` |
| `CORS_ORIGINS` | `*` | Origins permitidos, separados por vírgula |
| `MAX_UPLOAD_BYTES` | `8388608` (8 MB) | Tamanho máximo do upload |
| `MIN_FACE_CONFIDENCE` | `0.70` | Confiança mínima da detecção |
| `MIN_FACE_AREA_RATIO` | `0.05` | Área mínima do rosto vs imagem (5%) |

**Produção:** defina `BIOMETRIA_API_KEY` e `CORS_ORIGINS` com os origins do backend/app.

Exemplo:

```bash
export BIOMETRIA_API_KEY="sua-chave-secreta"
export CORS_ORIGINS="https://api.exemplo.gov.br,https://app.exemplo.gov.br"
```

---

## GPU (opcional)

Por padrão usa CPU (`CPUExecutionProvider`). Com CUDA instalado, altere em `main.py` para `CUDAExecutionProvider` e instale `onnxruntime-gpu`.
