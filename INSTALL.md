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
| `BIOMETRIA_API_KEY` | (vazio) | Se definido, exige também o header `X-API-Key` |
| `REQUIRE_SSO` | `true` | Se `true`, exige token SSO válido (`Authorization: Bearer` ou header `token`) |
| `SSO_BASE_URL` | `https://ssows.ssp.go.gov.br/` | Base do SSO para `/validate?token=` (homologação: `https://ssows-h.ssp.go.gov.br/`) |
| `SSO_TOKEN_CACHE_TTL` | `30` | Segundos de cache de token já validado |
| `CORS_ORIGINS` | `*` | Origins permitidos, separados por vírgula |
| `MAX_UPLOAD_BYTES` | `8388608` (8 MB) | Tamanho máximo do upload |
| `MIN_FACE_CONFIDENCE` | `0.70` | Confiança mínima da detecção |
| `MIN_FACE_AREA_RATIO` | `0.05` | Área mínima do rosto vs imagem (5%) |

**Produção:** deixe `REQUIRE_SSO=true` (padrão) e aponte `SSO_BASE_URL` para o SSO de produção. O app já envia o token do login. `/health` continua público.

```bash
export REQUIRE_SSO=true
export SSO_BASE_URL="https://ssows.ssp.go.gov.br/"
export CORS_ORIGINS="https://api.exemplo.gov.br,https://app.exemplo.gov.br"
```

Para testar a API **sem** SSO na máquina local:

```bash
export REQUIRE_SSO=false
```

---

## GPU (opcional)

Por padrão usa CPU (`CPUExecutionProvider`). Com CUDA instalado, altere em `main.py` para `CUDAExecutionProvider` e instale `onnxruntime-gpu`.
