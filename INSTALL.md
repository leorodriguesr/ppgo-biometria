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
| `SSO_TOKEN_CACHE_TTL` | `30` | Segundos de cache de token já validado |
| `CORS_ORIGINS` | (vazio) | Origens permitidas, separadas por vírgula. `*` é ignorado |
| `RATE_LIMIT_IP` | `30` | Máximo de POST biométrico por IP na janela. `0` desliga |
| `RATE_LIMIT_TOKEN` | `20` | Máximo de POST biométrico por token na janela. `0` desliga |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Duração da janela do rate limit |
| `AUTH_FAILURE_LIMIT` | `10` | Falhas de autenticação antes do bloqueio. `0` desliga |
| `AUTH_FAILURE_BLOCK_SECONDS` | `60` | Bloqueio inicial após falhas; dobra a cada ciclo, até 900s |
| `API_DOCS_ENABLED` | `false` | Publica `/docs` e `/openapi.json` |
| `MAX_UPLOAD_BYTES` | `8388608` (8 MB) | Tamanho máximo do upload |
| `MIN_FACE_CONFIDENCE` | `0.70` | Confiança mínima da detecção |
| `MIN_FACE_AREA_RATIO` | `0.05` | Área mínima do rosto vs imagem (5%) |

SSO é sempre o token do login (`Authorization: Bearer`). A API tenta o SSO de produção e, se não aceitar, o de homologação. Não precisa de variável no Kubernetes.

```bash
export REQUIRE_SSO=true
```

Para testar a API **sem** SSO na máquina local:

```bash
export REQUIRE_SSO=false
```

---

## GPU (opcional)

Por padrão usa CPU (`CPUExecutionProvider`). Com CUDA instalado, altere em `main.py` para `CUDAExecutionProvider` e instale `onnxruntime-gpu`.
