# Instalação – Reconhecimento Facial API

O pacote `face_recognition` depende do **dlib**, que normalmente exige **CMake** para compilar. No **macOS (Apple Silicon)** dá para usar wheel pré-compilado e evitar CMake.

---

## macOS (Apple Silicon) – sem CMake (recomendado)

Use o pacote **dlib-bin** (wheel pré-compilado). Com o venv ativado, na ordem:

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org dlib-bin
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org face_recognition --no-deps
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org Pillow face-recognition-models
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org "fastapi>=0.109.0" "uvicorn[standard]>=0.27.0" "python-multipart>=0.0.6" "numpy>=1.24.0"
```

Se não tiver erro de SSL, pode omitir `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

**Não** rode `pip install -r requirements.txt` depois disso, senão o pip tentará compilar o `dlib` de novo.

---

## Linux / Windows ou macOS com CMake

### Erro: "CMake is not installed on your system"

Siga **uma** das opções abaixo.

**Opção A – CMake via Homebrew (macOS)**

Se o Homebrew der erro de permissão, ajuste uma vez:

```bash
sudo chown -R $(whoami) /opt/homebrew /Users/leorodrigues/Library/Logs/Homebrew
brew install cmake
pip install -r requirements.txt
```

**Opção B – CMake via pip**

```bash
pip install cmake
pip install -r requirements.txt
```

(No build isolado do pip o wrapper do cmake pode falhar; nesse caso use Homebrew.)

---

### Erro de SSL no pip

Se aparecer `SSLError` ou `OSStatus -26276`:

- Atualizar certificados: **Applications → Python 3.x → Install Certificates.command**
- Ou use: `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <pacote>`
