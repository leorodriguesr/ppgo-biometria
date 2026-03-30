# Comandos para rodar a aplicação

## Primeira vez (instalação)

```bash
# Entrar na pasta do projeto
cd /Users/leorodrigues/projects/ppgo-goiaspen-app-back-cursor

# Criar ambiente virtual
python3 -m venv venv

# Ativar o ambiente virtual
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Instalar dependências (macOS Apple Silicon – sem compilar dlib)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org dlib-bin
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org face_recognition --no-deps
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org Pillow face-recognition-models
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org "fastapi>=0.109.0" "uvicorn[standard]>=0.27.0" "python-multipart>=0.0.6" "numpy>=1.24.0" "setuptools>=70,<75"
```

Se não tiver erro de SSL, use apenas: `pip install dlib-bin` (e assim por diante, sem `--trusted-host`).

Para outros sistemas ou uso de `requirements.txt`, veja **INSTALL.md**.

---

## Rodar o servidor

```bash
# Ativar o ambiente virtual (se ainda não estiver ativo)
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Subir a API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

A API ficará disponível em:

- **Local:** http://localhost:8000  
- **Documentação:** http://localhost:8000/docs  
- **Health:** http://localhost:8000/health  

Para o app no celular/emulador acessar, use o IP do seu computador na rede (ex.: `http://192.168.1.x:8000`).

---

## Parar o servidor

No terminal onde o uvicorn está rodando: **Ctrl+C**.
