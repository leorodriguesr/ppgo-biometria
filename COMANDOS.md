# Comandos para rodar a aplicação

## Primeira vez (instalação)

```bash
cd /Users/leorodrigues/projects/ppgo-biometria

python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

Detalhes e variáveis de ambiente: veja **INSTALL.md**.

---

## Rodar o servidor

```bash
source venv/bin/activate

# export BIOMETRIA_API_KEY="sua-chave"
# export API_DOCS_ENABLED=true

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

A API fica disponível em:

- **Local:** http://localhost:8000
- **Documentação:** http://localhost:8000/docs com `API_DOCS_ENABLED=true`
- **Health:** http://localhost:8000/health

Para o app no celular/emulador acessar, use o IP do computador na rede (ex.: `http://192.168.1.x:8000`).

### Exemplo com API key

```bash
curl -X POST http://localhost:8000/generate-embedding \
  -H "X-API-Key: sua-chave" \
  -F "file=@rosto.jpg"
```

---

## Parar o servidor

No terminal onde o uvicorn está rodando: **Ctrl+C**.
