# llm-lab

Código de la sección 15 del libro: *Laboratorio de modelos en el editor*.
La documentación está en `docs/15-laboratorio-de-modelos-en-el-editor/`.

| Directorio | Sección | Qué es |
|---|---|---|
| `continue/` | 15.1 | Copia versionada del `config.yaml`, **sin claves** |
| `mi-modelo/mi_modelo_server.py` | 15.2 | Shim compatible con la API de OpenAI |
| `proxy/` | 15.3, 15.4 | Proxy espía y política de enrutado |
| `bench/` | 15.5 | Banco de pruebas multi-modelo |
| `mi-modelo/train_cervantes.py` | 15.6 | Transformer por caracteres |

## Arranque rápido

```bash
cp .env.example .env          # y poner la clave de OpenRouter
docker compose up -d proxy
curl http://127.0.0.1:8080/salud
```

Sin Docker:

```bash
cd proxy && python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

## Avisos

- Todo escucha en `127.0.0.1`. No exponer: el proxy lleva la clave dentro.
- `proxy/logs/` contiene el código fuente completo que se envía a los modelos.
- Los identificadores de modelo caducan. Ver 15.7, «Mantenimiento».
