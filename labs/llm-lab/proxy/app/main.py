"""Proxy espía: reenvía a OpenRouter y registra todo en JSONL."""
import json
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.router import elegir_modelo

ORIGEN = os.getenv("UPSTREAM", "https://openrouter.ai/api/v1")
CLAVE = os.environ["OPENROUTER_API_KEY"]
LOGS = Path(os.getenv("LOG_DIR", "logs"))
ENRUTAR = os.getenv("ENRUTAR", "0") == "1"      # Lab 2: apagado por defecto
LOGS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="llm-lab proxy")
cliente = httpx.AsyncClient(timeout=httpx.Timeout(300.0))


def registrar(evento: dict) -> None:
    dia = time.strftime("%Y-%m-%d")
    with open(LOGS / f"trafico-{dia}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")


def _sse_json(linea: str) -> dict | None:
    if not linea.startswith("data: ") or linea[6:].strip() == "[DONE]":
        return None
    try:
        return json.loads(linea[6:])
    except json.JSONDecodeError:
        return None


@app.post("/v1/chat/completions")
async def espiar(request: Request):
    cuerpo = await request.json()
    rid, t0 = uuid.uuid4().hex[:12], time.time()

    if ENRUTAR:
        pedido = cuerpo.get("model")
        cuerpo["model"] = elegir_modelo(cuerpo)
        registrar({"id": rid, "direccion": "ruta",
                   "pedido": pedido, "elegido": cuerpo["model"]})

    registrar({
        "id": rid,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "direccion": "peticion",
        "modelo": cuerpo.get("model"),
        "n_mensajes": len(cuerpo.get("messages", [])),
        "roles": [m.get("role") for m in cuerpo.get("messages", [])],
        "caracteres": sum(len(str(m.get("content", "")))
                          for m in cuerpo.get("messages", [])),
        "mensajes": cuerpo.get("messages"),      # el prompt COMPLETO
        "herramientas": cuerpo.get("tools"),
    })

    cabeceras = {"Authorization": f"Bearer {CLAVE}",
                 "Content-Type": "application/json"}

    if cuerpo.get("stream"):
        async def reenviar():
            trozos = []
            async with cliente.stream("POST", f"{ORIGEN}/chat/completions",
                                      json=cuerpo, headers=cabeceras) as r:
                async for linea in r.aiter_lines():
                    d = _sse_json(linea)
                    if d:
                        try:
                            c = d["choices"][0]["delta"].get("content")
                            if c:
                                trozos.append(c)
                        except (KeyError, IndexError):
                            pass
                    yield linea + "\n\n"
            registrar({"id": rid, "direccion": "respuesta",
                       "ms": int((time.time() - t0) * 1000),
                       "texto": "".join(trozos)})

        return StreamingResponse(reenviar(), media_type="text/event-stream")

    r = await cliente.post(f"{ORIGEN}/chat/completions",
                           json=cuerpo, headers=cabeceras)
    datos = r.json()
    registrar({"id": rid, "direccion": "respuesta",
               "ms": int((time.time() - t0) * 1000),
               "uso": datos.get("usage"),
               "texto": datos.get("choices", [{}])[0]
                        .get("message", {}).get("content")})
    return JSONResponse(datos, status_code=r.status_code)


@app.get("/v1/models")
async def modelos():
    r = await cliente.get(f"{ORIGEN}/models",
                          headers={"Authorization": f"Bearer {CLAVE}"})
    return JSONResponse(r.json())


@app.get("/salud")
def salud():
    return {"ok": True, "origen": ORIGEN,
            "enrutado": ENRUTAR, "logs": str(LOGS.resolve())}
