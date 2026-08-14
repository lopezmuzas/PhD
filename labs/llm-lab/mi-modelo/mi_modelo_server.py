"""
Shim compatible con la API de OpenAI para servir TU modelo a Continue.dev.

Continue solo necesita dos endpoints:
  GET  /v1/models              -> lista de modelos disponibles
  POST /v1/chat/completions    -> generacion (con y sin streaming)

Arrancar:
    pip install fastapi uvicorn
    uvicorn mi_modelo_server:app --host 127.0.0.1 --port 8000

Y en ~/.continue/config.yaml:
    - name: "[?/5] Antonio · Modelo propio · LOCAL"
      provider: openai
      model: mi-modelo-v1
      apiBase: http://127.0.0.1:8000/v1
      apiKey: no-importa   # requerido por el cliente, tu servidor lo ignora
      roles: [chat]
"""

import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

MODEL_ID = "mi-modelo-v1"


# --------------------------------------------------------------------
# AQUI VA TU MODELO
# Sustituye el cuerpo de generar() por tu forward pass. Recibe la lista
# de mensajes [{"role": ..., "content": ...}] y devuelve texto.
# --------------------------------------------------------------------
def generar(mensajes: list[dict], max_tokens: int = 512) -> str:
    ultimo = mensajes[-1]["content"] if mensajes else ""
    # Ejemplo tonto: un eco con estadisticas. Reemplaza por tu modelo.
    #   tokens = mi_tokenizer.encode(ultimo)
    #   salida = mi_modelo.generate(tokens, max_tokens=max_tokens)
    #   return mi_tokenizer.decode(salida)
    return (
        f"[mi-modelo-v1] He recibido {len(mensajes)} mensaje(s). "
        f"El ultimo tiene {len(ultimo)} caracteres. "
        f"Aqui iria la salida real del modelo."
    )


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[dict]
    max_tokens: int | None = 512
    temperature: float | None = 1.0
    stream: bool | None = False


@app.get("/v1/models")
def listar_modelos():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "antonio",
            }
        ],
    }


def _sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    texto = generar(req.messages, req.max_tokens or 512)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    creado = int(time.time())

    # --- Modo streaming: Continue lo usa por defecto ---
    if req.stream:

        def stream_tokens():
            # Trocea como quieras: por palabra, por token real de tu modelo...
            for palabra in texto.split(" "):
                yield _sse(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": creado,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": palabra + " "},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            yield _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": creado,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    # --- Modo no streaming ---
    return {
        "id": cid,
        "object": "chat.completion",
        "created": creado,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": texto},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "")) for m in req.messages) // 4,
            "completion_tokens": len(texto) // 4,
            "total_tokens": 0,
        },
    }
