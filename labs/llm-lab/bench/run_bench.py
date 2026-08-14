"""Lanza los mismos prompts contra varios modelos y guarda una tabla (Lab 3).

Uso:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python run_bench.py
"""
import json
import os
import time
from pathlib import Path

import httpx

CLAVE = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/v1/chat/completions"

MODELOS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "z-ai/glm-5.2",
    "anthropic/claude-sonnet-5",
]

# Precio por millon de tokens (entrada, salida). Los :free van a 0.
PRECIOS = {
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "z-ai/glm-5.2": (0.50, 3.15),
    "anthropic/claude-sonnet-5": (2.00, 10.00),
}


def coste(modelo: str, uso: dict) -> float:
    ent, sal = PRECIOS.get(modelo, (0.0, 0.0))
    return (uso.get("prompt_tokens", 0) * ent
            + uso.get("completion_tokens", 0) * sal) / 1e6


def lanzar(modelo: str, prompt: str) -> dict:
    t0 = time.time()
    try:
        r = httpx.post(BASE, timeout=180,
                       headers={"Authorization": f"Bearer {CLAVE}"},
                       json={"model": modelo,
                             "messages": [{"role": "user", "content": prompt}]})
        d = r.json()
        if "error" in d:
            return {"modelo": modelo, "error": d["error"].get("message", "?")}
        uso = d.get("usage", {})
        return {
            "modelo": modelo,
            "ms": int((time.time() - t0) * 1000),
            "tokens_salida": uso.get("completion_tokens", 0),
            "coste_usd": round(coste(modelo, uso), 6),
            "gratis": ":free" in modelo,
            "texto": d["choices"][0]["message"]["content"],
        }
    except Exception as e:                       # noqa: BLE001
        return {"modelo": modelo, "error": repr(e)}


def main() -> None:
    salida = Path("resultados")
    salida.mkdir(exist_ok=True)
    prompts = sorted(Path("prompts").glob("*.txt"))
    if not prompts:
        print("No hay prompts en prompts/*.txt")
        return

    for f in prompts:
        prompt = f.read_text(encoding="utf-8")
        filas = [lanzar(m, prompt) for m in MODELOS]

        (salida / f"{f.stem}.json").write_text(
            json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")

        lineas = [f"# Resultados — {f.stem}", "",
                  "La columna de calidad se rellena A MANO leyendo resultados/"
                  f"{f.stem}.json", "",
                  "| Modelo | ms | tokens | coste $ | cuota | calidad 1-5 |",
                  "|---|---|---|---|---|---|"]
        for r in filas:
            if "error" in r:
                lineas.append(f"| `{r['modelo']}` | — | — | — | — | ERROR: "
                              f"{r['error'][:60]} |")
            else:
                cuota = "1 de 200" if r["gratis"] else "—"
                lineas.append(
                    f"| `{r['modelo']}` | {r['ms']} | {r['tokens_salida']} "
                    f"| {r['coste_usd']:.5f} | {cuota} |  |")
        (salida / f"{f.stem}.md").write_text("\n".join(lineas) + "\n",
                                             encoding="utf-8")
        ok = sum(1 for r in filas if "error" not in r)
        print(f"{f.stem}: {ok}/{len(filas)} modelos OK")


if __name__ == "__main__":
    main()
