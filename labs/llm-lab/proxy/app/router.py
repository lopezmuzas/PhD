"""Politica de enrutado (Lab 2). Devuelve el ID de modelo para una peticion."""

BARATO = "deepseek/deepseek-v4-flash"      # 0,14 $ / 0,28 $ por M
MEDIO = "deepseek/deepseek-v4-pro"         # 0,44 $ / 0,87 $ por M
CARO = "anthropic/claude-sonnet-5"         # 2 $ / 10 $ por M
GRATIS = "nvidia/nemotron-3-ultra-550b-a55b:free"

PALABRAS_DIFICILES = {
    "refactoriza", "arquitectura", "diseña", "depura", "por qué falla",
    "optimiza", "demuestra", "traza", "race condition", "deadlock",
}


def elegir_modelo(cuerpo: dict, forzar_gratis: bool = False) -> str:
    mensajes = cuerpo.get("messages", [])
    texto = " ".join(str(m.get("content", "")) for m in mensajes).lower()
    n_chars = len(texto)

    if forzar_gratis:
        return GRATIS
    if cuerpo.get("tools"):                       # modo agente: necesita cabeza
        return CARO
    if any(p in texto for p in PALABRAS_DIFICILES):
        return CARO
    if n_chars > 20_000:                          # contexto grande, barato y 1M
        return MEDIO
    if n_chars < 800 and len(mensajes) <= 2:      # pregunta corta y suelta
        return BARATO
    return MEDIO
