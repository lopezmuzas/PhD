"""Precios por millon de tokens (entrada, salida). Verificado 13/08/2026.
Los :free no estan aqui: su coste no es cero, es cuota consumida."""

PRECIOS = {
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "x-ai/grok-4.6": (2.00, 6.00),
    "x-ai/grok-4.20": (1.25, 2.50),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
    "z-ai/glm-5.2": (0.50, 3.15),
    "meta/muse-glimmer-30b": (0.35, 1.50),
    "openai/gpt-5.1": (1.25, 10.00),
}


def coste(modelo: str, uso: dict) -> float:
    ent, sal = PRECIOS.get(modelo, (0.0, 0.0))
    return (uso.get("prompt_tokens", 0) * ent
            + uso.get("completion_tokens", 0) * sal) / 1e6
