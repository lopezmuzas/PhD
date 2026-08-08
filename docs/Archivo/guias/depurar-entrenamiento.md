---
title: Depurar un entrenamiento que no converge
tags: [practica, debugging]
status: revisado
updated: 2026-08-06
---

# Depurar un entrenamiento que no converge

Orden de comprobación, de lo más barato a lo más caro.

## 1. Sobreajusta un batch diminuto

El test más informativo que existe. Coge 8 muestras y entrena hasta que la
pérdida sea ~0. Si no lo consigue, el fallo está en el modelo o en el bucle,
no en los datos ni en los hiperparámetros.

```python
from itertools import islice
tiny = list(islice(train_dl, 1))
for _ in range(200):
    for xb, yb in tiny:
        ...  # el loss debe caer a casi cero
```

## 2. Comprueba la pérdida inicial

Para clasificación con $C$ clases equiprobables, la pérdida de partida debe ser
$\ln(C)$. Con 10 clases, ~2.30. Si empiezas muy lejos, la inicialización o la
capa de salida están mal.

## 3. Mira los gradientes, no solo la pérdida

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name:40s} {p.grad.norm():.3e}")
```

Ceros → gradiente cortado o activación saturada. `NaN` → tasa de aprendizaje
demasiado alta o división por cero en alguna normalización.

## 4. Errores clásicos

| Síntoma | Causa habitual |
|---|---|
| Validación mucho mejor que entrenamiento | Olvidaste `model.eval()`, o dropout activo |
| Pérdida estancada exactamente igual | `optimizer.zero_grad()` mal colocado |
| Va bien y de pronto `NaN` | LR alto; prueba *gradient clipping* |
| Accuracy sospechosamente alta | Fuga de datos: normalizaste antes de partir train/val |
| Muy lento con GPU al 20 % | Cuello de botella en el DataLoader, no en el modelo |
