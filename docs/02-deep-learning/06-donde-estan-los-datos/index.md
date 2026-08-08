---
title: "6. Dónde están los datos"
tags: [federado, infraestructura]
status: esbozo
---

# 6. Dónde están los datos

> **Pregunta 3 de 3:** ¿dónde viven los datos y de quién son?

Es una pregunta **independiente** de las otras dos: cualquier modelo con
cualquier forma de aprender puede entrenarse en cualquiera de estas
configuraciones.

```none
                   ¿QUIÉN CONTROLA LOS DATOS?
                              │
        ┌─────────────────────┴──────────────────────┐
   UN SOLO DUEÑO                               VARIOS DUEÑOS
        │                                            │
   ┌────┴─────┐                        ┌─────────────┼─────────────┐
CENTRALIZADO  DISTRIBUIDO         FEDERADO     DESCENTRALIZADO  DATA SPACES
   (6.1)        (6.2)              (6.3)           (6.4)          (6.6)
todo en un    repartido por      los datos      ni siquiera     las reglas del
servidor      rendimiento        no se mueven   hay servidor    juego entre
                                                central         organizaciones
```

## La distinción que más se confunde

- **Distribuido** — reparto porque **quiero** (ir más rápido). Problema de
  ingeniería.
- **Federado** — reparto porque **no puedo juntarlos** (ley, secreto comercial,
  volumen). Problema de investigación.

## La magnitud que lo domina todo

El **ancho de banda**. Dentro de un centro de datos: cientos de Gb/s. Entre
organizaciones: quizá 1 Gb/s. Tres órdenes de magnitud de diferencia — y esa
es la razón de que cada configuración necesite algoritmos propios.
