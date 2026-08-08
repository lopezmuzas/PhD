---
title: "Anexo A — Lo que nadie sabe todavía"
tags: [teoria-abierta, tesis]
status: esbozo
---

# Anexo A — Lo que nadie sabe todavía

> Reservar sitio para la ignorancia es parte del mapa. Aquí vive la
> contribución original.

## A.1 Por qué generaliza

- **La paradoja central** — redes con capacidad de sobra para memorizar... no
  memorizan. La teoría clásica (VC, Rademacher) predice lo contrario.
- **Doble descenso** — el error de test baja, sube y **vuelve a bajar** al
  crecer el modelo. Rompe la curva en U de los libros de texto.
- **El ruido del SGD como regularizador** — el desorden del muestreo empuja
  hacia soluciones simples. Nadie sabe formalizarlo del todo.
- **Mínimos planos vs agudos** — la hipótesis de que los mínimos "anchos"
  generalizan mejor; discutida.
- **Grokking** — la red memoriza primero y, mucho después y de golpe, generaliza.

## A.2 Por qué la optimización funciona

- El paisaje no convexo debería estar lleno de trampas y el SGD las esquiva.
- **Conectividad de modos** — entre dos buenas soluciones hay caminos de bajo
  error. ¿Por qué?
- **Neural Tangent Kernel** — el régimen de anchura infinita explica algunas
  cosas y no las importantes.
- **Edge of stability** — el entrenamiento opera en el límite de lo estable, y
  parece que a propósito.

## A.3 Qué puede expresar una arquitectura

- **Aproximación universal ≠ aprendible** — existir no es poderse encontrar.
- **La ventaja de la profundidad** — hay funciones que una red profunda expresa
  con pocos parámetros y una somera necesita exponencialmente muchos.
- **Los límites de una GNN** — jerarquía Weisfeiler-Lehman.
- **Qué puede computar un Transformer** — límites formales del razonamiento en
  cadena.

## A.4 Qué hay dentro (interpretabilidad)

- **Circuitos** — ingeniería inversa de qué computa cada parte.
- **Superposición** — la red guarda más conceptos que neuronas tiene,
  mezclándolos en direcciones no ortogonales.
- **Autoencoders dispersos** — separar esa mezcla en características legibles.
- **¿El razonamiento mostrado es el real?** — la cadena de pensamiento puede ser
  una racionalización a posteriori.

## A.5 Cuándo falla (robustez)

- **Ejemplos adversariales** — perturbaciones invisibles que rompen el modelo.
- **Fuera de distribución** — detectar que la pregunta no se parece a nada visto.
- **Incertidumbre** — distinguir "no lo sé" (epistémica) de "es aleatorio"
  (aleatoria).
- **Alucinación** — causas, medición, mitigación.
- **Atajos** — la red aprende la correlación fácil, no la causa (la nieve, no el
  lobo).

## A.6 Aprendizaje repartido ★ (tu terreno)

- **A.6.1** Convergencia con no-IID severo y participación muy parcial:
  garantías realistas, no asintóticas.
- **A.6.2** La frontera privacidad-utilidad-comunicación: caracterizarla como
  superficie de Pareto, no como puntos sueltos.
- **A.6.3** Por qué funciona promediar pesos — y en qué régimen deja de
  funcionar. (Relacionado con la conectividad de modos de A.2.)
- **A.6.4** Equidad: que el modelo global no sacrifique a los clientes
  minoritarios mientras mejora la media.
- **A.6.5** Robustez sin agregador: defensas locales que no penalicen a los
  honestos que solo son distintos.
- **A.6.6** Incentivos veraces entre organizaciones que compiten.
- **A.6.7** Verificabilidad: probar criptográficamente que un nodo entrenó lo
  que declara.
- **A.6.8** **El puente contrato ↔ algoritmo** — traducir políticas de uso de un
  data space en restricciones ejecutables sobre el entrenamiento. Mínima
  literatura, máxima demanda.
- **A.6.9** Leyes de escalado descentralizadas: cómo cambian los exponentes
  cuando la sincronización es infrecuente.
- **A.6.10** Sostenibilidad: cuándo el entrenamiento repartido compensa de
  verdad frente al centralizado, contando el transporte.

## A.7 Aprender sin olvidar

- **Olvido catastrófico** — el gradiente sobrescribe lo anterior.
- **Estabilidad vs plasticidad** — el mismo dilema que el cerebro resuelve con
  plasticidad homeostática (→ 1.3.5).
- **Memorias a varias velocidades** — módulos que se actualizan a distintas
  frecuencias (Titans, Nested Learning).
- **En contexto vs en pesos** — dos memorias distintas; cuándo usar cada una.

## A.8 Alineamiento y sociedad

- **Especificar el objetivo** — la brecha entre lo que mides y lo que quieres
  (*reward hacking*).
- **Sesgos y equidad** — de datos, de representación, de despliegue.
- **Memorización y privacidad** — extraer datos de entrenamiento del modelo.
- **Energía** — el coste eléctrico como límite físico del escalado.
- **Gobernanza** — regulación, auditoría, trazabilidad.


## Para completar


- [ ] Elegir 2 preguntas de A.6 y hacer una revisión sistemática de cada una.
