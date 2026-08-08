---
title: "Fase 5 — Federated Learning clásico (preparación)"
tags: [federado, distribuido, data-spaces]
status: borrador
updated: 2026-08-08
---

# Fase 5 — Federated Learning clásico (preparación)

> **Objetivo de la fase**: dominar el aprendizaje federado **en su forma estándar (supervised, sin RL)** antes de combinarlo con el setting secuencial. Entender por qué la mera transición de "training centralizado" a "training distribuido sin compartir datos" introduce **tres problemas conceptuales nuevos** — heterogeneidad estadística (non-IID), heterogeneidad de sistemas (clientes lentos/caídos) y privacy bajo modelo de adversario realista — y dominar las técnicas que los atacan: **FedAvg** (McMahan 2017) como base, **FedProx** y **SCAFFOLD** para non-IID, **DP-SGD** (Abadi 2016) y **Secure Aggregation** (Bonawitz 2017) para privacy. Salir con un pipeline FedAvg + non-IID + DP funcionando primero en simulación con **Flower**, y después con un *Hello World* desplegado en una testnet de **Pontus-X / Compute-to-Data**.
>
> **Tu situación de partida**: vienes de Fase 4 con CQL e IQL replicados sobre D4RL/Minari, un mini-paper de replicación, un análisis tesis-relevante sobre qué algoritmos offline son más "federables", y un `notes_for_supervisor.md` con tu lectura del campo. Lo que falta es **la otra mitad de la palabra "Federated Offline RL"**: aquí cierras esa mitad.
>
> **Tiempo estimado realista**: 4–6 semanas a ritmo de 1–2 h/día. Más corto que la Fase 4 por dos razones honestas: (1) FL clásico **es maduro**, los algoritmos están bien documentados y los frameworks son sólidos; (2) tu objetivo aquí no es ser experto en FL, es **ser fluido**. La frontera no está en FL clásico, está en FRL — y eso es Fase 6. El bloque más costoso será **Pontus-X**, no porque sea conceptualmente difícil, sino porque tocar infraestructura nueva siempre tarda más de lo que parece.

---

## Cómo usar esta guía

Misma lógica que las fases anteriores, con tres avisos específicos:

**Este bloque es preparación, no destino**. A diferencia de Fase 4 (que ya era tesis-relevant), Fase 5 es **el último bloque puramente instrumental** antes de tu doctorado. No te obsesiones con cubrir cada paper de FL — busca **fluidez operativa** y **diagnóstico de problemas**, no completitud académica. Si te encuentras leyendo el survey número 5 sobre Personalized FL, has caído en la trampa.

**El verdadero examen de Fase 5 no es entender FedAvg, es entender por qué FedAvg falla**. McMahan 2017 te enseña qué hacer cuando los clientes son cooperativos, los datos son razonablemente IID y nadie es malicioso. **El 90% del trabajo posterior del campo trata las desviaciones de ese mundo de juguete**: non-IID, sistemas heterogéneos, adversarios, privacy. Asegúrate de entender *qué supuesto rompe cada paper que leas* — es la única forma de no perderte en la sopa de acrónimos (FedProx, SCAFFOLD, FedNova, FedDyn, FedOpt, FedBN, MOON, pFedMe, Ditto…).

**Pontus-X es operacional, no académico**. La sección dedicada a desplegar en Pontus-X **no te dará intuición de investigación**, te dará **infraestructura sobre la que iterar durante tu doctorado**. Trátalo como tratarías setup de cluster GPU: una vez funciona, no lo tocas. Pero tiene que funcionar, y debe funcionar **antes** de Fase 6.

---

## Mapa de la Fase 5

| Bloque | Tema | Peso |
|---|---|---|
| 5.1 | El setting federado: por qué existe y qué supuestos rompe | 🔥🔥🔥🔥🔥 La pregunta central |
| 5.2 | FedAvg desde cero: el algoritmo fundacional | 🔥🔥🔥🔥🔥 No negociable |
| 5.3 | Heterogeneidad estadística (non-IID) y por qué rompe FedAvg | 🔥🔥🔥🔥🔥 El problema central del campo |
| 5.4 | FedProx, SCAFFOLD, FedNova, FedOpt — la familia de fixes | 🔥🔥🔥🔥 Tu vocabulario operativo |
| 5.5 | Heterogeneidad de sistemas: stragglers, dropouts, comunicación | 🔥🔥🔥 Pragmática del setting real |
| 5.6 | Differential Privacy y DP-SGD | 🔥🔥🔥🔥🔥 Llave para Pontus-X |
| 5.7 | Secure Aggregation, MPC y HE (a nivel conceptual) | 🔥🔥🔥 Conoce el menú, no cocines |
| 5.8 | Ataques: gradient inversion, membership inference, poisoning | 🔥🔥🔥🔥 Sin esto, "privacy" es marketing |
| 5.9 | Personalized FL: cuando "un modelo para todos" no aplica | 🔥🔥🔥 Conexión directa con tesis |
| 5.10 | Flower como framework de referencia | 🔥🔥🔥🔥 Tu herramienta de trabajo |
| 5.11 | Pontus-X / Compute-to-Data — Hello World en testnet | 🔥🔥🔥🔥🔥 La pieza más diferenciadora |
| 5.12 | Disciplina experimental específica de FL | 🔥🔥🔥🔥 No es disciplina genérica |
| 5.13 | Proyecto integrador: FedAvg + non-IID + DP + Pontus-X | 🔥🔥🔥🔥🔥 La pieza pre-doctoral |

---

## 5.1 — El setting federado: por qué existe y qué supuestos rompe

### Por qué importa — y esta es la pregunta que define toda la fase

ML centralizado asume que tienes un dataset $\mathcal{D}$ en un servidor y entrenas con SGD sobre él. Federated Learning asume que $\mathcal{D}$ está **partido entre $K$ clientes**, $\mathcal{D}_k$ vive en el cliente $k$, **no se mueve nunca**, y el objetivo es entrenar un modelo global $\theta$ que minimice $F(\theta) = \sum_k \frac{|\mathcal{D}_k|}{|\mathcal{D}|} F_k(\theta)$. La diferencia parece administrativa pero rompe cinco supuestos fundamentales del ML clásico:

1. **Datos no son IID** — cada cliente tiene su propia distribución $P_k(x, y)$ y suelen ser distintas. Esto no es un detalle: es el problema central del campo durante los últimos 8 años.
2. **Comunicación es cara** — en ML clásico mueves gradientes dentro de una GPU; en FL mueves vectores de millones de parámetros por red, posiblemente a través de un blockchain (Pontus-X). Cada byte cuenta.
3. **Clientes son heterogéneos en cómputo** — un hospital puede tener un servidor potente, otro un Raspberry Pi. Algunos van rápido, otros lento, otros se caen a mitad de ronda.
4. **Privacy es contractual, no opcional** — la razón de ser de FL es que **los datos no salen del cliente**. Si tu sistema fuga información (y lo hace por defecto vía los gradientes), has roto la promesa.
5. **No hay adversario asumido benévolo** — algunos clientes pueden ser maliciosos (poisoning), curiosos (gradient inversion) o simplemente buggeados.

Para tu tesis: en Pontus-X cada participante del dataspace es un cliente y los cinco supuestos están **activos a la vez**. Cuando además añades que cada cliente tiene un **dataset secuencial (RL)** generado por **su propia política de comportamiento** (Fase 4), los cinco se componen con el distributional shift offline. Esa composición es el lienzo de tu doctorado.

### Recurso principal — la pieza canónica

- **Kairouz, McMahan, Avent, et al. (2021) — "Advances and Open Problems in Federated Learning"** → [arXiv:1912.04977](https://arxiv.org/abs/1912.04977). **El survey de referencia del campo**, escrito por ~50 autores de Google + academia. 120 páginas. **No lo leas linealmente**: úsalo como atlas. Lee la introducción, la taxonomía (sección 2), y después salta a las secciones específicas que necesites en cada bloque. Es la cita obligatoria de cualquier paper serio de FL.
- **McMahan, Moore, Ramage, Hampson & Arcas (2017) — "Communication-Efficient Learning of Deep Networks from Decentralized Data"** → [arXiv:1602.05629](https://arxiv.org/abs/1602.05629). **El paper original de FedAvg**. Solo 11 páginas. **Léelo entero, dos veces**. Es el "Sutton & Barto" de FL: todo lo que viene después se entiende en función de qué problema deja sin resolver.

### Vídeo

- **Curso CMU 11-868 — Federated Learning** (Virginia Smith, ediciones 2023/2024). Búscalo en YouTube como "CMU 11-868 federated learning". **El mejor curso académico abierto sobre FL hoy**. Si solo ves un curso entero, que sea este.
- **Charlas de Peter Kairouz / Brendan McMahan** en seminarios académicos. Búscalas en YouTube — la calidad pedagógica es alta y son los autores de los papers que vas a leer.

### Temas mínimos

Formulación matemática: $\min_\theta F(\theta) = \sum_k p_k F_k(\theta)$ donde $p_k = |\mathcal{D}_k|/|\mathcal{D}|$ y $F_k$ es el riesgo empírico local. **Cross-device vs. cross-silo**: cross-device = millones de teléfonos, clientes intermitentes, baja confianza; cross-silo = decenas de organizaciones, clientes estables, alta confianza. **Pontus-X es cross-silo por construcción** — esto te ahorra mucha complejidad de cross-device y enfoca tu lectura. **Horizontal vs. vertical FL**: horizontal = mismas features, distintos samples (varios hospitales con los mismos campos en historial clínico); vertical = mismos samples, distintas features (banco y compañía de seguros con clientes que se solapan pero con datos distintos). La mayoría de la literatura es horizontal, y tu tesis casi seguro también. **Rondas de comunicación vs. epochs locales**: cada ronda los clientes hacen $E$ epochs locales antes de enviar. $E$ grande reduce comunicación pero rompe convergencia en non-IID. **El supuesto de cliente honesto-pero-curioso vs. malicioso**: cambia radicalmente qué defensas necesitas.

### Ejercicio "wow"

Sin Flower aún, sin distribución real, en un único Jupyter Notebook: simula 5 clientes con MNIST partido **IID** (mezcla aleatoria) y MNIST partido **non-IID extremo** (cada cliente solo ve 2 dígitos). Entrena un CNN simple **localmente en cada cliente sin compartir nada** y mide accuracy global (en el test set completo). Después entrena el mismo CNN **centralmente** sobre la unión de los 5 datasets. La diferencia entre "centralizado" y "puramente local" en non-IID es **dramática** — esa brecha es el campo entero de FL intentando cerrarse. **Replicar este experimento es lo primero que debes hacer y la motivación que cualquier slide de FL te va a vender**.

### Checklist 5.1

- [ ] Leí McMahan 2017 entero, con notas
- [ ] Tengo Kairouz 2021 como referencia consultable (no leído entero)
- [ ] Sé enumerar los 5 supuestos del ML centralizado que FL rompe
- [ ] Sé distinguir cross-device de cross-silo y horizontal de vertical, y ubicar Pontus-X en el cuadrante correcto
- [ ] Reproduje la brecha "local-only vs. centralizado" en MNIST IID vs. non-IID extremo

---

## 5.2 — FedAvg desde cero: el algoritmo fundacional

### Por qué importa

FedAvg es a FL lo que SGD es a ML clásico: el algoritmo base que todo el mundo extiende. Es **trivial conceptualmente** (todos los clientes hacen SGD local, el servidor promedia los pesos) pero contiene **dos decisiones de diseño no triviales** que vas a ver repetidas en cada algoritmo posterior: (1) **promediado por número de muestras** (no uniforme), (2) **múltiples epochs locales por ronda** (no un solo gradiente por ronda como en mini-batch SGD distribuido). Estas dos decisiones son las que **rompen** la equivalencia con SGD centralizado en non-IID — y casi todos los fixes posteriores son intentos de recuperar esa equivalencia.

### Recurso principal

- **McMahan et al. 2017 — FedAvg paper** (ya citado en 5.1). Apéndice incluido. Mira la sección "Experiments" — la diferencia entre `IID` y `non-IID` ya aparece ahí, sutil pero clara.

### Lecturas complementarias

- **Li, Sahu, Talwalkar & Smith (2020) — "Federated Learning: Challenges, Methods, and Future Directions"** → [arXiv:1908.07873](https://arxiv.org/abs/1908.07873). Survey corto y digerible, anterior a Kairouz. Útil como segunda pasada conceptual.
- **Wang et al. (2021) — "A Field Guide to Federated Optimization"** → [arXiv:2107.06917](https://arxiv.org/abs/2107.06917). **Súper útil** como diccionario operativo de las variantes de FedAvg. Cuando alguien diga "usé FedOpt con Adam server-side", esta es tu referencia.

### Temas mínimos

Pseudocódigo de FedAvg: en cada ronda $t$, el servidor envía $\theta_t$ a un subconjunto $S_t$ de clientes; cada cliente $k \in S_t$ ejecuta $E$ epochs de SGD local sobre $\mathcal{D}_k$ para obtener $\theta_t^k$; el servidor agrega $\theta_{t+1} = \sum_{k \in S_t} \frac{n_k}{n_S} \theta_t^k$. **Hiperparámetros**: $E$ (epochs locales), $B$ (batch size local), $C$ (fracción de clientes por ronda), $\eta$ (lr local). **El trade-off $E$**: $E=1$ es básicamente SGD distribuido (estable pero lento); $E$ grande reduce comunicación pero hace "client drift" — la actualización local diverge de la dirección global. **Por qué se promedian los pesos y no los gradientes**: si todos los clientes parten del mismo $\theta_t$, promediar gradientes y promediar pesos son equivalentes; con $E>1$ epochs, **dejan de serlo** y el promediado de pesos introduce sesgo no nulo. **Convergencia teórica**: bajo IID y convexidad fuerte, FedAvg converge a la misma tasa que SGD (a constantes). Bajo non-IID, las garantías se degradan y hay un *floor* de error.

### Ejercicio obligatorio

Implementa FedAvg **desde cero, sin Flower**, en PyTorch puro. Una sola máquina, 5–10 clientes simulados como instancias de un `ClientModel` con su propio dataloader. Servidor en bucle:

```python
for round in range(R):
    selected = random.sample(clients, k=C*K)
    weights_list, sizes = [], []
    for client in selected:
        w = client.local_train(global_weights, epochs=E)
        weights_list.append(w); sizes.append(client.n_samples)
    global_weights = weighted_average(weights_list, sizes)
    eval_global(global_weights)
```

Entrena MNIST con $K=10$ clientes, $C=1.0$, $E=5$, $B=32$. Reproduce la **figura 1 de McMahan 2017**: accuracy vs. número de rondas para IID y non-IID. Verás que IID converge razonablemente y non-IID oscila o tarda 10× más. **Sin esto implementado a mano, no entiendes FedAvg**. Una vez funciona, **puedes migrar a Flower con la conciencia tranquila** sabiendo qué hace cada línea por debajo.

### Checklist 5.2

- [ ] Implementé FedAvg from scratch en PyTorch (≤200 líneas), sin Flower
- [ ] Reproduje la figura IID vs. non-IID de McMahan 2017 sobre MNIST
- [ ] Sé explicar por qué $E>1$ rompe la equivalencia gradient-averaging vs. weight-averaging
- [ ] Sé enumerar los hiperparámetros canónicos ($E$, $B$, $C$, $\eta$) y dar un default razonable para cada uno
- [ ] Tengo intuición del trade-off comunicación vs. convergencia controlado por $E$

---

## 5.3 — Heterogeneidad estadística (non-IID) y por qué rompe FedAvg

### Por qué importa — y este es el problema central del campo

Si los clientes tuvieran datos IID, FedAvg sería trivialmente correcto y FL sería un problema resuelto. **No los tienen, y por eso el campo existe**. Hay al menos **cinco fuentes de non-IID** que necesitas distinguir, porque cada una induce un patrón de fallo distinto:

1. **Label distribution skew**: cada cliente tiene distintos $P(y)$. Un hospital pediátrico ve solo niños, uno geriátrico solo ancianos. El más común y el que más se estudia.
2. **Feature distribution skew (covariate shift)**: $P(x)$ varía entre clientes. Mismas categorías de cáncer pero rayos X de distintos fabricantes.
3. **Same label, different features**: $P(x|y)$ varía. "Buen cliente" en un banco europeo vs. uno latinoamericano.
4. **Same features, different labels (concept drift)**: $P(y|x)$ varía. Diagnóstico que difiere entre escuelas médicas.
5. **Quantity skew**: algunos clientes tienen miles de muestras, otros decenas.

**Para tu tesis (Pontus-X / Offline FRL)**: la heterogeneidad es **inevitablemente alta** porque cada participante del dataspace genera datos con su propia política/proceso/dominio. Y peor: como los datos son **trayectorias RL**, las muestras están **correlacionadas temporalmente** — el supuesto IID que ya viola FL clásico se viola **dos veces** en tu setting.

### Recurso principal

- **Hsieh, Phanishayee, Mutlu & Gibbons (2020) — "The Non-IID Data Quagmire of Decentralized Machine Learning"** → [arXiv:1910.00189](https://arxiv.org/abs/1910.00189). El paper que **diagnostica el problema** sin proponer un fix. Lee la sección donde muestran que la BatchNorm es uno de los principales culpables de la divergencia en non-IID — esto es operacionalmente importante.
- **Li, He & Song (2022) — "Federated Learning on Non-IID Data Silos: An Experimental Study"** → [arXiv:2102.02079](https://arxiv.org/abs/2102.02079). **Benchmark experimental sistemático** de FedAvg, FedProx, SCAFFOLD, FedNova bajo distintas variantes de non-IID. Tu mapa visual del problema.

### Temas mínimos

**Cómo se simula non-IID en benchmarks**: partición por etiqueta (cliente $k$ solo ve $n_c$ clases); partición Dirichlet (cliente $k$ recibe muestras de cada clase con probabilidad $\propto p_c$, $p \sim \text{Dir}(\alpha)$, $\alpha \to 0$ extremo, $\alpha \to \infty$ uniforme). **Client drift**: en non-IID, $\nabla F_k(\theta) \neq \nabla F(\theta)$ — la dirección óptima local diverge de la global. Con $E$ epochs locales, esa divergencia se amplifica. **Patología visible**: la accuracy global oscila o se estanca; ciertas clases se aprenden y otras se "olvidan" ronda a ronda. **BatchNorm es problemático**: sus estadísticas (running mean/var) son **locales** y promediarlas no tiene sentido cuando los clientes ven distribuciones distintas. Soluciones: GroupNorm, LayerNorm, o FedBN (no se promedian las stats de BN).

### Ejercicio

Sobre tu FedAvg de 5.2, añade un esquema de partición Dirichlet con $\alpha \in \{0.05, 0.1, 0.5, 5.0\}$ sobre CIFAR-10 con $K=10$ clientes. **Reproduce la "curva de degradación"**: accuracy final vs. $\alpha$. Verás un colapso entre $\alpha=0.5$ y $\alpha=0.1$. **Esta gráfica es la que abre la sección "Motivation" de la mitad de los papers de FL** — tenla tú mismo en figura propia.

Segundo ejercicio (corto, revelador): repite con un modelo con BatchNorm vs. con GroupNorm. La diferencia en non-IID extremo es del orden de 10–20 puntos de accuracy.

### Checklist 5.3

- [ ] Sé enumerar las 5 fuentes de non-IID y dar un ejemplo realista de cada una
- [ ] Reproduje la curva accuracy vs. $\alpha$ Dirichlet en CIFAR-10
- [ ] Verifiqué experimentalmente el problema de BatchNorm en non-IID
- [ ] Entiendo conceptualmente qué es "client drift"
- [ ] Sé conectar las 5 fuentes con el setting RL: por qué un dataset offline + heterogeneidad federada amplifica el problema

---

## 5.4 — FedProx, SCAFFOLD, FedNova, FedOpt — la familia de fixes

### Por qué importa

Una vez identificado client drift como el problema, hay **cuatro familias de soluciones** que necesitas dominar a nivel de "saber decir qué hace cada una en una frase":

- **FedProx (Li et al. 2018)**: añade un **término proximal** $\frac{\mu}{2}\|\theta_k - \theta_{global}\|^2$ a la loss local. Penaliza alejarse del modelo global. Una línea de código, casi gratis, mejora estabilidad.
- **SCAFFOLD (Karimireddy et al. 2020)**: corrige el drift usando **control variates** — cada cliente mantiene una estimación del gradiente "global" y corrige el local con ella. Más sofisticado, más eficiente, más coste de comunicación (envía dos vectores en lugar de uno).
- **FedNova (Wang et al. 2020)**: corrige el sesgo introducido por **heterogeneidad en el número de epochs locales** entre clientes (algunos hacen más SGD steps que otros). Normaliza para que la dirección agregada sea no sesgada.
- **FedOpt / FedAdam (Reddi et al. 2020)**: trata la agregación del servidor como un **paso de optimizador**. En lugar de promediar pesos, computa la "pseudo-gradiente" $\Delta = \theta_{global} - \bar{\theta}_{clientes}$ y aplícale Adam/Yogi/Adagrad. Mejora robustez sin tocar a los clientes.

**Para tu tesis**: no necesitas elegir aún cuál usar como base — pero **necesitas saber cuál usaría cada paper de FRL que leas**. Y necesitas la intuición de cuál es más natural para extender al setting RL (spoiler: **FedProx y FedOpt** son los más utilizados como ingredientes en FRL).

### Lecturas obligatorias (en orden)

- **Li, Sahu, Zaheer, Sanjabi, Talwalkar & Smith (2018) — "Federated Optimization in Heterogeneous Networks" (FedProx)** → [arXiv:1812.06127](https://arxiv.org/abs/1812.06127).
- **Karimireddy, Kale, Mohri, Reddi, Stich & Suresh (2020) — "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"** → [arXiv:1910.06378](https://arxiv.org/abs/1910.06378).
- **Wang, Liu, Liang, Joshi & Poor (2020) — "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" (FedNova)** → [arXiv:2007.07481](https://arxiv.org/abs/2007.07481).
- **Reddi, Charles, Zaheer, Garrett, Rush, Konečný, Kumar & McMahan (2020) — "Adaptive Federated Optimization" (FedOpt)** → [arXiv:2003.00295](https://arxiv.org/abs/2003.00295).

### Temas mínimos

**FedProx**: el término proximal $\mu \|\theta_k - \theta_{global}\|^2$ es **trivial** de implementar (es L2 regularización hacia el modelo global). Hiperparámetro $\mu$ pequeño (orden 0.01). Funciona sorprendentemente bien para lo barato que es. **SCAFFOLD**: cada cliente mantiene $c_k$ y el servidor mantiene $c$. Update local corregido: $\theta_k \leftarrow \theta_k - \eta(g_k - c_k + c)$. Más overhead de comunicación pero garantías de convergencia significativamente mejores. **FedNova**: si cliente $k$ hace $\tau_k$ steps de SGD, el update efectivo es $\Delta_k / \tau_k \cdot \bar{\tau}$. Corrige el sesgo cuando los $\tau_k$ difieren entre clientes — crucial cuando los recursos computacionales son heterogéneos. **FedOpt/FedAdam**: el servidor mantiene momentos de primer y segundo orden sobre $\Delta$, aplica el step adaptativo. **Reglas heurísticas operativas**: si tienes non-IID moderado y quieres simple → FedProx. Si tienes non-IID severo y puedes pagar 2× comunicación → SCAFFOLD. Si los clientes son muy heterogéneos en compute → FedNova. Si quieres estabilidad gratis → FedOpt en el servidor sobre FedAvg.

### Ejercicio

Sobre tu FedAvg + non-IID Dirichlet de 5.3, añade **FedProx** (es 5 líneas extra) y **FedOpt** (es 10 líneas extra en el servidor). Mide la diferencia en convergencia bajo $\alpha=0.1$. Implementa **SCAFFOLD** solo si tienes margen (es ~50 líneas extra y un control variate por cliente que mantener). FedNova es opcional — léelo, no lo implementes.

Tabla obligatoria al cerrar el bloque:

| Algoritmo | CIFAR-10 $\alpha=0.5$ | CIFAR-10 $\alpha=0.1$ | CIFAR-10 $\alpha=0.05$ |
|---|---|---|---|
| FedAvg | | | |
| FedProx | | | |
| FedOpt | | | |
| SCAFFOLD (opc) | | | |

≥3 seeds. Reportar accuracy global IID en la primera fila como referencia "techo".

### Checklist 5.4

- [ ] Sé enunciar en una frase qué hace cada uno de FedProx, SCAFFOLD, FedNova, FedOpt
- [ ] Implementé FedProx y FedOpt sobre mi FedAvg base
- [ ] Tengo la tabla comparativa en CIFAR-10 con Dirichlet $\alpha$ variable
- [ ] Tengo intuición de cuál usar en función del tipo de heterogeneidad
- [ ] Sé identificar cuál se usa como base en los papers de FRL que voy a leer en Fase 6

---

## 5.5 — Heterogeneidad de sistemas: stragglers, dropouts, comunicación

### Por qué importa

Hasta ahora has asumido que todos los clientes responden a tiempo y nadie se cae. **En la práctica, no**. Tres patologías clásicas:

- **Stragglers** (rezagados): un cliente lento hace que la ronda entera espere. Soluciones: timeout, asynchronous FL, agregación parcial.
- **Dropouts**: clientes que se caen a mitad de ronda. Su update se pierde — y si era el cliente con la mayor cantidad de datos, sesga la agregación.
- **Comunicación**: enviar 100M de parámetros de un modelo grande a través de una red lenta (o un blockchain) es **el cuello de botella real** en producción.

**Para Pontus-X específicamente**: la comunicación cuesta **dinero literal** (tokens). Reducir bytes enviados no es optimización académica, es economía. Cuando hagas tu Fase 6, los métodos de compresión que vas a leer aquí pueden ser **una contribución doctoral por sí mismos** en el contexto de FRL sobre dataspaces.

### Lecturas

- **Bonawitz et al. (2019) — "Towards Federated Learning at Scale: System Design"** → [arXiv:1902.01046](https://arxiv.org/abs/1902.01046). El paper de Google sobre cómo se diseñó FL en producción para teclados Android (Gboard). Léelo por la perspectiva de sistemas — te abre los ojos sobre cuánto del problema es ingeniería y no ML.
- **Konečný, McMahan, Yu, Richtárik, Suresh & Bacon (2016) — "Federated Learning: Strategies for Improving Communication Efficiency"** → [arXiv:1610.05492](https://arxiv.org/abs/1610.05492). Sobre compresión de gradientes en FL.
- **Sattler, Wiedemann, Müller & Samek (2019) — "Robust and Communication-Efficient Federated Learning From Non-IID Data"** → [arXiv:1903.02891](https://arxiv.org/abs/1903.02891). STC (Sparse Ternary Compression).

### Temas mínimos

**Asynchronous FL**: el servidor no espera a todos; cuando llega un update lo agrega inmediatamente. Convergencia más sutil, requiere "staleness penalty". **Client sampling estratégico**: en lugar de uniforme, muestrea clientes con probabilidad $\propto n_k$ para reducir varianza, o $\propto$ pérdida local para acelerar convergencia. **Compresión de comunicación**: (1) **quantization** (8-bit, 4-bit, 1-bit como en sign-SGD); (2) **sparsification** (top-k gradientes, randomized sparsification); (3) **low-rank factorization** del update; (4) **sketching** (Count Sketch). **Local steps** como compresión implícita: $E$ epochs locales reducen comunicación por factor $E$, pero a costa de client drift. **Comunicación en blockchain (Pontus-X)**: cada transacción tiene coste fijo en gas/tokens. **Off-chain communication** (los modelos viajan por canales tradicionales, solo metadata y hashes van on-chain) es la práctica estándar — Compute-to-Data es exactamente esto.

### Ejercicio (ligero)

No implementes asynchronous FL ni sketching — son ratholes. **Sí implementa** dos cosas concretas:

1. **Quantization simple a 8-bit** sobre tus updates de FedAvg. Mide degradación de accuracy vs. reducción de bytes.
2. **Top-k sparsification** (k=10% de los pesos con mayor magnitud de update). Mismo análisis.

Esto te da intuición de **cuánto puedes comprimir antes de romper la convergencia** — útil cuando en Pontus-X cada byte cuente.

### Checklist 5.5

- [ ] Leí el paper de "FL at Scale" de Bonawitz 2019 y entiendo el setting cross-device industrial
- [ ] Implementé quantization 8-bit sobre mis updates de FedAvg y medí la degradación
- [ ] Implementé top-k sparsification y medí el trade-off bytes vs. accuracy
- [ ] Sé enumerar 4 técnicas de compresión y cuándo aplicar cada una
- [ ] Entiendo por qué local steps son "compresión implícita" y cómo se trade contra client drift

---

## 5.6 — Differential Privacy y DP-SGD

### Por qué importa — y este es el llave operativa para Pontus-X

FL **sin** privacy formal es marketing: aunque los datos crudos no se compartan, los gradientes filtran información (sección 5.8 lo demuestra). **Differential Privacy** te da una garantía matemática: el output del algoritmo es **estadísticamente indistinguible** entre el caso "el sample $x$ está en el dataset" y "no está". Formalmente, $(\epsilon, \delta)$-DP: $\Pr[M(\mathcal{D}) \in S] \leq e^\epsilon \Pr[M(\mathcal{D}') \in S] + \delta$ para datasets vecinos $\mathcal{D}, \mathcal{D}'$.

**El mecanismo estándar para deep learning es DP-SGD** (Abadi et al. 2016): en cada step, se **clipa** la norma del gradiente por sample y se añade ruido Gaussiano. Esto te da DP por sample, y por composition de rondas obtienes la garantía total. **Para Pontus-X / regulación europea, DP-SGD es probablemente requisito legal**, no opcional, en muchos dataspaces médicos y financieros bajo el Data Governance Act y la AI Act.

### Recurso principal — la pieza canónica

- **Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar & Zhang (2016) — "Deep Learning with Differential Privacy"** → [arXiv:1607.00133](https://arxiv.org/abs/1607.00133). **El paper de DP-SGD**. Léelo entero. Introduce el **moments accountant**, la herramienta matemática estándar para componer las garantías DP a lo largo del training.
- **Dwork & Roth (2014) — "The Algorithmic Foundations of Differential Privacy"** → [libro gratis, ~280 páginas](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf). **No te lo leas entero**, pero ten claros los capítulos 2 y 3 (definición, mecanismos básicos: Laplace, Gaussian).

### Lecturas FL-específicas

- **McMahan, Ramage, Talwar & Zhang (2018) — "Learning Differentially Private Recurrent Language Models"** → [arXiv:1710.06963](https://arxiv.org/abs/1710.06963). DP-FedAvg: cómo aplicar DP a nivel de **usuario** (no de sample) en el setting federado. **Crucial para entender qué garantías ofrece FL+DP**.
- **Geyer, Klein & Nabi (2017) — "Differentially Private Federated Learning: A Client Level Perspective"** → [arXiv:1712.07557](https://arxiv.org/abs/1712.07557). Vista alternativa de DP a nivel de cliente.

### Temas mínimos

**Definición $(\epsilon, \delta)$-DP**: $\epsilon$ controla cuánto se filtra ($\epsilon$ bajo = más privacy), $\delta$ es la probabilidad de fallo (suele ser $\ll 1/n$). **Mecanismo Gaussiano**: para una función con sensitivity $\Delta_2$, añadir $\mathcal{N}(0, \sigma^2 I)$ con $\sigma = \Delta_2 \sqrt{2\ln(1.25/\delta)}/\epsilon$ da $(\epsilon, \delta)$-DP. **DP-SGD**: clip por sample ($\|g_i\| \leq C$), agrega ruido a la suma, escala por batch size. **Privacy accounting**: composition naïve da $\epsilon$ que crece linealmente; el **moments accountant** (Abadi) y **RDP** (Mironov 2017) dan composition más ajustada. **Privacy-utility trade-off**: $\epsilon \to 0$ destruye accuracy; $\epsilon$ grande "compra" accuracy pero degrada garantía. La curva privacy-utility ($\epsilon$ vs. accuracy) **es el output central de cualquier paper DP-ML**. **DP a nivel de sample vs. cliente (user-level DP)**: en FL, lo que normalmente quieres garantizar es que "la participación de un cliente entero no afecta detectablemente al modelo final". Esto requiere clip del **update completo del cliente** (no de gradientes individuales).

### Herramientas

- **Opacus** ([opacus.ai](https://opacus.ai/)) — la librería de PyTorch para DP-SGD. Implementación calidad-producción. **Aprende a usarla**.
- **TF Privacy** — equivalente en TensorFlow. Solo si usas TF.
- **Flower + Opacus**: integración estándar para DP-FedAvg.

### Ejercicio obligatorio

Implementa **DP-SGD con Opacus** sobre MNIST centralizado primero (un solo cliente, sin FL aún). Dibuja la curva **privacy-utility**: $\epsilon \in \{0.5, 1, 2, 4, 8, \infty\}$ vs. accuracy de test, fijando $\delta = 10^{-5}$. **Esta curva es la primera curva DP de tu carrera y vas a redibujarla muchas veces**.

Después: combina DP-SGD con FedAvg en Flower (Flower tiene primitivas para esto). Dibuja la curva **user-level DP**: privacy a nivel de cliente vs. accuracy global. Compara con la curva sample-level. La user-level es **siempre más costosa** en privacy-utility — esa es la lección.

### Checklist 5.6

- [ ] Leí Abadi 2016 y sé enunciar la definición $(\epsilon, \delta)$-DP
- [ ] Sé qué hace cada paso de DP-SGD: per-sample clip, ruido, batch averaging
- [ ] Implementé DP-SGD con Opacus en MNIST centralizado
- [ ] Dibujé la curva privacy-utility para varios $\epsilon$
- [ ] Sé distinguir DP sample-level de DP user-level y por qué importa en FL
- [ ] Combiné DP-SGD con FedAvg en Flower (al menos un experimento)

---

## 5.7 — Secure Aggregation, MPC y HE (a nivel conceptual)

### Por qué importa

DP te da privacy **estadística** pero el servidor sigue **viendo** los updates individuales. **Secure Aggregation** (SA) es ortogonal: usa criptografía para que el servidor **solo vea la suma** de los updates, nunca uno individual. Combinar SA + DP es el estándar de oro en FL con garantías serias.

**Para tu tesis no necesitas implementar SA**. Necesitas saber **qué garantiza**, **qué cuesta**, y **por qué combinarlo con DP** es la receta canónica. Si tu director te pide criptografía profunda, eso es Fase 6 o más allá. Aquí, conoce el menú.

### Lecturas

- **Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal & Seth (2017) — "Practical Secure Aggregation for Privacy-Preserving Machine Learning"** → [paper de Google](https://eprint.iacr.org/2017/281). **El paper canónico de Secure Aggregation en FL**. Lectura conceptual, no implementacional.
- Secciones de **Kairouz et al. 2021** sobre privacy y secure aggregation (capítulo 4).

### Temas mínimos

**Secure Aggregation (Bonawitz 2017)**: cada par de clientes intercambia una "máscara aleatoria" usando key agreement (Diffie-Hellman). El cliente envía $\theta_k + \text{máscaras}$. Cuando el servidor suma todos los clientes, las máscaras se cancelan dos a dos y solo queda $\sum_k \theta_k$. **Resistente a dropouts** vía secret sharing. **MPC (Multi-Party Computation)**: paradigma más general — varias partes computan una función sin revelar inputs. SA es un caso particular de MPC para la suma. Frameworks: **CrypTen** (Facebook), **TF Encrypted**. Coste: ~2–10× más lento que sin MPC, según la operación. **Homomorphic Encryption (HE)**: cifrado que permite operaciones (suma, multiplicación) sobre el cifrado sin descifrar. CKKS (aproximado, suficiente para ML), BFV/BGV (exacto). **Coste**: ~100–1000× más lento que sin HE. **No es práctico para training completo** hoy; sí lo es para inferencia o para sumas pequeñas. **Frameworks**: Microsoft SEAL, OpenFHE, PySyft + TenSEAL.

**Cuál usar cuándo (regla operativa)**:
- "Quiero garantía estadística barata" → DP (5.6).
- "Quiero que el servidor no vea updates individuales" → Secure Aggregation.
- "Tengo regulación que requiere cifrado de datos en tránsito y en cómputo" → HE (caro, solo si te obligan).
- "Lo serio" → DP + SA combinados.

### Ejercicio (ligero)

**No implementes nada criptográfico desde cero**. Dos cosas pequeñas:

1. Lee el paper de Bonawitz 2017 con la pregunta concreta: "¿qué cuesta en bytes y rondas comparado con FedAvg vanilla?". Anota la respuesta.
2. **Toca PySyft / OpenMined** ([openmined.org](https://openmined.org/)) por una tarde — corre el ejemplo de Secure Aggregation que viene en sus tutoriales. **No es para usarlo en producción**, es para que **no te asuste el vocabulario**.

### Checklist 5.7

- [ ] Sé explicar conceptualmente cómo funciona Secure Aggregation (máscaras + cancelación)
- [ ] Sé enumerar 3 técnicas criptográficas (SA, MPC, HE) y dar costes relativos aproximados
- [ ] Toqué PySyft / OpenMined una vez (no necesito usarlo)
- [ ] Tengo claro cuándo combinar DP + SA y por qué es la receta canónica
- [ ] No me asusta el vocabulario de criptografía aplicada a FL

---

## 5.8 — Ataques: gradient inversion, membership inference, poisoning

### Por qué importa — y sin esto, "privacy" es marketing

La razón por la que necesitas DP y SA es **concreta**: hay ataques publicados, reproducibles, que **recuperan los datos crudos** a partir de los gradientes en FL sin defensas. Si no conoces estos ataques, no puedes evaluar si tu sistema tiene la privacy que crees que tiene.

**Tres categorías de ataque**:

1. **Gradient inversion (reconstruction)**: dado un gradiente compartido, **reconstruir el batch de inputs original**. Funciona inquietantemente bien en imágenes — Zhu et al. 2019 ("Deep Leakage from Gradients") recuperan imágenes pixel-a-pixel.
2. **Membership inference**: dado un sample $x$, determinar si estaba en el training set de algún cliente.
3. **Poisoning** (envenenamiento): cliente malicioso envía updates manipulados para degradar el modelo (untargeted) o introducir un backdoor (targeted).

**Para tu tesis**: en Pontus-X, los clientes son **organizaciones** con interés económico — el modelo de adversario realista incluye "competidores que quieren extraer información del dataset del otro" y "actores maliciosos que envían updates corruptos para sabotear". DP defiende del primero (parcialmente), Krum / Byzantine-robust aggregation del segundo.

### Lecturas obligatorias

- **Zhu, Liu & Han (2019) — "Deep Leakage from Gradients" (DLG)** → [arXiv:1906.08935](https://arxiv.org/abs/1906.08935). **El paper que prendió la alarma del campo**. Demuestran reconstrucción pixel-a-pixel de batches de CIFAR-10 desde gradientes compartidos. **Léelo entero — es una llamada de atención**.
- **Geiping, Bauermeister, Dröge & Moeller (2020) — "Inverting Gradients — How easy is it to break privacy in federated learning?"** → [arXiv:2003.14053](https://arxiv.org/abs/2003.14053). Versión más fuerte de DLG, funciona en setting realistas.
- **Bagdasaryan, Veit, Hua, Estrin & Shmatikov (2019) — "How To Backdoor Federated Learning"** → [arXiv:1807.00459](https://arxiv.org/abs/1807.00459). El paper canónico de backdoor attacks en FL.
- **Blanchard, El Mhamdi, Guerraoui & Stainer (2017) — "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (Krum)** → la defensa canónica contra updates Byzantinos.

### Temas mínimos

**Threat model**: honest-but-curious server (gradient inversion, membership inference) vs. malicious client (poisoning, backdoor) vs. malicious server (everything). Cada uno requiere defensas distintas. **Gradient inversion**: optimiza $\min_x \|\nabla_\theta L(f(x), y) - g_{shared}\|^2$ para recuperar $x$. Funciona porque los gradientes contienen casi toda la información del input para modelos sobre-parametrizados. **Defensas contra gradient inversion**: DP-SGD (la única defensa con garantía formal), reducir batch size aumenta el ataque (contraintuitivo), aumentar local epochs lo dificulta. **Membership inference**: clasificador binario sobre confidence/loss patterns. **Defensas**: DP, regularización, knowledge distillation. **Backdoor attacks**: cliente malicioso entrena para que el modelo prediga "X" sobre inputs con un trigger específico (un pixel coloreado). **Model replacement attack**: si el adversario controla parte sustancial de la agregación, puede sustituir el modelo entero. **Defensas Byzantine-robust**: **Krum**, **Median**, **Trimmed Mean**, **Bulyan** — agregación robusta que descarta outliers. Trade-off: robustez vs. accuracy en clientes honestos pero divergentes (que se confunden con maliciosos).

### Ejercicio

**Reproduce DLG sobre tu FedAvg de 5.2** sin defensas. Toma el update de un cliente sobre un batch de 4 imágenes de CIFAR-10. Implementa el ataque de Zhu 2019 (~50 líneas, código disponible en GitHub del paper). **Verás las imágenes reconstruidas** — ese momento es el "wow" que cambia tu intuición sobre qué garantiza FL solo.

Después, repite el ataque con DP-SGD activo ($\epsilon = 1$). La reconstrucción debería degradarse a ruido — esa es la **demostración empírica** de por qué necesitas DP, no solo "no compartir datos".

### Checklist 5.8

- [ ] Reproduje DLG sobre mi FedAvg y vi imágenes reconstruidas desde gradientes
- [ ] Verifiqué empíricamente que DP-SGD bloquea el ataque
- [ ] Sé enumerar 3 categorías de ataque (inversion, membership inference, poisoning) y un ejemplo de cada
- [ ] Conozco Krum / Median / Trimmed Mean como defensas Byzantine
- [ ] Tengo un threat model claro para Pontus-X: ¿qué adversario es realista en mi setting?

---

## 5.9 — Personalized FL: cuando "un modelo para todos" no aplica

### Por qué importa — y aquí hay conexión directa con tu tesis

Hasta ahora el objetivo era un **modelo global único**. Pero si los clientes son **muy heterogéneos**, ese modelo global puede ser **peor** que el modelo entrenado localmente. **Personalized FL (pFL)** abandona la idea del modelo único y produce **un modelo por cliente**, compartiendo solo *partes* del conocimiento.

**Para tu tesis**: una de las cinco vías de tu roadmap maestro es **"Personalized Federated RL"**: cada cliente del dataspace tiene un MDP con dinámicas ligeramente distintas (cada fábrica, cada hospital, cada banco es ligeramente distinto). El objetivo no es **un** agente, sino un **meta-agente** que se adapta rápido. La conexión con **MAML** (meta-learning) es directa, y la mayoría de papers de pFRL tienen ancestros en pFL.

### Lecturas

- **Tan, Yu, Cui & Yang (2022) — "Towards Personalized Federated Learning"** → [arXiv:2103.00710](https://arxiv.org/abs/2103.00710). Survey de pFL. Lee la **taxonomía** (datos vs. modelo vs. similitud).
- **Fallah, Mokhtari & Ozdaglar (2020) — "Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach" (Per-FedAvg)** → [arXiv:2002.07948](https://arxiv.org/abs/2002.07948). MAML aplicado a FL. **Lectura clave para tu vía 3 de roadmap maestro**.
- **T Dinh, Tran & Nguyen (2020) — "Personalized Federated Learning with Moreau Envelopes" (pFedMe)** → [arXiv:2006.08848](https://arxiv.org/abs/2006.08848).
- **Li, Hu, Chen, Pu, Jiang & Hu (2021) — "Ditto: Fair and Robust Federated Learning Through Personalization"** → [arXiv:2012.04221](https://arxiv.org/abs/2012.04221). Un favorito personal del campo — simple, efectivo, doble objetivo (global + local).

### Temas mínimos

**Familia A — Fine-tuning local**: entrenas FedAvg, después cada cliente hace fine-tuning sobre sus datos. La baseline trivial. Sorprendentemente competitiva. **Familia B — Meta-learning (Per-FedAvg)**: optimizas para que el modelo global sea un buen **punto de partida** para adaptarse rápido a cada cliente. MAML federado. **Familia C — Multi-task**: cada cliente tiene su modelo $\theta_k$ y un modelo global $\theta$; loss conjunta penaliza divergencia. Ditto, pFedMe entran aquí. **Familia D — Compartir parcialmente**: backbone compartido + head personalizado (FedPer). Útil para grandes modelos donde solo las capas finales son específicas del cliente. **Familia E — Clustering**: agrupa clientes similares y entrena un modelo por cluster (IFCA). **Métrica clave**: no es "accuracy global" sino "accuracy promedio por cliente" — un modelo global puede tener accuracy global alta pero promediada con varianza alta entre clientes.

### Ejercicio (ligero)

Sobre tu FedAvg + Dirichlet de 5.3, añade dos baselines de personalización:

1. **Fine-tuning local** post-FedAvg: cada cliente entrena 5 epochs más sobre sus datos. Mide accuracy promedio por cliente vs. accuracy global.
2. **FedPer** simple: comparte todas las capas excepto la última. Implementación: ~20 líneas extra.

Verás que en non-IID alto ($\alpha=0.1$), ambos métodos **mejoran significativamente** la accuracy por cliente. Esto es la **motivación experimental** para pFL — y para tu vía 3 de tesis.

### Checklist 5.9

- [ ] Leí el survey de Tan 2022 (al menos taxonomía y conclusiones)
- [ ] Leí Per-FedAvg con foco en cómo se conecta con MAML
- [ ] Implementé fine-tuning local y FedPer como baselines personalizadas
- [ ] Sé distinguir las 5 familias de pFL y dar un ejemplo de cada
- [ ] Conecto pFL con la vía 3 de tesis ("personalized FRL")

---

## 5.10 — Flower como framework de referencia

### Por qué importa

Una vez tienes FedAvg from-scratch, **migrar a Flower te da**: simulación de cientos/miles de clientes en una sola máquina, integración nativa con PyTorch/TF/JAX, estrategias FedAvg/FedProx/FedOpt/SCAFFOLD ya implementadas, integración con Opacus para DP, y **deployment real** sobre múltiples máquinas. Es **el estándar de facto en investigación de FL hoy** y lo que vas a usar en tu doctorado.

**Versión actual a fecha de este documento**: Flower 1.25.x es la estable. Flower 1.25 introduce el comando `flwr new` para scaffolding rápido de proyectos y permite tanto simulación local como deployment distribuido con la misma abstracción. Flower nació como proyecto de investigación en la Universidad de Oxford y soporta PyTorch, TensorFlow, Hugging Face, JAX, scikit-learn, XGBoost y más, incluyendo Flower Datasets para particionados estándar.

### Recurso principal — la pieza canónica

- **Flower Documentation** → [flower.ai/docs](https://flower.ai/docs/framework/). **Tutorial series oficial**. 5–6 partes. **Hazlo entero**, en orden. Es la mejor inversión de tiempo que puedes hacer en Fase 5.
- **Flower Baselines** → [github.com/flwrlabs/flower](https://github.com/flwrlabs/flower) (carpeta `baselines/`). **Implementaciones reproducibles** de FedAvg, FedProx, SCAFFOLD, FedOpt, etc. sobre benchmarks estándar. **Equivalente a CORL pero para FL**: úsalo como ground truth de implementación.
- **Flower Datasets** → primitivas oficiales para particionar datasets en non-IID (Dirichlet, label skew, etc.). Te ahorra escribir el particionado a mano.

### Vídeo

- **Flower YouTube channel** — la serie "Federated AI Simulations with Flower" (2024–2025) es la referencia audiovisual. Cada vídeo cubre una parte del tutorial oficial. Útil si te bloqueas con la documentación escrita.

### Temas mínimos

**Arquitectura Flower**: `ServerApp` (define estrategia + número de rondas) + `ClientApp` (define training/eval local) + estrategia (FedAvg, FedProx, FedAdam, etc.). **Simulación con Ray**: `flwr run` simula $K$ clientes en paralelo con Ray como backend. **Configs**: Flower 1.x usa `pyproject.toml` para configurar el run. **Custom strategy**: heredas de `flwr.server.strategy.FedAvg` y sobreescribes `aggregate_fit` para tu lógica. **Cliente custom**: heredas de `flwr.client.NumPyClient` (más simple) o `Client` (más flexible). **Estado del cliente** (importante para SCAFFOLD): se mantiene entre rondas vía `ClientState`. **Diferenciación deploy vs. sim**: el mismo código corre en simulación (Ray) o en deployment real (gRPC entre máquinas) cambiando solo la configuración.

### Ejercicio obligatorio

1. **Replica MNIST federado IID en Flower** siguiendo el tutorial oficial. ~1 día.
2. **Reproduce tus experimentos de 5.3 (CIFAR-10 Dirichlet) en Flower**. Compara tus números contra los de Flower Baselines en `baselines/fedavg/`. Diferencia ≤5% — si es mayor, hay bug.
3. **Implementa una `CustomStrategy`** que sea FedAvg + clipping de updates a norma máxima $C$. Es el primer paso hacia DP-FedAvg y te enseña la API de strategies.
4. **Integra Opacus con Flower** para DP-FedAvg user-level. La documentación oficial tiene un ejemplo. Reproduce tu curva privacy-utility del bloque 5.6 pero ahora en Flower.

### Checklist 5.10

- [ ] Completé el tutorial oficial de Flower (las 5–6 partes)
- [ ] Reproduje mis experimentos de CIFAR-10 Dirichlet en Flower
- [ ] Mis números reproducen Flower Baselines ±5%
- [ ] Implementé al menos una `CustomStrategy` propia
- [ ] Integré Opacus con Flower para DP-FedAvg user-level
- [ ] Tengo un repo Flower limpio que puedo extender en Fase 6

---

## 5.11 — Pontus-X / Compute-to-Data — Hello World en testnet

### Por qué importa — y esta es la pieza más diferenciadora de tu perfil doctoral

La inmensa mayoría de investigadores en FL **no han tocado nunca un dataspace real**. Llegan al doctorado sabiendo Flower y pasan tres años haciendo experimentos en simulación. Tú, por tu setting de tesis, vas a llegar **sabiendo cómo se publica un algoritmo en Pontus-X, cómo se consume Compute-to-Data, y qué cuesta operacionalmente**. Eso te diferencia inmediatamente — y, más importante, **te da una intuición de qué problemas son reales y cuáles son artefactos del benchmark**.

**Pontus-X en una frase**: una blockchain customizable construida con Polygon Edge sobre Polygon Supernets, operada por un conjunto de validators europeos (Arsys, deltaDAO, EuProGigant, Exoscale, IONOS, etc.), que provee infraestructura para data spaces y digital service marketplaces conformes con Gaia-X.

**Compute-to-Data en una frase**: los datos nunca salen de la infraestructura del propietario; en su lugar, el algoritmo se envía al dato, computa allí, y solo devuelve el resultado — Federated Learning es uno de los casos de uso ejemplares.

### Recurso principal — la pieza canónica

- **Pontus-X Documentation** → [docs.pontus-x.eu](https://docs.pontus-x.eu/). El portal oficial. **Empieza por "Getting Started" y "Compute-to-Data"**.
- **Nautilus** → [nautilus.delta-dao.com](https://nautilus.delta-dao.com/). Toolkit TypeScript open-source que provee APIs para publicar datasets/algoritmos/APIs en Pontus-X, descargar assets, lanzar compute jobs, y consultar estado y resultados. **Esta es la herramienta operativa**.
- **GitHub deltaDAO** → [github.com/deltaDAO](https://github.com/deltaDAO). Repos de `pontusx-docs`, `nautilus`, `nautilus-examples`, `mvg-portal`. Los ejemplos del repo `nautilus-examples` son tu punto de partida.

### Conceptos mínimos

**Wallet + EUROe**: necesitas una wallet Ethereum-compatible (MetaMask) conectada a la red Pontus-X (testnet o mainnet) y algo de EUROe (la stablecoin del ecosistema) para pagar compute jobs. **Testnet existe** y es gratuita — empieza ahí, no en mainnet. **DID (Decentralized Identifier)**: los datasets y algoritmos se identifican por DIDs del tipo `did:op:...` y se lanzan compute jobs especificando el DID del dataset y del algoritmo. **Compute job lifecycle**: publicar algoritmo → compute job sobre dataset publicado → polling de status → fetch de resultado. **Provider y Aquarius**: componentes del stack Ocean Protocol (base técnica de Pontus-X) que orquestan compute y metadata. No necesitas implementarlos, solo saber que existen. **Self-Sovereign Identity (SSI) y Gaia-X compliance**: para mainnet hace falta credencial verificable y onboarding como participante; para testnet de desarrollo es más ligero.

### Recursos operativos

- **deltaDAO contact** para onboarding de developer network (gratis): el plan "Basic" da acceso a testnet, marketplace, y transaction logs.
- **Pontus-X Portal** ([portal](https://www.pontus-x.eu/)) — interfaz web del marketplace. Vale la pena navegarlo para ver los assets existentes (datos, algoritmos, APIs) y entender el formato.
- **Tutorials de deltaDAO en YouTube y blog** — busca videos específicos de Compute-to-Data y publicar algoritmos.

### Ejercicio obligatorio — el Hello World que vas a presentar

Este es el ejercicio **operacional más diferenciador** de toda Fase 5. **No lo saltes**:

1. **Setup**: crea wallet MetaMask, conecta a Pontus-X testnet, consigue EUROe de faucet. Onboarding básico via deltaDAO (gratis para developer network).
2. **Publica un algoritmo simple**: un script Python que lea un CSV pequeño, entrene un modelo trivial (regresión logística), y guarde el modelo serializado en el output esperado. Empaquétalo como Docker container, publícalo en Pontus-X como algorithm asset usando Nautilus o el portal.
3. **Publica un dataset pequeño** desde tu cuenta — un CSV con datos sintéticos. **Crucial**: configúralo como "Compute-to-Data only" (no descargable, solo computable).
4. **Lanza un compute job** que ejecute tu algoritmo sobre tu dataset. Verifica que el resultado vuelve (el modelo serializado), pero el dataset crudo **nunca lo descargas**.
5. **Bonus** (si tienes tiempo y margen): publica un algoritmo de inferencia federada — una variante del paso 2 que reciba pesos del cliente, entrene una época local, y devuelva los pesos actualizados. Esto es **el bloque base de tu pipeline FRL de Fase 6**.

**Documenta cada paso en `notes_operations.md`**. Vas a olvidar la mitad de las decisiones técnicas (direcciones de contratos, DIDs, configuraciones de provider) y vas a necesitar este documento muchas veces.

### Checklist 5.11

- [ ] Tengo wallet conectada a Pontus-X testnet y EUROe de faucet
- [ ] Publiqué un algoritmo simple como asset en testnet
- [ ] Publiqué un dataset pequeño con configuración Compute-to-Data only
- [ ] Lancé un compute job exitoso y recuperé el resultado sin descargar el dataset crudo
- [ ] Tengo `notes_operations.md` con todas las direcciones, DIDs y comandos
- [ ] Conozco la diferencia testnet vs. mainnet y qué requiere cada una
- [ ] (Bonus) Algoritmo de "una época de training federado" publicado y testeado

---

## 5.12 — Disciplina experimental específica de FL

### Por qué importa

La disciplina experimental de Fase 2 (≥5 seeds, IQM, `rliable`) **sigue aplicando**, pero FL introduce **patologías propias** que tienes que conocer:

- **Particionado no especificado**: la mayoría de papers reportan "non-IID" sin decir si es Dirichlet $\alpha=0.1$, partición por label, quantity skew, o qué. **Una tabla de resultados sin especificar la partición es irreproducible**. Sé explícito siempre.
- **Comparación "FedX vs. FedAvg" tramposa**: si FedAvg está mal tuneado, cualquier cosa lo gana. **Reportar baselines bien tuneados es no negociable** — y eso suele significar más esfuerzo en FedAvg que en tu propio método.
- **Número de rondas insuficiente**: en non-IID, muchos métodos parecen mejores a 50 rondas pero convergen al mismo punto a 500. **Reporta convergencia a *budget* fijo, no a accuracy fijo**.
- **Número de clientes irreal**: $K=10$ clientes es un toy benchmark. Real cross-silo: $K \in [5, 100]$. Real cross-device: $K \in [10^3, 10^7]$. **Reportar en el régimen relevante a tu setting** (cross-silo para Pontus-X).
- **Privacy accounting honesto**: si dices "DP con $\epsilon=1$", reporta también $\delta$, mecanismo de accounting (RDP, moments accountant, Gaussian DP), y si es sample-level o user-level.

### Lecturas

- **Reddi et al. 2020 (FedOpt paper, ya citado)**: la sección experimental es un buen ejemplo de cómo reportar en FL.
- **Caldas, Wu, Li, Konečný, McMahan, Smith & Talwalkar (2018) — "LEAF: A Benchmark for Federated Settings"** → [arXiv:1812.01097](https://arxiv.org/abs/1812.01097). El benchmark canónico de FL con datasets realistas (FEMNIST, Shakespeare, Reddit). Conócelo.
- **FedScale, FedLab, FedML**: otros benchmarks/frameworks. No te disperses, pero conoce los nombres.

### Disciplina mínima específica de FL

- **Especifica siempre el particionado**: "Dirichlet $\alpha=0.1$, $K=10$, $C=1.0$, $E=5$, $B=32$" es suficiente; "non-IID" no lo es.
- **Reporta accuracy global Y accuracy promedio por cliente** (con varianza). Son métricas distintas y ambas importan.
- **Tune FedAvg como baseline tan rigurosamente como tunes tu método**. Si no, tu mejora puede ser fantasma.
- **Compara contra Flower Baselines** cuando estén disponibles para tu setup. Tratalos como ground truth de implementación.
- **Privacy accounting completo**: $\epsilon$, $\delta$, level (sample/user), mecanismo, librería usada.
- **Curva accuracy vs. ronda**, no solo accuracy final. La dinámica de convergencia revela patologías.

### Checklist 5.12

- [ ] Especifico siempre particionado completo en mis experimentos
- [ ] Reporto accuracy global y accuracy promedio por cliente
- [ ] Tuneo FedAvg como baseline con la misma rigurosidad que mi método
- [ ] Comparo contra Flower Baselines cuando aplica
- [ ] Reporto privacy accounting completo cuando uso DP
- [ ] Sé enumerar 5 errores comunes de papers de FL para evitarlos

---

## 5.13 — Proyecto integrador: FedAvg + non-IID + DP + Pontus-X

### Por qué importa

**Este es el segundo artefacto pre-doctoral de tu carrera** (el primero fue el de Fase 4). A diferencia de aquél, **éste ya combina dos mundos**: el académico (FL clásico con experimentos rigurosos) y el operacional (Pontus-X deploy). Esa combinación es **lo que muy pocos candidatos doctorales pueden mostrar**.

### Estructura mínima del repo

```
federated-learning-foundation/
├── README.md                          # narrativo, gráficas, tablas, link al repo de Fase 4
├── algorithms/
│   ├── fedavg.py                      # from-scratch, sin Flower
│   ├── fedprox.py                     # extiende fedavg.py
│   └── fedopt.py                      # variante server-side adaptive
├── flower_app/                        # mismo experimento pero en Flower
│   ├── client_app.py
│   ├── server_app.py
│   ├── strategy_custom.py             # FedAvg + clipping + DP
│   └── pyproject.toml
├── datasets/                          # interfaces a Flower Datasets (Dirichlet, label skew)
├── experiments/
│   ├── e1_iid_vs_noniid.py            # reproduce McMahan 2017 fig 1
│   ├── e2_partition_sensitivity.py    # accuracy vs alpha Dirichlet
│   ├── e3_algorithm_comparison.py     # FedAvg vs FedProx vs FedOpt vs SCAFFOLD
│   ├── e4_dp_privacy_utility.py       # curva eps vs accuracy con Opacus
│   └── e5_gradient_inversion.py       # DLG sobre FedAvg con/sin DP
├── pontus_x/                          # ← la carpeta diferenciadora
│   ├── README.md                      # qué hace cada algoritmo publicado
│   ├── algorithm_inference/           # Hello World: inferencia C2D
│   ├── algorithm_local_train/         # bonus: una época local federada
│   ├── notes_operations.md            # DIDs, contratos, comandos, gotchas
│   └── docker/                        # Dockerfiles de los algoritmos
├── configs/                           # Hydra YAMLs por experimento
├── figures/                           # generadas por rliable + matplotlib
├── reports/
│   └── fl_foundation_report.pdf       # 6 páginas: mini-paper de Fase 5
├── thesis_relevant/                   # ← carpeta nueva-en-Fase-5
│   ├── README.md                      # qué de este repo entra en tu tesis
│   ├── notes_for_supervisor.md        # análisis tesis-relevante
│   └── gaps_matrix_fl_to_frl.md       # qué de FL clásico no se sabe extender a FRL
├── tests/
├── requirements.txt
└── notebooks/
```

### Tabla obligatoria — la que vas a defender ante tu director

| Algoritmo | CIFAR-10 IID | CIFAR-10 $\alpha=0.5$ | CIFAR-10 $\alpha=0.1$ | CIFAR-10 $\alpha=0.05$ | + DP $\epsilon=4$ | + DP $\epsilon=1$ |
|---|---|---|---|---|---|---|
| Local-only | | | | | — | — |
| FedAvg | | | | | | |
| FedProx | | | | | | |
| FedOpt | | | | | | |
| SCAFFOLD (opc) | | | | | | |

≥3 seeds, accuracy media y std. **La columna "Local-only" es el suelo, la columna "IID" es el techo**: cualquier algoritmo serio debería caer en medio en non-IID, y degradarse de forma controlada con DP. **Esta tabla replica el ejercicio de la Fase 4 con su tabla de offline RL** — esa simetría no es casual, es la prueba de que dominas las dos mitades.

### Mini-paper de Fase 5 (6 páginas)

Formato NeurIPS/ICML. Estructura:

1. **Intro**: motivación, conexión con tesis (FRL sobre Compute-to-Data), reseña rápida de Fase 4.
2. **Background**: 1 página sobre FL clásico, non-IID, DP, secure aggregation.
3. **Algoritmos**: descripción rápida de FedAvg/FedProx/FedOpt + DP-FedAvg.
4. **Experimentos**: tabla principal, curva privacy-utility, reproducción de DLG con/sin DP.
5. **Deployment en Pontus-X**: 1 página — qué publicaste, qué cuesta, qué cuellos de botella detectaste.
6. **Análisis tesis-relevante**: ¿qué de FL clásico sobrevive cuando los datos son trayectorias RL? Hipótesis (sin demostrar todavía).
7. **Trabajo futuro**: el puente a Fase 6 — FRL on/offline.

### `thesis_relevant/notes_for_supervisor.md`

Actualización del documento que abriste en Fase 4. Ahora añades:

- **Tu análisis de qué primitivas de FL son fáciles de portar a RL** (gradient averaging) y cuáles no (DP user-level cuando una "unidad" es una trayectoria larga).
- **Una matriz de gaps actualizada**: ahora la cruza de "FL clásico" con "Offline RL" — qué de cada uno se ha combinado, qué no.
- **Cinco preguntas concretas** para tu supervisor potencial sobre la dirección de tesis. Más afiladas que las de Fase 4 porque ya tienes infraestructura.
- **Decisión preliminar de vía de tesis**: de las 5 vías del roadmap maestro (Offline FRL non-IID, FRL con privacy, Personalized FRL, comunicación-eficiente, Sim2Real federado), **cuál te parece más prometedora y por qué**. Esta decisión no es definitiva, pero **mostrar que tienes una opinión informada vale mucho** ante un supervisor.

### `thesis_relevant/gaps_matrix_fl_to_frl.md`

Documento corto, 1–2 páginas, con una tabla cruzada:

|  | Estudiado en FL | Estudiado en RL offline | Estudiado en FRL | Gap |
|---|---|---|---|---|
| Non-IID Dirichlet | ✅ | ❌ | ⚠️ poco | ⭐⭐⭐ |
| DP user-level con trayectorias | ✅ (samples) | ❌ | ❌ | ⭐⭐⭐⭐ |
| Compresión de comunicación + offline RL | ✅ | ❌ | ❌ | ⭐⭐⭐⭐ |
| Personalized + meta-learning + offline RL | ✅ | ⚠️ | ❌ | ⭐⭐⭐ |
| Byzantine-robust + offline RL | ✅ | ❌ | ❌ | ⭐⭐⭐⭐ |
| OPE federado | ❌ | ✅ | ❌ | ⭐⭐⭐⭐⭐ |

**Las celdas con ⭐⭐⭐⭐ o más son donde puede estar tu contribución doctoral**. Esta tabla **es el activo más valioso del repo** para una conversación con un supervisor: muestra lectura crítica del estado del arte y formulación de gaps específicos.

### Checklist 5.13

- [ ] Repo público con FedAvg/FedProx/FedOpt en versión from-scratch y versión Flower
- [ ] Tabla de comparación accuracy × algoritmo × $\alpha$ × DP completa
- [ ] DLG reproducido con y sin DP, figuras incluidas
- [ ] Pontus-X deploy funcional, documentado en `pontus_x/notes_operations.md`
- [ ] Mini-paper de 6 páginas en `reports/`
- [ ] `notes_for_supervisor.md` actualizado con decisión preliminar de vía de tesis
- [ ] `gaps_matrix_fl_to_frl.md` completa
- [ ] CI verde, tests, configs en Hydra, logging W&B

---

## Stack técnico canónico (resumen)

Para que no busques estas decisiones cada vez:

- **Framework FL principal**: **Flower 1.x** ([flower.ai](https://flower.ai/)). Single source of truth. Tu herramienta de trabajo durante todo el doctorado.
- **Reproducción de baselines**: **Flower Baselines** (carpeta `baselines/` en el repo flwr/flower). Tu CORL para FL.
- **Datasets**: **Flower Datasets** para particionados estándar; **LEAF** para datasets realistas (FEMNIST, Shakespeare).
- **DP**: **Opacus** ([opacus.ai](https://opacus.ai/)). Integración estándar con Flower.
- **Criptografía** (cuando aplique, conceptual): **PySyft / OpenMined** ([openmined.org](https://openmined.org/)) para SA y MPC; **TenSEAL** para HE.
- **Ataques**: implementaciones de DLG en GitHub (varias, busca "deep leakage gradients pytorch").
- **Dataspace**: **Pontus-X** ([docs.pontus-x.eu](https://docs.pontus-x.eu/)) + **Nautilus** ([nautilus.delta-dao.com](https://nautilus.delta-dao.com/)).
- **Evaluación estadística**: **`rliable`** (sigue siendo el estándar transversal).
- **Tracking**: **W&B**.

---

## Pre-final: lo que se ve cuando se cierra la Fase 5

Cuando termines esto, deberías poder:

- Leer cualquier paper de FL del 2020–2026 sin perderte en la notación o los acrónimos. Identificar en 2 minutos qué supuesto rompe y qué fix propone.
- Implementar FedAvg desde cero en una tarde, FedProx en 15 minutos, FedOpt en una hora.
- Defender en pizarra cómo se combinan DP user-level y Secure Aggregation, qué garantiza cada uno, y qué cuesta.
- Saber publicar un algoritmo en Pontus-X testnet, lanzar un compute job, y debugear cuando falla. Esto **te diferencia operativamente del 99% de candidatos doctorales en FL**.
- Tener una **opinión formada y defensible** sobre cuál de las 5 vías de tesis del roadmap maestro es la más prometedora para tu setting.
- Leer un paper de **Federated Reinforcement Learning** (Fase 6) y reconocer **exactamente** qué algoritmo de FL clásico está debajo (FedAvg/FedProx/FedOpt), qué problema RL específico añade, qué fixes ya están explorados de FL y cuáles habría que reinventar. **Ese reconocimiento es la prueba de que Fase 5 está cerrada y Fase 6 puede empezar a tomar forma real.**

---

## Plan sugerido de 6 semanas

| Semanas | Foco principal | Foco secundario |
|---|---|---|
| 1 | 5.1 Setting + 5.2 FedAvg from-scratch | Reproduce fig 1 de McMahan 2017 |
| 2 | 5.3 Non-IID + 5.4 FedProx/FedOpt/SCAFFOLD | Curva $\alpha$ Dirichlet, tabla comparativa |
| 3 | 5.6 DP-SGD con Opacus + 5.5 sistemas (ligero) | Curva privacy-utility centralizada y federada |
| 4 | 5.10 Migración a Flower + 5.8 ataques (DLG) | Reproduce experimentos en Flower, DLG con/sin DP |
| 5 | 5.11 Pontus-X — Hello World en testnet | Setup wallet, publicación algoritmo+dataset, compute job |
| 6 | 5.7 + 5.9 + 5.12 + 5.13 mini-paper y repo | Documento para supervisor, gaps matrix, cierre |

Si el bloque 5.11 (Pontus-X) te lleva más de una semana, **estíralo sin culpa**. Es lo más diferenciador operacionalmente y lo menos cubierto en cualquier curso de FL.

---

## Cómo seguimos

Cuando estés listo, dime qué bloque quieres profundizar primero y abrimos un documento dedicado con:

- Implementaciones de referencia comentadas (FedAvg, FedProx, DP-FedAvg con Opacus, custom strategies en Flower)
- Análisis críticos de los papers principales — qué prometen, qué cumplen, qué supuesto esconden
- Walkthrough operacional de Pontus-X con comandos concretos, ejemplos de DIDs, contratos en testnet, y troubleshooting de problemas típicos
- Conexiones específicas con tu pipeline de Fase 4 (CQL/IQL) para preparar la fusión en Fase 6

Mi recomendación de orden:

1. **5.1 + 5.2 + 5.3 como un bloque dedicado** (setting + FedAvg + non-IID): es la columna conceptual. Sin esto sólido, el resto son trucos sueltos.
2. **5.6 + 5.8 como otro bloque dedicado** (DP + ataques): aquí es donde "privacy en FL" deja de ser slogan y se vuelve garantía cuantificable. Sin esto, no puedes argumentar privacy en tu tesis.
3. **5.11 como bloque dedicado y operacional**: Pontus-X tiene su propia lógica (criptografía, blockchain, smart contracts, Ocean Protocol) que no aparece en ningún curso de FL. Vas a necesitar acompañamiento técnico paso a paso.
4. **5.13 como bloque final**: aquí cierras el artefacto pre-doctoral.

Y un aviso final, que repito porque vale repetir: **Fase 5 es preparación instrumental**. La frontera de tu tesis no está aquí — está en Fase 6, en la composición de Offline RL con Federated Learning sobre dataspaces. Pero **sin Fase 5 sólida y sin Pontus-X funcionando, Fase 6 es ciencia ficción**. Trata esta fase como infraestructura: invierte lo justo para que sea robusta y reusable, no más. Cuando una semana antes de Fase 6 vuelvas a este repo y todo siga corriendo limpio, es cuando sabrás que la inversión fue correcta.
[]()