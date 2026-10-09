# Referencia de autoresearch: cierre

Esta ejecución constituye el baseline de la skill autoresearch para comparar posteriormente con una tool propia en otra branch. Branch: `autoresearchToolTest`.

## Metodología y decisiones del agente

El agente fue el asistente de esta conversación; el runner sólo entrenó y registró candidatos. El agente leyó el EDA y los resultados acumulados, propuso hipótesis y eligió el siguiente cambio. No hubo una API de LLM dentro del runner ni un optimizador automático.

Decisiones tomadas: cantidad de árboles, learning rate, hojas, mínimo de sesiones por hoja, L2 y muestreo. Tras cada resultado, conservó el candidato si reducía log loss; Brier se informó como secundaria. Cada cinco ejecuciones revisó qué regiones convenía explorar. Ejemplo: 63 hojas empeoró; reducir learning rate y muestrear filas mejoró. La selección de 47 hojas fue un empate práctico con 31, no evidencia de superioridad estadística.

Datos, features, splits y métricas quedaron fijos. Enero–abril para entrenar, mayo para seleccionar, junio para evaluación final. CPU local, cuatro hilos, semilla 42, sin early stopping, pesos de clase ni calibración. Presupuesto: 20 entrenamientos o 60 minutos; terminó por cantidad. Fueron 19 configuraciones y una repetición exacta del ganador. Protocolos y snapshots se commitearon antes del entrenamiento; todos los resultados se conservaron.

## Modelo congelado

Run_017: 300 árboles, learning_rate=0,025, num_leaves=47, min_child_samples=30, reg_lambda=1, subsample=0,6 y subsample_freq=1. Entrenamiento determinista y faltantes/categorías nativos de LightGBM. Features actuales, con historia y sin month.

La selección y los hashes de ambos modelos se congelaron en el commit `5ec76f1`, antes de evaluar junio. No se reentrenó con mayo ni se ajustó nada usando junio. La repetición del ganador y la recarga del modelo reprodujeron sus predicciones.

## Resultados

| Período | Modelo | Log loss ↓ | Brier ↓ |
|---|---|---:|---:|
| Mayo: validación | Logística | 0,03757169 | 0,01132709 |
| Mayo: validación | LightGBM | **0,03350781** | **0,01037929** |
| Junio: test | Logística | 0,03325592 | 0,00998270 |
| Junio: test | LightGBM | **0,02980344** | **0,00921297** |

En junio, LightGBM reduce log loss **10,38%** y Brier **7,71%** respecto a logística. Hay 63.578 sesiones y 946 positivas (1,49%). La mejora se mantiene en ambos segmentos:

| Historia en junio | Sesiones | Log loss logística | Log loss LightGBM | Brier logística | Brier LightGBM |
|---|---:|---:|---:|---:|---:|
| Sin días anteriores | 52.512 | 0,02400923 | **0,02174291** | 0,00695339 | **0,00648295** |
| Con días anteriores | 11.066 | 0,07713469 | **0,06805342** | 0,02435784 | **0,02216786** |

Los segmentos tienen tasas de compra distintas; sus pérdidas no son directamente comparables entre sí. Los resultados globales de mayo y junio tampoco prueban una mejora entre meses: cambian prevalencia y población. Son métricas de sesiones terminadas, no de anticipación online. Se mantiene el supuesto de transacciones vacías=0.

## Comparación futura con la tool propia

Usar el mismo dataset, features, entrenamiento, selección en mayo, métricas, recursos y presupuesto (incluyendo repeticiones y errores). Registrar versiones, semillas y propuestas para distinguir diferencias de metodología de diferencias de recursos. Comparar métricas finales y trayectoria de mejora, además de tiempo y cantidad de intentos.

Junio ya fue observado: no ajustar la tool propia para mejorar en junio. Congelar también su candidato antes de compararlo. Una nueva evaluación independiente exigiría otro período no utilizado. El costo en tokens/tiempo del agente no se instrumentó en esta referencia; no se puede afirmar igualdad de ese costo.

## Artefactos

- [Historial de búsqueda](../experiments/lightgbm/history.csv).
- [Configuración seleccionada](../experiments/lightgbm/run_017/config.json) y [modelo local](../experiments/lightgbm/run_017/model.txt).
- [Protocolo congelado](../configs/final_evaluation.json).
- [Métricas finales completas](../experiments/final_test/metrics.csv) y [hashes de test](../experiments/final_test/provenance.json).
- [Resumen estructurado](../experiments/lightgbm/summary.json).

La búsqueda y su evaluación final están cerradas. Los modelos y predicciones se conservan localmente y están excluidos de Git; código, protocolos, configuraciones y métricas están versionados.
