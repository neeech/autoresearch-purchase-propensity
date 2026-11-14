# Comparación retrospectiva de marketing — mayo de 2017

Evaluación realizada el 13/11/2026. Sin reentrenar ni utilizar junio. Se recuperó cada modelo con su snapshot original de features y se reprodujeron log loss y Average Precision guardados en sus entrenamientos.

65.371 sesiones; 1.115 con compra (1,706 %). Penalizaciones principales: TP=1, TN=0, FP=5, FN=10. Son unidades simuladas, no dinero ni uplift.

| Modelo/estrategia | Log loss | Umbral | Coste medio | Precisión | Recall | Contactos |
|---|---:|---:|---:|---:|---:|---:|
| Probabilidad constante | 0.087228 | 1.000000 | 0.170565 | 0.00% | 0.00% | 0 |
| Regresión logística | 0.038736 | 0.184553 | 0.133469 | 48.97% | 57.40% | 1307 |
| HistGradientBoosting seleccionado | 0.033616 | 0.238546 | 0.123816 | 51.28% | 64.48% | 1402 |

El umbral de la referencia constante es apenas mayor que 1: representa no contactar a nadie. La precisión sin contactos se informa como 0 por convención.

El candidato reduce el coste principal 7.23% frente a regresión logística y 27.41% frente a no contactar a nadie. Contactar a todos cuesta 4,931774 por sesión.

## Sensibilidad a las penalizaciones

| Escenario | Coste logístico | Coste candidato |
|---|---:|---:|
| budget | 0.085221 | 0.084181 |
| balanced | 0.083141 | 0.079301 |
| opportunities | 0.133469 | 0.123816 |

## Alcance y trazabilidad

- Cada modelo elige su propio umbral en mayo. Ese mes también se utilizó para seleccionar el candidato; los costes mínimos son descriptivos y potencialmente optimistas.
- Los umbrales quedan guardados para aplicarlos sin reajuste a otro período. Junio ya observado no es un test completamente ciego.
- TP/FP describen selección retrospectiva; no se midió respuesta a una campaña ni compras futuras.
- Candidato: commit `0a29149a136fe2fb3ee193f69df39f2a02cd676b`, run_005. Baseline logístico: run_000 de la misma investigación `20261027T193002594365Z`.
- El candidato conserva early stopping y un warning conocido al dividir historia vacía. Se reprodujeron probabilidades finitas y métricas, pero sigue sin aprobación final.
- Resultados detallados, hashes de modelo/datos y predicciones: `experiments/evaluation/marketing-v1/` (local, ignorado).
- MLflow local registra estas evaluaciones como no aprobadas; todavía no registra un modelo desplegable en Model Registry.

Runs MLflow:
- `constant-run000`: `19ff9e0dbfc640c48c2250db4cf7c33a`
- `0a29149a-run005`: `90cb63aa203d48459f9d7f5d7374b3f5`
- `logistic-run000`: `409d0a2cbdbb4f1790f66e82b11ae41d`
