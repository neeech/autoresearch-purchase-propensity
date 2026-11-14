# Comparación de modelos

## Comparación implementada

Se compararon tasa constante, LightGBM con Optuna y AutoResearch independiente.
La referencia constante usa la tasa de compra de train; Optuna optimiza parámetros
con features de sesión fijas; el agente puede modificar features y modelos.

| Condición | Optuna | AutoResearch independiente |
|---|---|---|
| Entrenamiento | Enero–abril de 2017 | Enero–abril de 2017 |
| Validación/objetivo | Mayo / minimizar log loss | Mayo / minimizar log loss |
| Presupuesto ejecutado | 20 ensayos | 10 propuestas + entrenamiento inicial |
| Punto de partida | Espacio LightGBM predefinido | [LightGBM inicial](../configs/lightgbm_initial.json) |
| Libertad | Hiperparámetros | Features, estimador e hiperparámetros |

Semilla 42; CPU local; LightGBM con 4 threads, sin early stopping en esta
comparación. Se verificaron hashes iguales de datos y alineación de sesiones.
Los presupuestos no son equivalentes: los resultados no prueban superioridad
universal de un optimizador. Ver [resultados](independent-results.md).
La [investigación incremental anterior](autoresearch-optuna-results.md) comenzó
desde el ganador de Optuna y es un antecedente distinto.

## Reproducir una búsqueda desde el repositorio

Desde la raíz, generar primero las particiones:

```bash
uv sync --extra tuning --extra tracking
uv run python -m autoresearch_purchase_propensity.preprocess
uv run --extra tuning --extra tracking python -m autoresearch_purchase_propensity.tune \
  --trials 20 --output experiments/optuna/reproduction --mlflow
```

El directorio de salida debe ser nuevo. Los archivos de estudio, modelos y
predicciones se **generan con el comando** y no están incluidos en Git.

Para AutoResearch independiente, configurar GEMINI_API_KEY en el entorno y
comenzar con checkout limpio:

```bash
uv run autoresearch --provider gemini --model gemini-3.6-flash \
  --config configs/lightgbm_initial.json --experiments 10 --timeout 180 --rpm 10
```

El modelo del proveedor debe estar disponible para la cuenta. El controlador crea
un workspace nuevo; no usar --resume para una comparación independiente. El
baseline comienza sin tuning de Optuna. Ver [opciones y recuperación](autoresearch.md).
Una reproducción genera artefactos nuevos; no descarga pesos históricos.

## Evaluación común

Seleccionar umbrales por modelo bajo las mismas penalizaciones, con el procedimiento
versionado en [evaluation.md](evaluation.md). Distinguir gasto por contactos de
penalización por errores/oportunidades. Mayo se reutiliza para selección, por lo
que sus métricas son exploratorias. Junio se usa como test temporal de decisiones
congeladas; la exposición histórica confirmada está documentada en
[test-results.md](test-results.md).
