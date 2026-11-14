# Comparación de modelos y siguiente metodología

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
que la evaluación es exploratoria. Junio ya observado no es un test ciego.

## Siguiente metodología: features con AutoResearch y tuning con Optuna

**Propuesta, todavía no implementada.** Separar responsabilidades para comparar:

1. Tasa constante de entrenamiento.
2. LightGBM con columnas originales permitidas y tuning Optuna. La limpieza y
   codificación necesarias son comunes a todos los modelos.
3. LightGBM con features propuestas por AutoResearch y tuning Optuna final.

Durante la exploración de features se congelarían estimador e hiperparámetros.
El agente sólo podría editar el código de features; esa restricción debe aplicarse
en el controlador, no sólo en el prompt. El controlador actual aún permite
features.py y modeling.py, por lo que no se debe presentar como modo features-only.

No hace falta tuning completo después de cada propuesta. La estrategia prevista
es evaluar features con parámetros fijos y retunear pocos finalistas con el mismo
espacio y presupuesto Optuna que el baseline. Puede descartar features que sólo
funcionen con otros parámetros; retunear finalistas reduce, pero no elimina, ese
riesgo. Un tuning por propuesta es posible, pero multiplica los entrenamientos.

Antes de implementarlo, definir validación temporal interna para búsqueda y un
período posterior para comparar diseños congelados. No llamar independiente a
una evaluación usada para orientar propuestas. Los resultados actuales se
conservan y no se reinterpretan como evidencia del esquema híbrido.
