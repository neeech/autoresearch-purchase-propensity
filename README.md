# Purchase Propensity con AutoResearch — proyecto cerrado

**Cerrado el 14/11/2026.** La formulación clasifica compras de sesiones ya
terminadas; no permite justificar su uso para anticipar compras y activar campañas.
El código y los resultados se conservan como referencia técnica, no como solución
lista para producción. Ver [motivo del cierre y recursos AWS](docs/project-closure.md).

Pipeline de Machine Learning para clasificar sesiones web según su probabilidad
de compra. Incluye preparación temporal de datos, comparación de modelos,
seguimiento con MLflow y un contenedor de inferencia preparado para AWS.

**Objetivo operativo original, no completado:** procesar lotes de sesiones nuevas una vez terminadas,
guardar sus probabilidades y generar un reporte trazable por ejecución. La
inferencia fue verificada localmente; la ejecución batch en AWS sigue pendiente.

## El problema

A partir de la actividad de una sesión —páginas vistas, clicks, duración, origen
del tráfico y dispositivo— y del historial disponible del usuario, se busca
identificar sesiones con compra.

Los datos corresponden a **sesiones terminadas**. Por eso, el proyecto realiza
una clasificación retrospectiva; no anticipa una compra durante la navegación
ni mide el efecto de una campaña de marketing.

El target es `transactions > 0`, asumiendo cero cuando faltan transacciones.
La historia utiliza únicamente sesiones de días anteriores. El entrenamiento
usa enero–abril de 2017 y la validación, mayo. Junio permanece fuera del loop de
experimentación y se usa como test temporal con modelos y umbrales congelados.
Una evaluación histórica previa impide considerarlo ciego para todo el proyecto.

El [notebook de análisis exploratorio](notebooks/01_eda.ipynb) presenta la estructura,
calidad y evolución de los datos, y justifica las principales decisiones.
Para ejecutarlo: `uv sync --extra eda` y seleccionar el Python de `.venv`.

## Cómo funciona

```text
Sesiones → procesamiento e historia → baselines → AutoResearch
                                                     ↓
                                  evaluación y selección de umbrales
                                                     ↓
                                      registro local en MLflow
                                                     ↓
                                   modelo e imagen preparados para AWS
                                                     ↓
                                   Batch Transform → predicciones en S3
```

1. **Preparación:** normalización de datos, historia por usuario y separación temporal.
2. **Comparación actual:** tasa de compra constante, LightGBM optimizado con Optuna
   y AutoResearch independiente desde LightGBM sin tuning. La regresión logística queda disponible como baseline local.
3. **AutoResearch:** un LLM propone cambios en features y modelos. El controlador
   ejecuta cada experimento y conserva sólo mejoras de **log loss** en validación.
4. **Evaluación:** se eligen umbrales según penalizaciones simuladas y se comparan
   contra estrategias como contactar a todos o no contactar a nadie.
5. **Trazabilidad y despliegue:** MLflow registra evaluaciones; Docker, S3, ECR y
   SageMaker permiten preparar la ejecución del modelo por lotes.

## Resultados de test

Comparación sobre **63.578 sesiones de junio**, con 946 compras. Modelos y
umbrales seleccionados en mayo y congelados antes de esta evaluación:

| Modelo | Log loss ↓ | Penalización simulada total ↓ |
|---|---:|---:|
| Tasa constante | 0,077659 | 9.460 |
| LightGBM + Optuna | 0,030528 | 7.382 |
| AutoResearch independiente | **0,029069** | **6.930** |

AutoResearch mejora log loss un **4,78 %** frente a Optuna en junio. Las búsquedas
usaron presupuestos y libertades distintos: 20 ensayos Optuna y 10 propuestas
AutoResearch. Las penalizaciones son simuladas, no retorno causal de marketing.
Ver [resultados de test y auditoría de junio](docs/test-results.md) y
[selección en mayo](docs/independent-results.md).

## Empezar

Requisitos: **Python ≥ 3.13**, [uv](https://docs.astral.sh/uv/) y Git configurado.
Colocar el dataset en `data/raw/data.csv`.

```bash
uv sync
uv run python -m autoresearch_purchase_propensity.preprocess
uv run python -m autoresearch_purchase_propensity.train
```

Para ejecutar AutoResearch, configurar `GROQ_API_KEY` en el entorno y comenzar
con los cambios del repositorio guardados en Git:

```bash
uv run autoresearch --experiments 10 --timeout 180 --rpm 10
```

Cada investigación conserva hipótesis, cambios, métricas y errores en un workspace
separado. Por defecto el código comienza con regresión logística; para repetir la
comparación independiente usar `--config configs/lightgbm_initial.json`. El
candidato seleccionado queda identificado en `configs/candidate.json`. Los
artefactos de investigación se generan localmente; no se descargan con el repositorio.
Ver [configuración y reanudación de AutoResearch](docs/autoresearch.md).

Para ejecutar las pruebas:

```bash
uv run python -m unittest discover -s tests -v
```

## Estado operativo

- Datos, entrenamiento, búsquedas y evaluación temporal implementados.
- Modelo versionado en MLflow, con código, dependencias y procedencia.
- Contenedor LightGBM verificado con 64 sesiones; imagen y paquete publicados en
  ECR y S3, con definición de modelo SageMaker.
- Pendiente de verificar: primer Batch Transform, procesamiento de lotes nuevos,
  reporte por ejecución y programación diaria. La cuota limita el ensayo AWS.

La aprobación del modelo es manual. Pesos, predicciones y estado local de MLflow
no se distribuyen en Git. La muestra usada para probar el contenedor proviene de
validación; no representa datos nuevos ni una campaña real.

## Documentación

- [Arquitectura del sistema](docs/architecture.md)
- [Contrato de datos y faltantes](docs/data-contract.md)
- [AutoResearch: ejecución, restricciones y recuperación](docs/autoresearch.md)
- [Evaluación, umbrales y MLflow](docs/evaluation.md)
- [Model Registry y recuperación](docs/model-registry.md)
- [Comparación de test y candidato](docs/test-results.md)
- [Resultados de validación](docs/independent-results.md)
- [Inferencia con Docker](inference/README.md)
- [Permisos AWS](infra/iam/README.md) y [Batch Transform](infra/batch/README.md)
