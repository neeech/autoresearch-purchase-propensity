# Purchase Propensity con AutoResearch

Proyecto de Machine Learning que estima la probabilidad de compra asociada a
una sesión web y explora mejoras del modelo mediante un agente de AutoResearch.
Incluye evaluación de estrategias de marketing simuladas, seguimiento con MLflow
y preparación de inferencia batch en AWS.

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
   y AutoResearch independiente desde LightGBM sin tuning. La regresión logística se conserva
   como antecedente.
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

## Estado del proyecto

- **Implementado:** procesamiento temporal, baselines, AutoResearch, evaluación
  de costes simulados, tracking y Model Registry locales con MLflow e inferencia Docker verificada.
- **AWS preparado:** modelo en S3, imagen en ECR y configuración de IAM y
  Batch Transform. La primera ejecución no se ha verificado; requiere consultar la cuota vigente.
- **Próximos pasos:** verificar el batch en AWS, aprobar una versión
  y automatizar inferencia sobre lotes diarios simulados.

El entrenamiento y la aprobación de modelos siguen siendo manuales. Los modelos,
predicciones y registros locales de MLflow quedan fuera de Git.

## Documentación

- [Arquitectura del sistema](docs/architecture.md)
- [Contrato de datos y faltantes](docs/data-contract.md)
- [AutoResearch: ejecución, restricciones y recuperación](docs/autoresearch.md)
- [Evaluación, umbrales y MLflow](docs/evaluation.md)
- [Model Registry y recuperación](docs/model-registry.md)
- [Comparación de test y candidato](docs/test-results.md)
- [Resultados de validación](docs/independent-results.md)
- [Resultados históricos de marketing](docs/marketing-results.md)
- [Inferencia con Docker](inference/README.md)
- [Permisos AWS](infra/iam/README.md) y [Batch Transform](infra/batch/README.md)

La comparación actual evalúa **tasa constante, LightGBM con Optuna y AutoResearch
independiente desde LightGBM sin tuning**. Ver [protocolo y comandos](docs/model-comparison.md).
La corrida anterior que inició desde Optuna se conserva como
[antecedente incremental](docs/autoresearch-optuna-results.md), no como una
comparación independiente de estrategias.

La siguiente metodología propuesta separa AutoResearch para features y Optuna
para tuning; aún no está implementada. El [protocolo](docs/model-comparison.md)
distingue este plan de los experimentos ya realizados.
