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
experimentación, aunque ya fue observado y no constituye un test ciego.

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
   y AutoResearch iniciado desde esa referencia. La regresión logística se conserva
   como antecedente.
3. **AutoResearch:** un LLM propone cambios en features y modelos. El controlador
   ejecuta cada experimento y conserva sólo mejoras de **log loss** en validación.
4. **Evaluación:** se eligen umbrales según penalizaciones simuladas y se comparan
   contra estrategias como contactar a todos o no contactar a nadie.
5. **Trazabilidad y despliegue:** MLflow registra evaluaciones; Docker, S3, ECR y
   SageMaker permiten preparar la ejecución del modelo por lotes.

## Resultados de la investigación anterior

Evaluación sobre **65.371 sesiones de mayo**, de las cuales 1.115 tuvieron compra:

| Modelo | Log loss ↓ | Penalización media ↓ |
|---|---:|---:|
| Probabilidad constante | 0,087228 | 0,170565 |
| Regresión logística | 0,038736 | 0,133469 |
| HistGradientBoosting seleccionado | **0,033616** | **0,123816** |

En el escenario principal, el candidato reduce la penalización un **7,23 %**
frente a la regresión logística. Los umbrales se seleccionaron en la misma
validación: son resultados exploratorios, no una estimación de retorno económico
real. Ver [evaluación y limitaciones](docs/marketing-results.md).

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
separado. El modelo inicial del código principal es logístico; el candidato
seleccionado se conserva en los artefactos de su investigación.
Ver [configuración y reanudación de AutoResearch](docs/autoresearch.md).

Para ejecutar las pruebas:

```bash
uv run python -m unittest discover -s tests -v
```

## Estado del proyecto

- **Implementado:** procesamiento temporal, baselines, AutoResearch, evaluación
  de costes simulados, tracking local con MLflow e inferencia Docker verificada.
- **AWS preparado:** modelo en S3, imagen en ECR y configuración de IAM y
  Batch Transform. La primera ejecución sigue pendiente de aprobación de cuota.
- **Próximos pasos:** verificar el batch en AWS, registrar versiones de modelos
  en MLflow y automatizar inferencia sobre lotes diarios simulados.

El entrenamiento y la aprobación de modelos siguen siendo manuales. Los modelos,
predicciones y registros locales de MLflow quedan fuera de Git.

## Documentación

- [Arquitectura del sistema](docs/architecture.md)
- [AutoResearch: ejecución, restricciones y recuperación](docs/autoresearch.md)
- [Evaluación, umbrales y MLflow](docs/evaluation.md)
- [Resultados de marketing simulado](docs/marketing-results.md)
- [Inferencia con Docker](inference/README.md)
- [Permisos AWS](infra/iam/README.md) y [Batch Transform](infra/batch/README.md)

La comparación actual evalúa **tasa constante, LightGBM con Optuna y AutoResearch
independiente desde LightGBM sin tuning**. Ver [protocolo y comandos](docs/model-comparison.md).
La corrida anterior que inició desde Optuna se conserva como
[antecedente incremental](docs/autoresearch-optuna-results.md), no como una
comparación independiente de estrategias.
