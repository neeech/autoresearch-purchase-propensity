# Purchase Propensity · AutoResearch y despliegue batch

Proyecto demostrable de Machine Learning para estimar la probabilidad de compra
asociada a una sesión web terminada. Combina un pipeline temporal de datos,
baselines, experimentación asistida por LLM, evaluación con penalizaciones de
marketing simuladas y preparación de inferencia batch en AWS.

**Estado:** entrenamiento, AutoResearch, evaluación, MLflow local e inferencia
Docker funcionan. El primer Batch Transform en AWS sigue pendiente de cuota.
El candidato publicado para la demo todavía no es un modelo aprobado para producción.

## Problema y datos

Cada fila representa una sesión: visitas, acciones, páginas vistas, duración,
fuente de tráfico, dispositivo y geografía. `fullVisitorId` identifica al usuario.
El target es `transactions > 0`; se asume que las transacciones faltantes son cero.
Los datos originales se esperan en `data/raw/data.csv`.

La mayoría de usuarios son nuevos cada mes. Por eso se consideran tanto las
variables de la sesión como su historia disponible, cuando existe. La historia
contiene sólo sesiones del mismo usuario de **días estrictamente anteriores**;
excluye la sesión actual y otras del mismo día. Se construye cronológicamente
antes de separar los períodos, sin usar el futuro.

| Partición | Período de 2017 | Uso |
|---|---|---|
| Entrenamiento | Enero–abril | Ajustar modelo y transformaciones |
| Validación | Mayo | Seleccionar modelos y umbrales |
| Evaluación posterior | Junio | Fuera de AutoResearch; ya observado, no es un test ciego |

No se usan transacciones, ingresos ni identificadores actuales como features.
Las variables de comportamiento sólo están completas al terminar la sesión:
**este proyecto no predice antes de la compra ni mide el efecto de una campaña**.

## Workflow

```text
CSV de sesiones
  → normalización + target + historia temporal
  → particiones Parquet
  → baselines: probabilidad constante y regresión logística
  → AutoResearch: proponer → entrenar → evaluar → conservar/revertir
  → candidato congelado
  → probabilidades + umbrales por costes simulados + comparación de baselines
  → MLflow local: registro de evaluaciones
  → revisión humana → empaquetado del modelo
  → S3 (modelo) + ECR (imagen Docker) → SageMaker Batch Transform
  → probabilidades en S3 + logs en CloudWatch
```

El último tramo está preparado, pero aún no se ejecutó en AWS. La generación de
lotes simulados diarios y su programación son trabajo pendiente.

## Ejecutar localmente

Requisitos: Python >= 3.13, [uv](https://docs.astral.sh/uv/) y Git configurado.
Docker y acceso AWS sólo son necesarios para las prácticas de despliegue.

```bash
uv sync
uv run python -m autoresearch_purchase_propensity.preprocess
uv run python -m autoresearch_purchase_propensity.train
uv run python -m unittest discover -s tests -v
```

El entrenamiento guarda modelo, código, configuración, métricas y probabilidades
de validación en `experiments/training/<run>/`. No entrena sobre junio.

### AutoResearch

Con un checkout limpio y `GROQ_API_KEY` configurada en el entorno:

```bash
uv run autoresearch --experiments 10 --timeout 180 --rpm 10
```

El LLM propone una hipótesis y cambios estructurados en `features.py` y
`modeling.py`. El controlador entrena desde cero y conserva sólo mejoras estrictas
**de log loss en mayo**. No optimiza F1 ni el coste de marketing. El código base
actual sigue siendo logístico: el candidato ganador vive en su snapshot de investigación.

Cada investigación usa rama/worktree propios, checkpoints Git, datos congelados,
logs e historial JSONL. Se puede continuar con `--resume RUN_ID`. Datos, target,
splits, métricas y controlador están protegidos frente a propuestas. Estos
controles no equivalen a una sandbox de seguridad para código Python hostil;
usar propuestas confiables y no ejecutar código generado con credenciales AWS.

Sin API: `uv run autoresearch --experiments 0`. También se admiten propuestas
guardadas y Gemini. Ver [opciones, límites y recuperación](docs/autoresearch.md)
y [contrato del agente](program.md). No guardar claves en archivos versionados.

### Umbrales y costes simulados

Después de congelar el modelo, se elige el umbral de menor penalización en
validación. Los costes están en [configs/marketing.json](configs/marketing.json):

| Escenario | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| Cuidar presupuesto (`budget`) | 1 | 0 | 10 | 5 |
| Equilibrado (`balanced`) | 1 | 0 | 5 | 5 |
| Principal: oportunidades (`opportunities`) | 1 | 0 | 5 | 10 |

Coste medio = `(TP*cTP + TN*cTN + FP*cFP + FN*cFN) / N`.
Son penalizaciones hipotéticas: FP incluye desperdicio de campaña y FN una
preferencia por no dejar compradores fuera. No representan dólares ni ventas
perdidas demostradas. `budget` no se refiere al presupuesto AWS.

```bash
# Sustituir RUN por el directorio generado por train.
uv run python -m autoresearch_purchase_propensity.marketing \
  experiments/training/RUN/candidate_validation_predictions.csv \
  --model-id RUN --period validation-2017-05 \
  --output experiments/evaluation/RUN-validation
```

Se compara contra contactar a todos y a nadie. Repetir con el modelo baseline,
elegir su propio umbral y comparar sobre las mismas filas. `--thresholds` aplica
umbrales congelados a otro período sin reajustarlos. Ver [protocolo](docs/evaluation.md).

### MLflow

```bash
uv sync --extra tracking
# Añadir --mlflow al comando de evaluación para registrar sus resultados.
uv run --extra tracking mlflow ui \
  --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

Abrir http://127.0.0.1:5000 → `purchase-propensity-marketing` → **Runs**.
SQLite conserva metadatos; los artefactos contienen reportes y umbrales.
El registro es opcional y local: no reentrena ni despliega modelos. Model Registry
y la aprobación/versionado de modelos en MLflow todavía están pendientes.

## Resultados de validación

Comparación sobre 65.371 sesiones de mayo, 1.115 con compra. Cada modelo eligió
su umbral en ese mismo período; estos costes pueden ser optimistas.

| Modelo | Log loss ↓ | Umbral principal | Penalización media ↓ |
|---|---:|---:|---:|
| Probabilidad constante | 0,087228 | No contactar | 0,170565 |
| Regresión logística | 0,038736 | 0,184553 | 0,133469 |
| HistGradientBoosting seleccionado | **0,033616** | **0,238546** | **0,123816** |

El candidato reduce la penalización un 7,23 % frente al baseline logístico en
este escenario. No es evidencia de retorno económico real ni de generalización
fuera de mayo. Ver [resultados, trazabilidad y limitaciones](docs/marketing-results.md).
El candidato conserva early stopping y un warning conocido de división en historia
vacía; ambos deben revisarse antes de aprobar una versión final.

## AWS e inferencia

| Herramienta | Función y estado |
|---|---|
| S3 | Modelo empaquetado y lote de prueba almacenados; destino de predicciones |
| Docker / ECR | Servidor reproducible, imagen probada localmente y publicada |
| IAM | Rol de ejecución con política acotada preparada |
| SageMaker Batch Transform | Definición/configuración preparadas; primer job pendiente de cuota |
| CloudWatch | Logs previstos; falta comprobarlos con un job real |
| AWS Budgets | Budget creado; alertas/configuración final pendientes de verificar |
| Orquestación diaria | Pendiente; sólo se automatizará inferencia |

```bash
uv run python scripts/package_model.py RUTA_AL_ARTEFACTO \
  experiments/deployment/MI_VERSION/model.tar.gz
```

Empaquetar no aprueba ni sube el modelo. La imagen recibe sesiones procesadas con
historia en JSON Lines y devuelve probabilidades; el umbral se aplica después.
Ver [servidor y Docker](inference/README.md), [permisos IAM](infra/iam/README.md),
[primer batch](infra/batch/README.md) y [arquitectura](docs/architecture.md).
Los ejemplos de infraestructura están ligados a la cuenta/región de esta demo;
revisar identificadores, permisos, cuotas y costes antes de reutilizarlos.

## Estructura

```text
configs/                 Datos, entrenamiento y penalizaciones
src/autoresearch_purchase_propensity/
  preprocess.py          Limpieza, target y splits
  history.py             Disponibilidad temporal del pasado
  features.py            Features editables por AutoResearch
  modeling.py            Modelo y preprocesamiento editables
  train.py / evaluate.py  Entrenamiento y métricas protegidos
  research.py            Loop, proveedores LLM, checkpoints y memoria
  marketing.py           Umbrales, comparaciones y tracking opcional
scripts/                 Exportación de predicciones y empaquetado
inference/               Servidor, dependencias y Dockerfile
infra/                   Políticas IAM y configuración batch
tests/                  Pruebas de datos, loop, métricas e inferencia
docs/                   Arquitectura, protocolos y resultados
```

Los archivos Parquet procesados, investigaciones, predicciones, modelos, SQLite,
artefactos MLflow y materiales personales de aprendizaje quedan fuera de Git.
El manifiesto de datos existente permanece versionado como metadato. Las tareas
se gestionan en GitHub Issues. El entrenamiento y AutoResearch siguen manuales;
no hay reentrenamiento ni aprobación automática en producción.
