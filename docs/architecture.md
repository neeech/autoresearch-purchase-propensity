# Arquitectura y estado MLOps

El sistema clasifica sesiones web terminadas. El entrenamiento y la aprobación
son manuales; la inferencia diaria por lotes es el objetivo operativo.

```text
CSV → normalización + historia temporal → particiones Parquet
  → entrenamiento / Optuna / AutoResearch
  → probabilidades → evaluación y umbrales → tracking MLflow
  → revisión del candidato → paquete versionado
  → imagen en ECR + modelo en S3 → SageMaker Batch Transform
  → probabilidades en S3 y logs en CloudWatch
```

## Componentes implementados

| Componente | Responsabilidad | Código |
|---|---|---|
| Procesamiento | Esquema, target, faltantes y split temporal | `src/autoresearch_purchase_propensity/preprocess.py` |
| Historia | Sesiones del mismo usuario de días anteriores | `src/autoresearch_purchase_propensity/history.py` |
| Features y modelo | Transformaciones y estimador | `features.py`, `modeling.py` del mismo paquete |
| Entrenamiento | Ajuste sólo en train; métricas y artefactos | `train.py`, `evaluate.py` |
| Optuna | Búsqueda de hiperparámetros LightGBM | `tune.py` |
| AutoResearch | Propuestas, evaluación, Git y recuperación | `research.py`, [contrato](../program.md) |
| Marketing | Umbrales y penalizaciones simuladas | `marketing.py` |
| Empaquetado | Código, modelo, configuración y hashes | [script](../scripts/package_model.py) |
| Inferencia HTTP | Contrato SageMaker para el pipeline de demostración | [inference](../inference/README.md) |

El paquete fuente completo está en `src/autoresearch_purchase_propensity/`.
[README](../README.md) y [evaluación](evaluation.md) contienen los comandos.

## Datos, tracking y versiones

Train: enero–abril de 2017; validación: mayo; junio fuera de las búsquedas y ya
observado. El target es compra en la sesión; no es respuesta causal a marketing.
Las transformaciones aprendidas se ajustan sólo en train. La historia excluye el
mismo día y el futuro. El [EDA](../notebooks/01_eda.ipynb) justifica estas decisiones.

MLflow local usa SQLite para metadatos y almacenamiento local para artefactos.
No requiere servidor remoto. Optuna registra ensayos; marketing registra
selecciones de umbral. El controlador conserva su propio historial JSONL, sin
registrar automáticamente cada intento en MLflow.

Los resultados agregados están versionados en [independent-results.md](independent-results.md).
[configs/candidate.json](../configs/candidate.json) identifica el candidato por
hash, métricas, umbrales y procedencia. Sus rutas de artefactos son referencias
históricas a archivos generados: no son recursos incluidos en un clon del repositorio.
El código principal conserva baselines reutilizables; no reemplaza automáticamente
sus features por las del candidato. Los pesos y el estado MLflow no están versionados.

## Despliegue y pendientes

La infraestructura de demostración usa S3, ECR, IAM, SageMaker Batch Transform y
CloudWatch. Las configuraciones versionadas son ejemplos ligados a una cuenta;
no crean recursos automáticamente. Ver [IAM](../infra/iam/README.md) y
[batch](../infra/batch/README.md).

El servidor admite el baseline sklearn y el candidato LightGBM. La imagen
local candidate-v2 verificó 64 sesiones por HTTP, con equivalencia de predicciones
a tolerancia 1e-12, incluyendo historia y duración ausente. Se publicó en ECR como candidate-v2 y se creó una definición SageMaker; no se ejecutó un job.
El candidato **no está aprobado ni desplegado**: falta revisar las agregaciones
históricas, ejecutar un job AWS,
verificar salidas/logs y registrar la versión aprobada. La cuota de SageMaker debe consultarse al ejecutar, no asumirse.

Pendientes posteriores: Model Registry, batches simulados con historia coherente,
reporte de datos/probabilidades y programación diaria con reintentos controlados.
No se implementó aún el esquema híbrido de AutoResearch sólo para features y
Optuna para tuning; está definido como [siguiente metodología](model-comparison.md).

## LLM externo e inferencia en AWS

El LLM participa sólo en AutoResearch. El controlador envía código editable,
configuración, métricas, esquema e historial de experimentos; no envía filas del
CSV como parte de ese contexto. Las propuestas se entrenan y evalúan por código
local. El pipeline de inferencia carga el modelo entrenado: no requiere Gemini,
Groq ni una API key LLM para producir probabilidades.

El lugar donde corre el controlador y el proveedor LLM son decisiones separadas:
el controlador podría ejecutarse en AWS y seguir consultando una API externa.
Para este proyecto demostrable el controlador manual local es suficiente. Antes
de usar información confidencial, revisar el contenido enviado y las condiciones
del proveedor. Gemini distingue uso de datos en servicios gratuitos y pagos;
no asumir las mismas garantías de privacidad para ambos.

Referencia: [condiciones Gemini API](https://ai.google.dev/gemini-api/terms).
