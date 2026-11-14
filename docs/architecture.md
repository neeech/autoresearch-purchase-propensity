# Arquitectura del pipeline (referencia del proyecto cerrado)

El sistema recibe sesiones web terminadas, construye sus features y produce una
probabilidad de compra por sesión. Entrenamiento y aprobación son manuales; el
objetivo operativo es inferencia por lotes diarios con resultados trazables.

## Desarrollo del modelo

```text
CSV → validación y normalización → historia de días anteriores → Parquet
  → entrenamiento / Optuna / AutoResearch → selección en mayo
  → modelo y umbrales congelados → evaluación temporal en junio
  → registro MLflow → revisión y empaquetado
```

| Código | Responsabilidad |
|---|---|
| preprocess.py / history.py | Datos, target, particiones y pasado disponible |
| features.py / modeling.py | Features y pipeline de estimación |
| train.py / evaluate.py | Ajuste en train y métricas probabilísticas |
| tune.py | Búsqueda de hiperparámetros con Optuna |
| research.py | Propuestas LLM, ejecución, selección y recuperación Git |
| marketing.py | Umbrales y penalizaciones simuladas |
| final_evaluation.py | Test de modelos y umbrales congelados |

Los módulos están en `src/autoresearch_purchase_propensity/`.
El [contrato de datos](data-contract.md) define campos, faltantes e historia.
[Evaluación](evaluation.md) describe enero–abril para train, mayo para validación
y junio para test. Los resultados están en [test-results.md](test-results.md).

Optuna y AutoResearch son procesos de desarrollo, no dependencias de inferencia.
El LLM recibe código, esquema, configuración, métricas e historial; no recibe filas
del dataset en ese contexto. El contenedor de predicción no necesita una API key LLM.

## Registro y artefactos

MLflow usa SQLite local para metadatos y archivos locales para artefactos. Tracking
registra ensayos/evaluaciones; Model Registry conserva versiones recuperables con
su código de features. Ver [guía del registro](model-registry.md).

[configs/candidate.json](../configs/candidate.json) identifica la selección,
hashes, métricas, umbrales y referencias de registro/despliegue. Su contenido es
metadato, no pesos. Las rutas históricas apuntan a salidas generadas: un clon no
incluye modelos entrenados ni el registro poblado. Para reproducir un entrenamiento,
seguir [el protocolo de comparación](model-comparison.md).

[scripts/package_model.py](../scripts/package_model.py) empaqueta el modelo y su
código original. La configuración declarada, los parámetros efectivos del estimador
y el snapshot permiten reconstruir qué se evaluó; no se sustituye el baseline del
código principal automáticamente con un experimento ganador.

## Inferencia operativa

```text
Lote nuevo → validación y preparación con historia disponible → S3
  → SageMaker Batch Transform [imagen ECR + modelo S3]
  → probabilidades S3 → reporte del lote
                      → logs CloudWatch
```

El lote de inferencia no contiene target. Identificadores, fecha del lote, versión
del modelo y orden se conservan fuera del estimador para vincular cada resultado.
La historia sólo utiliza días anteriores; las transformaciones se recuperan del
modelo, sin ajustarlas con el lote nuevo. La normalización debe respetar el
[contrato de entrada](data-contract.md).

El [servidor](../inference/README.md) y Docker se verificaron con 64 sesiones de
validación, incluyendo usuarios con/sin historia y duraciones ausentes. Las
probabilidades coinciden con la referencia a tolerancia 1e-12. La imagen
candidate-v2 y el paquete están publicados; su definición SageMaker está en
[infra/batch](../infra/batch/README.md). Todavía no se ejecutó Batch Transform.

IAM limita accesos a S3, ECR y logs. El job usa recursos temporales; un budget no
es un límite automático de ejecución. Para el primer ensayo se prevé seguimiento
manual y solicitud de parada si supera 15 minutos, confirmando su finalización.

## Trabajo que quedó sin implementar al cierre

1. Validar y procesar un lote nuevo sin etiquetas con el mismo esquema e historia.
2. Guardar probabilidades y un reporte de volumen, faltantes, distribución de scores,
   versión y duración. Con etiquetas posteriores, agregar métricas predictivas.
3. Verificar el primer job AWS y el acceso efectivo a archivos/logs.
4. Aprobar una versión y programar inferencia diaria con control de duplicados,
   fallos y recuperación de la versión anterior.

El candidato actual sigue pendiente de aprobación. Las sesiones nuevas permiten
probar operación; sólo datos etiquetados independientes permiten medir desempeño
predictivo. Simulaciones y penalizaciones de marketing no prueban impacto comercial.

El proyecto no continuará con estos pasos. Ver [cierre y alcance](project-closure.md).
