# Evaluación: validación y test temporal

El objetivo es probabilidad de compra por sesión terminada. [Contrato de datos](data-contract.md).
La métrica principal es log loss (menor es mejor); Average Precision es secundaria.

## Separación de decisiones

| Datos | Uso permitido |
|---|---|
| Enero–abril | Ajustar modelo y transformaciones |
| Mayo | Buscar features/modelos/parámetros y elegir umbrales |
| Junio | Comparar los modelos congelados, sin reajustarlos |

Junio es el test temporal del pipeline actual. Hay exposición histórica confirmada:
el commit Git 43bc26a evaluó modelos anteriores en junio. Eso no convierte junio en
entrenamiento, pero impide presentarlo como test ciego para todo el proyecto.
Ver la evidencia y resultados actuales en [test-results.md](test-results.md).

## Umbrales y campaña simulada

[configs/marketing.json](../configs/marketing.json) fija las penalizaciones:

| Escenario | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| budget | 1 | 0 | 10 | 5 |
| balanced | 1 | 0 | 5 | 5 |
| opportunities (principal) | 1 | 0 | 5 | 10 |

Seleccionar el umbral en mayo con marketing.py. Para p >= threshold se contacta;
ante costes iguales se prefieren menos contactos. La constante usa exclusivamente
la tasa de compra de train y sólo permite seleccionar a todos o a nadie.

Penalización = TP*cTP + TN*cTN + FP*cFP + FN*cFN. Separar gasto directo simulado
(contactos × coste de contacto) de penalización total. Los adicionales por errores
no son pagos observados, y la selección retrospectiva no mide ventas incrementales.

## Ejecutar desde artefactos generados

Entrenar con train.py genera un directorio bajo experiments/training. RUN es un
marcador de ese directorio, no una carpeta que deba existir en un clon:

```bash
uv run python -m autoresearch_purchase_propensity.train
uv run python -m autoresearch_purchase_propensity.marketing \
  experiments/training/RUN/candidate_validation_predictions.csv \
  --model-id RUN --period validation-2017-05 \
  --output experiments/evaluation/RUN-validation --mlflow
```

El CSV contiene session_row_id,target,probability. Para recuperar otro modelo
propio confiable, [export_predictions.py](../scripts/export_predictions.py) carga
su código original y acepta --split validation o --split test. No reentrena.
Joblib sólo debe cargar artefactos confiables.

## Test de los tres modelos

[configs/test_evaluation.json](../configs/test_evaluation.json) congela artefactos,
hashes de datos, penalizaciones y umbrales elegidos en mayo. Se versionó antes de
calcular las métricas actuales de junio. El módulo final_evaluation verifica los
hashes, usa el código original de cada modelo y aplica esos umbrales; nunca fit.

```bash
uv run --extra tracking python -m autoresearch_purchase_propensity.final_evaluation \
  --output experiments/evaluation/MI_TEST --mlflow
```

El comando requiere los artefactos propios identificados en el protocolo, que no
se distribuyen en Git. Para reproducir otra investigación, generar primero sus
modelos y umbrales y escribir un protocolo nuevo antes de evaluar; no sustituir
pesos bajo los hashes históricos. La salida debe ser un directorio nuevo.

Se generan métricas globales, costes por escenario, segmentos con/sin historia,
probabilidades y procedencia. El comando no optimiza con test ni cambia la selección
del candidato. Si se usan los resultados para elegir otra versión o features,
se requiere un período nuevo para una evaluación independiente posterior.

## MLflow y AWS

Instalar --extra tracking para registrar evaluaciones en MLflow local. SQLite
guarda metadatos; los reportes quedan como artefactos. Ver
[Model Registry](model-registry.md) para versionar y recuperar modelos.

La calidad predictiva puede evaluarse localmente: AWS no es necesario para ello.
El ensayo batch de 64 sesiones de mayo comprueba equivalencia de despliegue, no
reemplaza el test. También es posible inferir junio en AWS y evaluar después
contra sus etiquetas, con los mismos modelos y umbrales congelados.
