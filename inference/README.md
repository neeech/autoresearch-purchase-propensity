# Servidor de inferencia de demostración

[server.py](server.py) implementa WSGI con Gunicorn y el contrato HTTP de
SageMaker: puerto 8080, GET /ping y POST /invocations. Recibe JSON Lines y
responde una probabilidad por línea, conservando el orden. No entrena modelos.

**Compatibilidad:** el adaptador actual requiere un pipeline sklearn con una
etapa `features` que contenga la selección numérica ajustada. El baseline
logístico de train.py cumple ese contrato. El candidato LightGBM seleccionado
no lo cumple: requiere adaptación antes de usar este servidor.

## Preparar los archivos desde un clon

Desde la raíz, generar datos y un entrenamiento logístico:

```bash
uv sync
uv run python -m autoresearch_purchase_propensity.preprocess
uv run python -m autoresearch_purchase_propensity.train
```

El entrenamiento imprime resultados y genera un directorio bajo
experiments/training. Sustituir RUN por ese directorio; no es un archivo incluido
en Git. Crear un paquete nuevo:

```bash
uv run python scripts/package_model.py experiments/training/RUN \
  experiments/deployment/local/model.tar.gz
mkdir -p experiments/deployment/local/model
tar -xzf experiments/deployment/local/model.tar.gz -C experiments/deployment/local/model
```

Generar una muestra de validación a partir de los datos procesados:

```bash
uv run python - <<'PY'
from pathlib import Path
import pandas as pd
frame = pd.read_parquet('data/processed/validation_X.parquet').drop(columns='month').head(16)
Path('experiments/deployment/local/sample.jsonl').write_text(
    frame.to_json(orient='records', lines=True, double_precision=15))
PY
```

Estos archivos son salidas locales de los comandos, no recursos descargados.
El paquete conserva features/modelo, configuración, métricas y hashes; las
versiones de dependencias corresponden al entorno de empaquetado.

## Construir y probar

```bash
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  --output type=docker,oci-mediatypes=false \
  -t purchase-propensity-inference:local inference
docker run --rm --name purchase-propensity-local -p 127.0.0.1:8080:8080 \
  --mount "type=bind,src=$PWD/experiments/deployment/local/model,dst=/opt/ml/model,readonly" \
  purchase-propensity-inference:local serve
```

En otra terminal:

```bash
curl --fail http://localhost:8080/ping
curl --fail http://localhost:8080/invocations \
  -H 'Content-Type: application/jsonlines' \
  --data-binary @experiments/deployment/local/sample.jsonl
docker stop purchase-propensity-local
```

No son comandos de publicación ni crean recursos AWS. La exportación Docker de
una sola plataforma evita el índice OCI con attestations para la compatibilidad
del despliegue preparado. Revisar imagen/dependencias al cambiar de modelo.

## Contrato y verificación

Cada entrada contiene las columnas del manifiesto generado por preprocess,
excepto month, sin target ni IDs. history es una lista (vacía para usuarios nuevos);
los faltantes son null. La salida es `{"probability": 0.01}` por fila. Los IDs se
conservan fuera del modelo para unir resultados por orden. El servidor valida
esquema y tipo de contenido; no reconstruye historia a partir de datos crudos.

La demostración anterior se verificó localmente con 16 sesiones y coincidencia
con predicción directa (tolerancia absoluta 1e-12). Eso no acredita compatibilidad
del nuevo candidato ni una ejecución AWS. Repetir la prueba con cada versión.
Los tests del servidor están en [tests/test_inference.py](../tests/test_inference.py).

Cargar únicamente paquetes confiables: joblib puede ejecutar código Python.
La imagen no contiene modelo, datos ni credenciales. En AWS SageMaker descarga y
descomprime el modelo en /opt/ml/model. Ver [configuración batch](../infra/batch/README.md).
