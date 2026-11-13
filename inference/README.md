# Inferencia de demostración

Servidor WSGI con Gunicorn para el pipeline sklearn de demo-v1. No entrena.
El contrato de SageMaker requiere puerto 8080, `GET /ping` y `POST /invocations`:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-batch-code.html

## Construcción y ejecución local

Desde la raíz del repositorio:

```bash
docker build -t purchase-propensity-inference:demo-v1 inference
mkdir -p experiments/deployment/demo-v1/model
tar -xzf experiments/deployment/demo-v1/model.tar.gz -C experiments/deployment/demo-v1/model
docker run --rm --name purchase-propensity-local -p 127.0.0.1:8080:8080 \
  --mount "type=bind,src=$PWD/experiments/deployment/demo-v1/model,dst=/opt/ml/model,readonly" \
  purchase-propensity-inference:demo-v1 serve
```

En otra terminal:

```bash
curl --fail http://localhost:8080/ping
curl --fail http://localhost:8080/invocations \
  -H 'Content-Type: application/jsonlines' \
  --data-binary @experiments/deployment/demo-v1/sample.jsonl
```

`sample.jsonl` es la muestra local de verificación generada en esta sesión; no está
versionada. Para otras instalaciones preparar filas con el contrato siguiente.
Detener con Ctrl+C o `docker stop purchase-propensity-local`.

## Contrato

Una sesión por línea JSON, con todas las columnas de `data_manifest.json.features`
excepto `month`, y sin columnas extra. `history` es una lista de sesiones previas;
para usuarios nuevos es `[]`. Los faltantes se representan con `null`.
Las columnas numéricas originales se recuperan de la selección numérica ajustada
del pipeline. Esto evita tratarlas como categorías cuando un minilote sólo contiene
faltantes. Este adaptador es específico del pipeline sklearn de demostración;
un candidato con otra estructura requiere revisar el adaptador y dependencias.

La salida contiene una línea `{"probability": 0.01}` por sesión, en el mismo orden.
Los IDs se conservan aparte: no entran al modelo. En Batch Transform se utilizarán
`ContentType=application/jsonlines`, `SplitType=Line`, `AssembleWith=Line`,
`BatchStrategy=MultiRecord`, `MaxConcurrentTransforms=1` y `MaxPayloadInMB=6`.

SageMaker descarga y descomprime el paquete de S3 en `/opt/ml/model`. La imagen
contiene servidor y dependencias; no contiene modelo, datos ni credenciales.
El código de features se carga desde el paquete del modelo. Cargar únicamente
paquetes propios y confiables: joblib carga objetos Python ejecutables.

## Verificación realizada

- 47 tests del proyecto, incluidos tres del servidor: correctos.
- Imagen construida y servidor HTTP ejecutado localmente en un contenedor sin red externa.
- `/ping`: 200 y cuerpo vacío.
- 16 sesiones de mayo (8 con historia): probabilidades iguales al modelo local,
  tolerancia absoluta 1e-12 y orden conservado.
- Esquema inválido: 400; tipo de contenido incorrecto: 415.
- Contenedor temporal detenido; no se publicó en ECR ni se ejecutó un job AWS.

La muestra está en `experiments/deployment/demo-v1/sample.jsonl`. No se utilizó junio.
El candidato conserva sus limitaciones conocidas: early stopping y warning de
una división histórica. No constituye un modelo final aprobado.
