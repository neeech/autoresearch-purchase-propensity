# Model Registry local

MLflow Tracking registra evaluaciones y métricas. Model Registry agrega versiones
recuperables del modelo, código de features y dependencias. SQLite guarda metadatos;
los archivos del modelo se guardan como artefactos locales. No crea infraestructura
AWS y no sustituye la aprobación ni las pruebas de despliegue.

## Registro de un paquete confiable

Instalar el extra tracking y partir de un paquete propio generado por
[scripts/package_model.py](../scripts/package_model.py), extraído en un directorio.
El modelo debe coincidir con el SHA de la selección versionada. Las rutas del
siguiente comando son marcadores del paquete generado, no archivos incluidos en Git:

```bash
uv sync --extra tracking
uv run --extra tracking python scripts/register_model.py RUTA_AL_PAQUETE_EXTRAIDO \
  --selection configs/candidate.json
```

El script verifica hashes antes de cargar, conserva el código del paquete y registra
el modelo en purchase-propensity. Reutiliza una versión con el mismo SHA; si cambia
el modelo crea otra versión. Asigna alias candidate, nunca producción. La selección
predeterminada identifica el candidato histórico: no sirve para un modelo distinto.

El registro usa cloudpickle para soportar las funciones de features propias y
LightGBM. Sólo cargar paquetes confiables: la serialización Python puede ejecutar
código. El código original se incluye para no depender de las features editables
del checkout actual. Los artefactos y SQLite no se distribuyen en Git; un clon no
contiene un registro poblado ni descarga modelos históricos.

## Consultar y recuperar

```bash
uv run --extra tracking mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

En http://127.0.0.1:5000 abrir Registered Models / Models, seleccionar
purchase-propensity y la versión deseada. Las evaluaciones siguen en Experiments.

```python
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.sklearn.load_model('models:/purchase-propensity/1')
# model.predict_proba(frame)[:, 1] devuelve la probabilidad positiva.
```

La URI numérica fija la versión; models:/purchase-propensity@candidate resuelve el
alias, que puede cambiar. El modelo pyfunc también expone predict_proba: devuelve
dos columnas, una por clase. La interfaz no reconstruye historia: recibe las
features originales e historia previa con el mismo contrato de entrenamiento.

## Estado verificado

El candidato seleccionado fue registrado localmente como versión 1 con aprobación
pending. Se recuperó por versión con sklearn y por alias con pyfunc; las 64
predicciones coincidieron con la referencia a tolerancia 1e-12 (máxima diferencia
9,28e-17). Repetir el registro reutilizó la misma versión. Identificación y metadatos
en [configs/candidate.json](../configs/candidate.json).

Registrado no significa aprobado o desplegado. Pendiente: verificar Batch Transform
en AWS, revisar semántica de agregaciones y decidir aprobación antes de programar
inferencia diaria. La [rama de inferencia](../inference/README.md) contiene la
adaptación y configuración; el registro local no ejecuta jobs remotos.

Fuente: [Model Registry workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/).

El test temporal de junio de los modelos congelados está documentado en
[test-results.md](test-results.md). La evaluación local mide calidad; la inferencia
AWS verifica ejecución del mismo modelo, sin cambiar splits ni umbrales.
