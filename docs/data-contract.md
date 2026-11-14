# Contrato de datos

Una fila representa una sesión web terminada. El dataset original versionado es
[data/raw/data.csv](../data/raw/data.csv). El [EDA](../notebooks/01_eda.ipynb)
describe su calidad; el código fuente de las reglas es
[preprocess.py](../src/autoresearch_purchase_propensity/preprocess.py).

## Campos y faltantes

| Campos | Tipo lógico y tratamiento |
|---|---|
| fullVisitorID | Texto no vacío; sólo para historia, nunca predictor |
| date | Fecha YYYYMMDD válida; orden temporal, no predictor absoluto |
| visitNumber, hits | Contadores numéricos no negativos, obligatorios |
| bounces | 0/1; null significa no rebote y se normaliza a 0 |
| pageviews, timeOnSite | Numéricos no negativos; ausencia se conserva como NaN |
| transactions | Define target: >0 significa compra; null→0 es una hipótesis explícita |
| transactionRevenue | Resultado de la sesión, excluido de predictores |
| source, channelGrouping, browser, deviceCategory, country, city | Categóricas; marcadores de ausencia se normalizan a null |

No deduplicar sólo por usuario/día: puede haber sesiones legítimas múltiples.
El procesador genera session_row_id para alinear features, etiquetas y salidas.
No imputar IDs ni fechas. Entradas inválidas de campos obligatorios se rechazan.

LightGBM inicial trata NaN y categorías nativamente. La regresión logística usa
mediana e indicadores para numéricas y categoría de ausencia + one-hot para
categóricas. Toda transformación aprendida se ajusta sólo en entrenamiento y
se reutiliza al predecir; nunca se recalcula con el lote de inferencia.
El candidato puede tener transformaciones propias: conservar su snapshot y revisar
su semántica antes de aprobarlo, especialmente promedios históricos con faltantes.

## Tiempo y particiones

[configs/data.json](../configs/data.json) define enero–abril de 2017 para train,
mayo para validación y junio para test. Optuna/AutoResearch y selección de umbrales
usan mayo; junio evalúa modelos y decisiones congelados, sin fit ni tuning.

[history.py](../src/autoresearch_purchase_propensity/history.py) entrega sesiones
del mismo usuario de días estrictamente anteriores. Puede incluir días previos de
mayo o junio, si esa información ya estaba disponible. Excluye el mismo día y el
futuro; las transformaciones por fila no aprenden parámetros de test.

## Salidas y consumo

Preprocess genera Parquet X/y/metadata por partición y manifiesto con esquema,
resumen y hash de origen. Son salidas generadas, no pesos ni estado de un servicio.
El entrenamiento y la exportación verifican alineación de IDs; el protocolo de
test verifica además hashes del conjunto y de los artefactos.

La inferencia recibe las columnas originales permitidas y history, sin month,
IDs, transacciones ni target. Los null JSON representan faltantes y history debe
ser una lista. Ver [contrato HTTP](../inference/README.md). Los IDs se conservan
fuera del modelo y se unen por el orden verificado de las predicciones.

La ubicación local o AWS no determina si los datos son nuevos para el modelo.
No enviar etiquetas al contenedor de inferencia; unirlas después para calcular
métricas. El resultado actual es retrospectivo: no prueba causalidad de marketing.
