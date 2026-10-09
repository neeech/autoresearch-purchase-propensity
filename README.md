# Autoresearch: propensión de compra por sesión

Benchmark offline para clasificar sesiones terminadas. El target es `transactions > 0`; se asume que transacciones vacías equivalen a cero. Los totales de actividad pueden incluir efectos de una compra y no validan una predicción anticipada.

## Reproducir

```bash
uv sync
uv run python -m autoresearch_purchase_propensity.preprocess --config configs/data.json
uv run python -m unittest discover -s tests -v
```

Los paths de configuración son relativos al directorio desde el que se ejecuta; ejecutar desde la raíz del proyecto. Si las dependencias ya están instaladas:

```bash
PYTHONPATH=src python -m autoresearch_purchase_propensity.preprocess
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Estructura

- `data/raw/`: fuente original, no modificada.
- `data/processed/`: nueve archivos Parquet y `manifest.json` con hash de fuente, features, exclusiones y resumen del split.
- `configs/data.json`: paths, fronteras temporales y supuesto del target.
- `src/autoresearch_purchase_propensity/eda.py`: EDA reproducible, tablas y gráficos PNG/PDF.
- `src/autoresearch_purchase_propensity/preprocess.py`: lectura validada, normalización, exportación y split.
- `src/autoresearch_purchase_propensity/features.py`: historia por usuario y variables de calendario.
- `src/autoresearch_purchase_propensity/eda_detail.py`: auditoría profunda reproducible de datos originales y normalizados.
- `reports/eda.md` y `reports/eda.html`: reporte exploratorio, diccionario y decisiones.
- `tests/`: comprobaciones de historia temporal y particiones.
- `experiments/`: reservado para los futuros candidatos de autoresearch.

## Contrato de datos

Para cada partición (`train`, `validation`, `test`), `*_X.parquet` contiene sólo features. `*_y.parquet` contiene `session_row_id` y `target`; `*_metadata.parquet` contiene ID de fila, usuario, fecha e indicador de historia. Los tres archivos tienen exactamente el mismo orden de filas; no reordenar uno independientemente. El ID de fila es la posición original en el CSV, no un identificador de sesión de la fuente.

Las features excluyen `transactions`, `transactionRevenue`, `fullVisitorID`, `date`, `session_row_id` y `target`. Los IDs se leen como texto para preservar ceros iniciales. Según la definición proporcionada, `bounces` vacío significa sin rebote y se codifica como 0; rebote como 1. No se genera `bounces_missing`. Los otros faltantes numéricos se conservan; se agregan indicadores para duración (en segundos) y pageviews. Las categorías sin información se normalizan a faltantes. No se ajustan imputadores, escaladores ni codificadores: eso corresponde al pipeline entrenado sólo con train.

Historia: sesiones previas, sesiones previas con compra, tasa de compra, medias de hits/pageviews/duración y días desde última sesión/compra. Se consideran exclusivamente días anteriores: sesiones del mismo día comparten historia. Los promedios usan sólo valores observados. Usuarios sin historia reciben contadores cero, recencias/promedios faltantes y `has_history=0`. La historia se actualiza secuencialmente dentro de validación/test usando días ya terminados; el modelo no se reentrena con esos datos.

Split: enero–abril de 2017 para train, mayo para validation y junio para test. La definición de usuario nuevo es relativa al período observado. El EDA incluye estadísticas descriptivas del test; sus resultados de modelos deben permanecer reservados para la evaluación final.

Próximo paso: predictor constante y regresión logística, comparando features con/sin historia; log loss, Average Precision, Brier score y calibración, globales y por presencia de historia.

## Ejecutar sólo el análisis exploratorio

```bash
uv run python -m autoresearch_purchase_propensity.eda_detail --config configs/data.json
```

No modifica el CSV ni los Parquet. Compara faltantes originales y normalizados, verifica claves y consistencia, y calcula relaciones con compra sólo sobre train. Las estadísticas de cobertura y drift incluyen todo el período. El reporte HTML referencia figuras locales: compartir junto con `reports/figures/`.

## Baselines simples

```bash
uv run python -m autoresearch_purchase_propensity.train
```

Compara probabilidad constante y regresión logística con contexto, actividad e historia. Entrena sólo con train y evalúa mayo; no lee test. Reporta log loss y Average Precision, sin calibración. La imputación y codificación se ajustan sólo con train; se excluye month.

Resultados: `reports/baselines.md` y `reports/baselines.html`. Cada ejecución conserva código, configuración, manifiesto de datos y modelos en `experiments/baselines/<run_id>/`. `latest.json` indica la ejecución actual. Los experimentos anteriores se conservan como historial.

## Autoresearch LightGBM

El agente de esta conversación propone configuraciones en `configs/lightgbm.json`. El runner ejecuta un candidato; no consulta un LLM ni genera candidatos automáticamente.

```bash
uv run python -m autoresearch_purchase_propensity.research --prepare
# Commit local del protocolo, config y snapshot antes de entrenar.
uv run python -m autoresearch_purchase_propensity.research --run
# Commit local de resultados y estado después de entrenar.
```

Estado/presupuesto: `research-state.json`. Historial: `experiments/lightgbm/history.csv`. Decisiones: `research-log.md`; síntesis: `findings.md`. Cada candidato guarda código, configuración, versiones, hashes de datos y modelo nativo de LightGBM. No se lee test ni se usa early stopping; presupuesto máximo de 20 ejecuciones o 60 minutos desde el primer protocolo. No eliminar el estado para continuar: repetir prepare/run hasta el límite. Al finalizar, el mejor modelo está indicado por `research-state.json`.

## Cierre y referencia para otra tool

Resultados y metodología: [reports/autoresearch.md](reports/autoresearch.md). Se congelaron los modelos antes de evaluar junio, sin reentrenar; métricas finales en `experiments/final_test/metrics.csv`. El protocolo congelado identifica los artefactos mediante hashes. La evaluación final rechaza otra ejecución si ya existe su carpeta de resultados.

```bash
# Estas operaciones ya se ejecutaron; se muestran como referencia del protocolo.
uv run python -m autoresearch_purchase_propensity.final_evaluation --freeze
uv run python -m autoresearch_purchase_propensity.final_evaluation
```

Este baseline se conserva en `autoresearchToolTest`. Para la futura tool, mantener igual protocolo y presupuesto y congelar su candidato antes de comparar con junio. Junio ya no es un período desconocido: no usar sus métricas para ajustar la búsqueda.
