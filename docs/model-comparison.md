# Comparación independiente: Optuna y AutoResearch

## Objetivo

Comparar tres soluciones con los mismos datos y evaluación temporal:

1. **Constante:** tasa de compra de enero–abril.
2. **LightGBM + Optuna:** búsqueda de hiperparámetros con features de sesión fijas.
3. **AutoResearch independiente:** inicia con `configs/lightgbm_initial.json`,
   sin parámetros ganadores ni historial de Optuna. Puede modificar features,
   historia, modelos e hiperparámetros dentro de los permisos existentes.

Es una comparación de estrategias completas, no sólo de optimizadores: el agente
puede hacer ingeniería de features y cambiar de estimador; Optuna usa el espacio
LightGBM predefinido. El punto inicial del agente usa valores convencionales
explícitos (100 árboles, learning_rate 0.1, 31 hojas), no un resultado de tuning.
El modelo inicial sólo sirve como punto de partida; no es una cuarta búsqueda.

## Protocolo fijo

- Train: enero–abril de 2017. Validation: mayo. Junio fuera de ambas búsquedas.
- Historia: únicamente días estrictamente anteriores a cada sesión; nunca futuro.
- Log loss como objetivo; Average Precision como métrica complementaria.
- Semilla 42, CPU local, LightGBM con 4 threads. Sin early stopping ni pruning.
- Optuna: 20 ensayos entrenados; AutoResearch: hasta 20 propuestas, más medición
  inicial. Registrar propuestas fallidas, entrenamientos efectivos y tiempos;
  iguales cantidades solicitadas no implican igual cómputo.
- Misma política de faltantes: LightGBM inicial trata NaN y categorías nativamente.
- El agente recibe configuración activa, código editable y sólo el historial de
  su nueva investigación. No recibe resultados de otras investigaciones.

Mayo se reutiliza para tuning y selección de umbrales: resultados exploratorios,
potencialmente optimistas. Junio ya observado no es un test completamente ciego;
confirmar generalización requerirá un período independiente. No modificar splits.

## Ejecutar

```bash
uv sync --extra tuning --extra tracking
uv run --extra tuning --extra tracking python -m autoresearch_purchase_propensity.tune \
  --trials 20 --output experiments/optuna/MI_ESTUDIO --mlflow
```

La referencia Optuna existente `marketing-baseline-v1` completó 20 ensayos y obtuvo
log loss 0,03407013. Conserva artefactos, datos congelados, parámetros y tiempos.
No hace falta repetirla si los hashes de datos y la preparación siguen iguales.

Con GEMINI_API_KEY configurada y checkout limpio, iniciar **una investigación nueva**:

```bash
uv run autoresearch --provider gemini --model gemini-3.6-flash \
  --config configs/lightgbm_initial.json \
  --experiments 20 --timeout 180 --rpm 10
```

No usar `--resume` ni `best_config.json`. El identificador y la rama se generan
automáticamente. Primero se mide el modelo inicial; después se conserva sólo una
mejora estricta. Las cuotas y disponibilidad del modelo dependen del proveedor;
revisar el error si se detiene. Una clave nueva puede compartir cuota con otras.

## Estado de las investigaciones

| Investigación | Origen | Resultado | Uso |
|---|---|---|---|
| Optuna marketing-baseline-v1 | Espacio LightGBM definido | 0,03407013 | Referencia independiente |
| AutoResearch 20261114T172825124860Z | Ganador de Optuna | 0,03321754 | Antecedente incremental; no comparación independiente |
| Nuevo AutoResearch | lightgbm_initial.json | Pendiente | Comparación independiente |

La corrida anterior no se altera. Ver [resultados incrementales](autoresearch-optuna-results.md).
El mejor código permanece en su workspace, no se copia al baseline inicial.

## Evaluación y MLOps

Recuperar cada modelo con su snapshot original y comparar predicciones sobre las
mismas filas. Elegir umbral por modelo en mayo con los escenarios de marketing;
congelarlo antes de evaluar otro período. La constante sólo permite contactar a
todos o a nadie. Registrar métricas, umbrales, hashes y reportes en MLflow local.

Separar contacts, direct_spend (1 unidad simulada por contacto, configurable) y
total_penalty. FP/FN incluyen penalizaciones hipotéticas; no presentar su reducción
como dinero ahorrado ni efecto causal de marketing. La clasificación es retrospectiva
sobre sesiones terminadas, no una predicción de compras futuras por una campaña.

Después de revisión y aprobación: empaquetar una nueva versión, adaptar y probar
la imagen de inferencia antes de subirla a ECR/S3. No sobrescribir demo-v1: su
servidor asume el pipeline anterior. AWS y el Model Registry siguen pendientes.
