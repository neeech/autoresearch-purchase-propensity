# Comparación de tres modelos

1. **Constante:** tasa de compra calculada sólo en enero–abril.
2. **LightGBM + Optuna:** 20 ensayos secuenciales, CPU local (4 threads), semilla
   42, log loss de mayo, sin early stopping ni pruning. Duración NaN y categorías
   son tratadas nativamente; no se eliminan rebotes ni se imputa duración cero.
3. **AutoResearch:** parte de la configuración ganadora de Optuna y puede mejorar
   features, historia y modelos. No hay un cuarto modelo LightGBM fijo.

La regresión logística y resultados antiguos se conservan como antecedentes,
no como integrantes de esta nueva comparación. El candidato antiguo de
AutoResearch no demuestra mejora sobre la nueva referencia de Optuna.

## Ejecutar

```bash
uv sync --extra tuning --extra tracking
uv run --extra tuning --extra tracking python -m autoresearch_purchase_propensity.tune \
  --trials 20 --output experiments/optuna/MI_ESTUDIO --mlflow
```

Se reutilizan train.py, load_data, evaluate.metrics y los artefactos de
entrenamiento. El estudio congela exclusivamente train/validation y manifiesto,
guarda Optuna SQLite, resultados por ensayo, código, modelos y best_config.json.
MLflow registra parámetros/métricas/tiempos de cada ensayo, con referencias a los
artefactos locales; no registra automáticamente cada modelo en Model Registry.

Con checkout limpio, para iniciar una investigación nueva desde el ganador:

```bash
uv run autoresearch --config experiments/optuna/MI_ESTUDIO/best_config.json \
  --experiments 20 --timeout 180 --rpm 10
```

La configuración inicial se congela dentro del protocolo; al reanudar se usa ese
protocolo, no se cambia el modelo inicial. Se reentrena desde cero. En LightGBM
la función activa es build_native_features, editable por el agente. Los archivos
originales de datos, los splits y el evaluador no se modifican. No hace falta AWS.
No se dispara el LLM automáticamente al terminar Optuna.

## Decisiones de campaña

Para cada modelo, usar marketing.py sobre sus probabilidades de mayo; cada uno
elige sus umbrales con las mismas penalizaciones. El baseline constante sólo
puede seleccionar a todos o a nadie. Añadir --mlflow para registrar la evaluación.

El reporte distingue contacts, direct_spend y total_penalty. Con --contact-cost 1
cada contacto representa una unidad monetaria simulada; los adicionales de FP/FN
son penalizaciones hipotéticas, no desembolsos observados. Por tanto, el menor
total_penalty no significa necesariamente menor direct_spend. Se registran ambos.

Mayo se reutiliza para tuning/modelos y umbrales: sus resultados son exploratorios.
Ninguna transformación se ajusta en mayo; ninguna fila ve historia futura ni del
mismo día. Junio queda fuera de estas decisiones y ya no es un test ciego. Para
confirmar generalización hará falta un período independiente. Se comparan tanto
número de ensayos como tiempos; AutoResearch modifica features además de parámetros,
por lo que esto mide mejora incremental, no igualdad entre optimizadores.

## Paso a AWS

Primero congelar modelo, código, configuración y umbrales; revisar y aprobar.
Después empaquetar una versión nueva. La imagen demo-v1 no se sobrescribe ni se
considera compatible automáticamente con el nuevo pipeline: su adaptador actual
asume el pipeline anterior. Revisar dependencias y contrato, probar localmente y
recién publicar en ECR/S3. Orquestación diaria y registro de modelos siguen pendientes.

## Primera referencia medida (14/11/2026)

20 ensayos completados y registrados en MLflow. Ganador: ensayo 13,
log loss 0.03407013; constante: 0.08722830.
Sólo train enero–abril y validación mayo, sin early stopping.

| Modelo | Umbral principal | Contactos / gasto directo simulado | Penalización total |
|---|---:|---:|---:|
| Constante | No contactar | 0 | 11150 |
| LightGBM + Optuna | 0.266334 | 1340 | 8274 |
| AutoResearch desde Optuna | Pendiente | Pendiente | Pendiente |

La penalización cae 25.79 % respecto a la constante, pero el gasto directo
crece de 0 a 1340 unidades. No son ahorros monetarios observados. Precisión
51.04%, recall 61.35%, 684 TP, 656 FP y 431 FN.
Los umbrales fueron seleccionados en mayo; no son resultados de test independiente.

Resultados locales: `experiments/optuna/marketing-baseline-v1/` y
`experiments/evaluation/optuna-v1/`. Se conserva SQLite, modelos, snapshots, hashes
y predicciones fuera de Git. La nueva búsqueda del agente no se ejecutó porque
GROQ_API_KEY no está disponible en el proceso del asistente. Desde una terminal
con esa variable configurada y el checkout limpio:

```bash
uv run autoresearch \
  --config experiments/optuna/marketing-baseline-v1/best_config.json \
  --experiments 20 --timeout 180 --rpm 10
```

Este comando requiere conservar el estudio local. En otra máquina, repetir primero
el estudio para generar la configuración y los datos congelados. No confundir
el candidato histórico con el resultado todavía pendiente de esta investigación.
