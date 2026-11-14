# Evaluación de propensión y simulación de marketing

AutoResearch sigue minimizando log loss en validación. El umbral se selecciona
**después** de congelar el modelo; no cambia entrenamiento ni inferencia AWS.

Los escenarios de `configs/marketing.json` usan unidades de penalización, no
importes económicos medidos. Principal: `opportunities` (TP=1, TN=0, FP=5, FN=10).
FP incluye contacto y desperdicio; FN representa una oportunidad no seleccionada,
no una venta perdida demostrada. Esta clasificación retrospectiva de sesiones
terminadas no estima el efecto de descuentos ni predice compras futuras.

## Flujo

1. Entrenar con el pipeline existente. Cada entrenamiento nuevo exporta las
   predicciones de validación del candidato y la referencia constante (tasa de
   compra de entrenamiento). No se reutilizan probabilidades de otro modelo.
2. Seleccionar umbrales sólo en validación. La búsqueda considera todas las
   decisiones distintas, agrupa probabilidades empatadas y, ante costes iguales,
   prefiere menos contactos. `probability >= threshold` define positivo; un umbral
   apenas mayor que 1 representa no contactar a nadie.
3. Comparar coste medio, contactos, precisión y recall con contactar a todos y
   a nadie. Ejecutar también el comando sobre las predicciones de un baseline
   sencillo, con el mismo período y costes. Cada modelo elige su propio umbral
   en validación para una comparación justa.
4. Congelar los umbrales y aplicarlos a otro período etiquetado. No buscar de
   nuevo en ese período. Mayo ya se utiliza para seleccionar modelos: sus costes
   optimizados son descriptivos y pueden ser optimistas. Junio ya observado no
   debe presentarse como test completamente ciego. Un futuro período intacto
   dará una evaluación más sólida. No cambiamos los splits existentes.

```bash
uv run python -m autoresearch_purchase_propensity.train
# Sustituir RUN por el directorio de entrenamiento generado.
uv run python -m autoresearch_purchase_propensity.marketing \
  experiments/training/RUN/candidate_validation_predictions.csv \
  --model-id RUN --period validation-2017-05 \
  --output experiments/evaluation/RUN-validation
```

CSV de entrada: `session_row_id,target,probability`, una fila por sesión. Conservar
identificadores al generar predicciones de otro período para poder unir etiquetas
sin confundir el orden. `--thresholds ruta/thresholds.json` aplica umbrales ya
seleccionados; el comando nunca carga ni entrena un modelo. El usuario debe
proveer un período independiente, con las mismas definiciones y el mismo modelo.

## MLflow local y opcional

```bash
uv sync --extra tracking
# Añadir --mlflow al comando de evaluación anterior.
uv run --extra tracking mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

SQLite guarda metadatos; MLflow guarda reportes como artefactos locales. Registra
identificador del modelo, período, hash del CSV, métricas, costes y umbrales.
Esto es **tracking de evaluaciones**, todavía no un Model Registry ni aprobación
de despliegue. No sube datos a AWS ni registra automáticamente cada experimento.
Los CSV con etiquetas no se copian a MLflow; se registra su hash. Reportes y estado
local quedan fuera de Git. No borrar SQLite y artefactos si se desea conservar el
historial.

Fuentes: [Tracking](https://mlflow.org/docs/latest/ml/tracking/quickstart/) y
[SQLite/backend](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/).

## Recuperar predicciones de entrenamientos anteriores

Para artefactos confiables que aún no exportaban CSV, ejecutar en un proceso nuevo:

```bash
uv run python scripts/export_validation.py RUTA_AL_ARTEFACTO \
  --output experiments/evaluation/MI_VERSION/predictions
```

Carga el modelo con su código original y comprueba el manifest contra los datos
actuales; no reentrena. Escribe candidato, constante y hashes de procedencia.
Sólo debe utilizarse con artefactos propios confiables: joblib carga código Python.
La primera comparación real está en [resultados de mayo](marketing-results.md).
