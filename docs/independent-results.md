# Cierre: AutoResearch independiente frente a Optuna

Evaluación retrospectiva sobre 65.371 sesiones de mayo de 2017, 1.115 compras.
Entrenamiento enero–abril. No se utilizó junio en este cierre. La configuración
inicial de AutoResearch coincide con `configs/lightgbm_initial.json`; los hashes
congelados coinciden con los de Optuna y las predicciones se alinearon por sesión.

## Comparación

| Modelo | Log loss ↓ | Average Precision ↑ | Umbral principal | Contactos/gasto directo | Penalización total ↓ |
|---|---:|---:|---:|---:|---:|
| Tasa constante | 0,087228 | 0,017056 | No contactar | 0 | 11150 |
| LightGBM + Optuna | 0,034070 | 0,537777 | 0,266334 | 1340 | 8274 |
| AutoResearch independiente | **0,032932** | **0,570275** | 0,275096 | 1259 | **7911** |

Escenario principal TP=1, TN=0, FP=5, FN=10. Contacto=1 unidad monetaria
simulada. AutoResearch reduce log loss 3,34 %, penalización 4,39 % y contactos
6,04 % frente a Optuna en esta validación. No es una estimación de retorno real.
TP=681, FP=578, FN=434, TN=63678; precisión 54,09 %, recall 61,08 %.
El umbral de cada modelo fue elegido en mayo y quedó guardado para uso posterior.

| Escenario | Penalización Optuna | Penalización AutoResearch |
|---|---:|---:|
| budget | 5441 | 5417 |
| balanced | 5261 | 5073 |
| opportunities | 8274 | 7911 |

## Qué se ejecutó

- Optuna: 20 ensayos secuenciales, features de sesión fijas, CPU 4 threads,
  semilla 42, sin early stopping ni pruning.
- AutoResearch: Gemini `gemini-3.6-flash`, 10 propuestas completadas, 5 keep y
  5 discard, más entrenamiento inicial. Timeout 180 s por entrenamiento y máximo
  10 requests/minuto. Log loss inicial 0,03919710; mejor experimento 9.
- Mejores cambios: regularización del modelo, agregaciones históricas y ventanas
  de recencia/frecuencia, promedios de actividad, ratios de la sesión y tuning final.
- El ganador sigue siendo LightGBM. Los hiperparámetros efectivos se modifican
  en modeling.py: leer sólo config.json no basta para reconstruirlos. Por eso
  se preservan modelo, configuración **y snapshot completo de código**.

No son presupuestos equivalentes: Optuna tuvo más entrenamientos y AutoResearch
más libertad. No atribuimos la mejora exclusivamente a features ni afirmamos
que AutoResearch sea un optimizador de hiperparámetros superior. No se midió
coste monetario de llamadas API; el tiempo del controlador incluye LLM y entrenamiento.

## Trazabilidad y estado del candidato

- Investigación: `20261114T175721832207Z`; mejor commit
  `1f793b21a471b93f83b03026301b8459daf61c6b`, experimento 9.
- Selección versionada: [configs/candidate.json](../configs/candidate.json), con
  SHA del modelo, hashes de validación, métricas, umbrales y penalizaciones.
- La evaluación incluyó usuarios con y sin historia. Log loss sin historia:
  constante 0,062313; Optuna 0,025760; AutoResearch 0,025314. Con historia:
  constante 0,196287; Optuna 0,070447; AutoResearch 0,066274.
- Se registraron evaluación, procedencia y paquete en MLflow local. Ese estado y
  los pesos no se distribuyen en Git; este documento contiene los resultados
  agregados y candidate.json los metadatos de selección. Para generar otros
  artefactos, ejecutar [el protocolo](model-comparison.md), no asumir que las
  rutas históricas existen en otro clon.
- Estado: **candidato no aprobado, no desplegado**. No se sobrescribió demo-v1.
  La imagen existente necesita adaptar el servidor y sus dependencias para este
  pipeline LightGBM. Empaquetado no equivale a compatibilidad de inferencia verificada.
- Revisar antes de aprobar la semántica de promedios históricos cuando faltan
  valores: el código generado convierte parte de esos faltantes en cero.

Mayo se reutilizó para seleccionar modelos y umbrales; estos resultados pueden
ser optimistas. Junio ya observado no constituye un test ciego. Confirmar el
resultado requiere un período independiente. La política clasifica sesiones
terminadas; no mide compras incrementales causadas por marketing.

La corrida anterior [Optuna + AutoResearch](autoresearch-optuna-results.md) es
un antecedente distinto y no debe mezclarse con esta comparación independiente.
