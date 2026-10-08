# Autoresearch LightGBM

20 entrenamientos: 19 configuraciones y una repetición del ganador. CPU local, features fijas, sin early stopping. Entrenamiento enero–abril; selección en mayo; junio reservado. Cada hipótesis se registró en Git antes de entrenar; todos los candidatos se conservaron.

| Modelo | Log loss ↓ | Brier Score ↓ |
|---|---:|---:|
| Regresión logística | 0,03757169 | 0,01132709 |
| LightGBM seleccionado | **0,03350781** | **0,01037929** |

Mejora relativa de log loss: **10,82%** en validación.

## Configuración seleccionada

```json
{
  "n_estimators": 300,
  "learning_rate": 0.025,
  "num_leaves": 47,
  "min_child_samples": 30,
  "reg_lambda": 1,
  "subsample": 0.6,
  "subsample_freq": 1
}
```

Semilla 42, cuatro hilos, objective=binary, deterministic=true y force_col_wise=true. Sin pesos de clase. Se mantienen las features actuales, excluyendo month; categorías de entrenamiento y faltantes nativos de LightGBM. No se reutilizan el escalado ni el one-hot de la logística.

## Qué aprendimos

- Menor learning rate y muestreo de filas mejoraron el objetivo.
- Más árboles no siempre ayudaron; 450 empeoró la región ganadora.
- 31 y 47 hojas quedaron prácticamente empatadas: la elección refleja el menor valor numérico, no superioridad estadística demostrada.
- Repetir el ganador dio exactamente las mismas probabilidades. El modelo guardado también se verificó al recargarlo.

## Archivos

- [Historial de los 20 entrenamientos](../experiments/lightgbm/history.csv).
- [Configuración del ganador](../experiments/lightgbm/run_017/config.json).
- [Modelo nativo](../experiments/lightgbm/run_017/model.txt).
- [Resumen numérico](../experiments/lightgbm/summary.json).
- [Estado y mejor candidato](../research-state.json).

La selección usa una única validación temporal; el resultado no demuestra generalización a junio. El presupuesto de 20 entrenamientos se agotó y la búsqueda terminó. Se mantiene el supuesto de transacciones vacías=0 y el alcance de clasificación de sesiones terminadas.
