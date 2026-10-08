# Análisis exploratorio de sesiones

Generado por `eda.py` a partir de los datos normalizados, antes de construir historia.

392,892 sesiones, 308,130 visitantes y 5,277 sesiones con compra (1.34%). Fechas: 2017-01-01 a 2017-06-30.

## Volumen, compras e historia

| Mes | Sesiones | Con compra | Tasa de compra | Visitantes nuevos observados | Sesiones con historia de días anteriores |
|---|---:|---:|---:|---:|---:|
| 2017-01 | 64,694 | 697 | 1.08% | 100.00% | 12.11% |
| 2017-02 | 62,192 | 708 | 1.14% | 95.77% | 15.49% |
| 2017-03 | 69,931 | 883 | 1.26% | 95.48% | 15.72% |
| 2017-04 | 67,126 | 928 | 1.38% | 94.85% | 16.17% |
| 2017-05 | 65,371 | 1,115 | 1.71% | 93.72% | 18.60% |
| 2017-06 | 63,578 | 946 | 1.49% | 93.84% | 17.41% |

Un visitante nuevo observado no necesariamente es nuevo para el sitio: no tenemos historia anterior al inicio del archivo. La proporción de visitantes nuevos y la proporción de sesiones sin historia tienen denominadores diferentes.

![Evolución mensual](figures/monthly.png)

## Calidad y decisiones

- El target es `transactions > 0`. Se asume explícitamente que un valor vacío significa cero compras; debe confirmarse con la fuente. Hay 0 filas con ingresos positivos sin compra y 0 compras sin ingresos registrados.
- `transactions` y `transactionRevenue` se excluyen del modelo. Los totales de actividad describen sesiones terminadas y pueden incluir consecuencias de la compra; el benchmark no demuestra anticipación online.
- `fullVisitorID` se conserva como texto para historia y auditoría, pero se excluye de las features.
- `timeOnSite` falta en 50.01%; `bounces` en 50.12%. No se reemplazan automáticamente por cero. Se agregan indicadores de faltantes.
- Los textos `(not set)` y `not available in demo dataset` se normalizan como faltantes en categorías. La imputación y codificación se ajustarán sólo con entrenamiento.
- Hay 59,346 filas de usuarios con múltiples sesiones el mismo día. La historia usa únicamente días anteriores, evitando inventar un orden intradía.
- Resúmenes de sesión duplicados: 0. No se eliminan automáticamente: no hay un identificador único de sesión en la fuente. `session_row_id` identifica la fila original.
- Historia sin observaciones: contadores en cero, promedios/recencia desconocidos como faltantes y `has_history=0`.

## Evaluación propuesta

Entrenar enero–abril, validar mayo y reservar junio para test. La historia se actualiza secuencialmente con días anteriores, incluso dentro de validación/test, simulando disponibilidad de sesiones ya terminadas. Esto no ajusta parámetros del modelo con esos períodos.

Comparar baseline sin historia y candidato con historia. Usar log loss como objetivo y Average Precision, Brier score y calibración como métricas complementarias; reportar nuevos observados y recurrentes por separado. No seleccionar modelos ni features con resultados del test. Este EDA describe todo el período y no constituye un test estadístico completamente ciego.

Los CSV adjuntos contienen faltantes, distribuciones numéricas y tasas por categoría. Las asociaciones son descriptivas; no prueban causalidad. La tabla mensual permite revisar desbalance y cambios temporales antes de entrenar.
