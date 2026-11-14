# Test temporal: junio de 2017

Los tres modelos se congelaron con sus umbrales de mayo antes de calcular estas métricas. Protocolo versionado en el commit `657e257`, disponible en [configs/test_evaluation.json](../configs/test_evaluation.json). No hubo entrenamiento, tuning ni selección de umbral con junio.

Conjunto: **63,578 sesiones y 946 compras**, tasa 1.488 %. La evaluación se ejecutó localmente el 14/11/2026; la ubicación no cambia el carácter temporal de la prueba.

## Comparación principal

| Modelo | Log loss ↓ | Average Precision ↑ | Umbral de mayo | Contactos / gasto simulado | Penalización total ↓ |
|---|---:|---:|---:|---:|---:|
| Tasa constante | 0.077659 | 0.014879 | No contactar | 0 | 9460 |
| LightGBM + Optuna | 0.030528 | 0.499220 | 0.266334 | 1186 | 7382 |
| AutoResearch independiente | 0.029069 | 0.550740 | 0.275096 | 1132 | 6930 |

AutoResearch reduce log loss **4.78 %** y penalización **6.12 %** frente a Optuna. Se seleccionan 1.132 contactos frente a 1.186. Son penalizaciones simuladas, no retorno económico causal de una campaña.

Escenario principal: TP=1, TN=0, FP=5, FN=10; contacto=1 unidad simulada. AutoResearch: TP=585, FP=547, FN=361, TN=62085; precisión 51,68 %, recall 61,84 %. El gasto directo no incluye las penalizaciones por oportunidades no seleccionadas.

## Sensibilidad a prioridades

| Escenario | Optuna | AutoResearch |
|---|---:|---:|
| budget | 4876 | 4698 |
| balanced | 4524 | 4414 |
| opportunities | 7382 | 6930 |

## Usuarios con y sin historia

| Modelo | Sin historia: log loss | Con historia: log loss |
|---|---:|---:|
| constant | 0.053859 | 0.190599 |
| optuna | 0.021636 | 0.072725 |
| autoresearch | 0.021459 | 0.065177 |

52.512 sesiones sin días previos y 11.066 con historia. Sus tasas de compra difieren; comparar modelos dentro de cada segmento, no pérdidas entre grupos como si tuvieran la misma población.

## Auditoría de exposición previa

Junio ya fue evaluado con modelos anteriores: el commit `43bc26a` registró logística (log loss 0,03325592) y LightGBM (0,02980344), después de congelar una selección anterior en `5ec76f1`. Esta evidencia se encontró al revisar Git; no era sólo una referencia a fechas o estructura.

La prueba actual evalúa modelos distintos, seleccionados en mayo, pero no debe presentarse como test ciego para todo el proyecto. No se encontró uso de junio en el fit o búsqueda de las corridas actuales; la auditoría no prueba ausencia de influencia humana fuera del código. No cambiar features ni umbrales para mejorar estas métricas. Si se toman decisiones de desarrollo a partir de ellas, reservar un período nuevo para la siguiente evaluación independiente.

## Reproducibilidad y MLOps

- El protocolo contiene hashes de test y de los artefactos, umbrales, costes y referencia a los snapshots. El comando rechaza cambios en los archivos congelados.
- El exportador guarda parámetros efectivos del estimador además del código original. Así se distingue configuración declarada de modificaciones realizadas en modeling.py.
- Resultados registrados en MLflow, experimento purchase-propensity-test. Los pesos, predicciones y estado SQLite se generan localmente y no se distribuyen en Git; este reporte conserva las métricas agregadas.
- [configs/candidate.json](../configs/candidate.json) vincula el candidato con validación, test, registro y preparación AWS. No se cambia automáticamente la aprobación por obtener una mejora en test.
- El batch de 64 sesiones de mayo verifica funcionamiento del despliegue, no calidad predictiva independiente. Puede repetirse la inferencia de junio en AWS sin enviar etiquetas y unirlas después, usando los mismos pesos y umbrales.

El procedimiento y comandos están en [evaluation.md](evaluation.md). Los resultados de mayo se conservan como [validación](independent-results.md), no como test final.
