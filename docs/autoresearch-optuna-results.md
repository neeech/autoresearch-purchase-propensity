# Optuna + AutoResearch: investigación incremental (antecedente)

Corrida `20261114T172825124860Z`, Gemini `gemini-3.6-flash`, 14/11/2026.
Partió del LightGBM ganador de Optuna: mismos hiperparámetros y mismos archivos
congelados (hashes comprobados). Su baseline reprodujo log loss 0,03407013.
No partió de un LightGBM sin optimizar ni continuó la investigación anterior.

**No es una comparación independiente de Optuna contra AutoResearch:** el agente
recibió la configuración ganadora de Optuna. Se conserva como antecedente de
mejora incremental. El protocolo vigente está en [model-comparison.md](model-comparison.md).

## Resultados en mayo

65.371 sesiones; mismos IDs y etiquetas verificados para las tres referencias.
Escenario principal TP=1, TN=0, FP=5, FN=10; contacto=1 unidad simulada.
Cada umbral se seleccionó en mayo.

| Modelo | Log loss | Average Precision | Umbral | Contactos/gasto directo | Penalización total |
|---|---:|---:|---:|---:|---:|
| Tasa constante | 0,087228 | 0,017056 | No contactar | 0 | 11150 |
| LightGBM + Optuna | 0,034070 | 0,537777 | 0,266334 | 1340 | 8274 |
| AutoResearch desde Optuna | 0,033218 | 0,559736 | 0,236376 | 1474 | 8020 |

AutoResearch reduce log loss 2,50 % y penalización principal 3,07 % frente a
Optuna. Requiere 134 contactos adicionales: no significa gastar menos dinero.
El coste simulado mejora al reducir oportunidades no seleccionadas. Precisión
50,88 %, recall 67,26 %; TP=750, FP=724, FN=365, TN=63532.

## Sensibilidad

| Escenario | Penalización Optuna | Penalización AutoResearch |
|---|---:|---:|
| budget | 5441 | 5477 |
| balanced | 5261 | 5146 |
| opportunities | 8274 | 8020 |

Mejor log loss no garantiza menor penalización en todos los escenarios; bajo
budget, Optuna obtiene un resultado ligeramente mejor.

## Qué hizo el agente

Presupuesto solicitado: 20 propuestas, timeout de entrenamiento 180 s, máximo
10 llamadas por minuto. Se completaron 13 propuestas entrenadas: 3 keep y 10
discard. El intento 14 no entrenó: Gemini informó cuota diaria agotada. El baseline
es la ejecución 0. No se completaron los 20 ensayos. Reanudar requiere conservar su workspace
y protocolo originales; no se presupone que estén disponibles en otro clon.

Mejoras aceptadas: (1) conteos/compras/recencias históricas; (4) tasas de compra,
ventanas temporales y promedios de actividad histórica; (12) diferencia entre hits
y pageviews como señal derivada. La hipótesis del LLM sobre qué representa esa
diferencia no está confirmada por el esquema: no equivale necesariamente a eventos
de carrito. El estimador ganador sigue siendo LightGBM con los parámetros de
Optuna. Varias hipótesis mencionan regresión logística por error o proponen cambios
ya presentes; los empates se descartaron. Se informa el código efectivo, no sólo
la explicación del LLM.

## Trazabilidad y límites

- Mejor commit: `5d0f01f59f641b927b3f0f523d2e52169c1d8cf2`, experimento 12.
- Predicciones reproducidas en un proceso nuevo con el snapshot original; evaluación
  registrada en MLflow, experimento `purchase-propensity-marketing`.
- Este reporte conserva resultados agregados; los pesos, workspaces y registros
  MLflow de esa ejecución no se distribuyen con el repositorio.
- Se mantiene un warning de división por cero en agregaciones históricas; las
  probabilidades y métricas finales son finitas. Revisarlo antes de aprobación.
- Mayo se utilizó para tuning, selección y umbrales: evaluación exploratoria,
  potencialmente optimista. Junio no fue usado ni leído para esta comparación.
- El nuevo modelo no fue aprobado, integrado al código principal ni desplegado.
  Requiere revisión y un adaptador/imagen nuevos antes de utilizarlo en AWS.

Duración acumulada registrada por el controlador: 667.9 s (incluye LLM y entrenamiento, no sólo fit).
