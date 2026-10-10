# Autoresearch: propensión de compra

Empezá siempre por baselines: probabilidad constante y regresión logística. El
loop debe medirlas antes de pedir la primera propuesta. Sólo después investigá
una hipótesis por ensayo, partiendo del mejor estado confirmado.

Sólo podés editar estos archivos:

- `src/autoresearch_purchase_propensity/features.py`
- `src/autoresearch_purchase_propensity/modeling.py`

`features.py` define `build_features(frame)`: recibe las columnas de sesión y una
columna `history` con las sesiones anteriores disponibles. Puede agregar o quitar columnas, pero debe preservar el
índice y orden de filas. Las operaciones por fila no aprenden de validación.
Imputación, codificación y cualquier transformación aprendida se ajustan sólo
con train, dentro del pipeline.

`modeling.py` define `build_model(columns, config)`: devuelve un estimador con
`fit(X, y)` y `predict_proba(X)`. Conservá las interfaces y usá semilla 42. Empezá
por el modelo logístico suministrado. Podés cambiarlo después usando dependencias
ya instaladas.

Datos, target, disponibilidad del pasado, splits, entrenamiento y evaluación son fijos.
No leas archivos, ejecutes comandos, escribas artefactos ni consultes servicios
desde los dos archivos editables. No uses IDs, labels, transacciones, revenue,
fechas absolutas ni información de test como features. No accedas a test.
El objetivo es clasificar sesiones terminadas,
no predecir compras antes de que ocurran.

Entrená enero–abril y compará en mayo. Minimizá log loss; Average Precision es
complementaria. Sólo se conserva una mejora estricta. Empates, errores y timeouts
se descartan. No repitas hipótesis que ya se probaron.

La baseline inicial usa sólo la sesión actual. Después de medir las baselines,
investigá explícitamente si agregar el historial mejora la validación. Vos elegís
las agregaciones en `features.py`: contadores, tasas, promedios, recencias,
tendencias o ventanas de 7/30 días. Compará su contribución contra la baseline.

`frame['history']` contiene una lista de registros por fila. Cada registro incluye
los campos normalizados de una sesión anterior, `purchased` (0/1) y `days_ago`.
El contexto `data.history_fields` enumera los campos disponibles de esos registros.
Puede contener faltantes como `None`. Sólo aparecen sesiones de días estrictamente
anteriores del mismo usuario, también durante validación. No hay IDs, fechas
absolutas, revenue ni resultados de la sesión actual. No modifiques las listas
recibidas. Podés usar las compras HISTÓRICAS; el target actual sigue prohibido.
Agregá esas listas a columnas escalares y eliminá `history` antes del estimador.
El acceso al pasado es fijo; las variables derivadas son parte del experimento.

Ejemplo de feature histórica, a implementar dentro de `build_features`:

```python
out['purchases_last_7_days'] = frame.history.map(
    lambda events: sum(event['purchased'] for event in events if event['days_ago'] <= 7)
)
```

Respondé únicamente JSON con esta estructura:

```json
{"hypothesis": "Una hipótesis concreta", "files": {"src/autoresearch_purchase_propensity/features.py": "contenido Python completo del archivo modificado"}}
```

Incluí sólo los archivos que cambies y su contenido completo. Nunca devuelvas
comandos Git, parches ni modificaciones a otros archivos.
