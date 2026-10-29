# Autoresearch: propensión de compra

Tu tarea es proponer un cambio de features o modelo que mejore el log loss de
validación. En cada ensayo recibís el mejor código confirmado, su score y el
historial de experimentos. Devolvé una hipótesis y los archivos modificados.
El loop se encarga de entrenar, evaluar, crear los commits y conservar las mejoras.

Trabajá sobre estos archivos:

- `src/autoresearch_purchase_propensity/features.py`
- `src/autoresearch_purchase_propensity/modeling.py`

`features.py` define `build_features(frame)`: recibe las columnas de sesión y una
columna `history` con las sesiones anteriores disponibles. Puede agregar o quitar
columnas, pero debe preservar el índice y orden de filas. Las operaciones por
fila no aprenden de validación.
Imputación, codificación y cualquier transformación aprendida se ajustan sólo
con train, dentro del pipeline.

`modeling.py` define `build_model(columns, config)`: devuelve un estimador con
`fit(X, y)` y `predict_proba(X)`. Conservá las interfaces y usá semilla 42.
Partí del modelo recibido en el contexto y explorá alternativas usando las
dependencias disponibles. Un ensayo que modifica sólo `modeling.py` es válido: podés cambiar
la familia del estimador, sus hiperparámetros o el preprocesamiento aprendido,
manteniendo las features actuales.

Dependencias y algoritmos disponibles según `pyproject.toml` (entorno preparado
según las versiones mínimas declaradas):

| Dependencia | Opciones para clasificación binaria |
|---|---|
| `scikit-learn>=1.9.1` | `LogisticRegression`, `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `HistGradientBoostingClassifier`, `SGDClassifier(loss='log_loss')`, entre otros |
| `xgboost>=3.4.1` | `xgboost.XGBClassifier`: boosting de árboles, por ejemplo con `tree_method='hist'` y `objective='binary:logistic'` |
| `lightgbm>=4.7.0` | `lightgbm.LGBMClassifier`: boosting de árboles con distintas profundidades, hojas y regularización |
| `catboost>=1.2.10` | `catboost.CatBoostClassifier`: boosting, con posibilidad de tratamiento nativo de categóricas |
| `pandas>=2.2`, `pyarrow>=18` | Manipulación de datos y soporte Parquet |

Podés elegir entre estas opciones y otros algoritmos disponibles en el entorno.
Prepará propuestas para CPU, ajustá su complejidad y usá hasta 4 hilos donde el
estimador lo permita, para respetar tiempo y memoria disponibles.

Conservá `build_features` como primer paso del pipeline y elegí las columnas
después de generar las features, para que las nuevas variables entren al modelo.
El código fijo llama `model.fit(train_x, train_y)` y luego
`model.predict_proba(val_x)[:, 1]`: devolvé un estimador compatible, con probabilidad
de compra en la segunda columna. Configurá el pipeline para esas llamadas.
Cualquier calibración, selección aprendida o validación interna debe usar
exclusivamente train.

Adaptá el preprocesamiento al formato y tratamiento de faltantes que acepta el
estimador. Si usás categóricas nativas, prepará los
tipos y faltantes y declaralas según la API del modelo. Si usás one-hot, controlá
la cardinalidad y elegí una representación que entre en memoria.
En CatBoost configurá `allow_writing_files=False` para evitar artefactos del
estimador dentro del worktree.
Elegí modelos e hiperparámetros buscando mejorar el log loss de validación.
Si probás pesos de clase en `modeling.py`, justificá su uso como una hipótesis
y compará su efecto sobre el log loss registrado por el loop.

Implementá las transformaciones sobre los datos recibidos en memoria. Derivá las
features de las columnas de sesión disponibles y del historial de días anteriores.
El objetivo es clasificar sesiones terminadas: el target es `transactions > 0`.

El loop entrena con enero–abril y evalúa en mayo. Minimizá log loss; Average Precision es
complementaria. Sólo se conserva una mejora estricta. Empates, errores y timeouts
se descartan. Usá el historial de experimentos recibido en el contexto para elegir
la próxima hipótesis y explicar brevemente por qué vale la pena probarla. Evitá
repetir experimentos equivalentes; podés reevaluar una feature con otra familia
de modelos si justificás la interacción que querés comprobar.

La ejecución comienza midiendo las baselines y luego solicita propuestas.
Investigá el aporte del historial y compará alternativas de modelos. Vos elegís
las agregaciones en `features.py`: contadores, tasas, promedios, recencias,
tendencias o ventanas de 7/30 días. Compará su contribución contra la baseline.

Explorá tanto features como modelos. En los primeros ensayos, compará al menos
una familia de modelos alternativa a la actual, manteniendo las features fijas,
si el presupuesto de ensayos lo permite. Explorá `modeling.py` mientras seguís
investigando features.

Podés intercalar ensayos de features, de modelo y de hiperparámetros según los
resultados. Elegí el orden según la evidencia del historial. Si llevás varios
ensayos consecutivos en una sola dirección, considerá explícitamente explorar
la otra. Reservá ensayos para comparar modelos incluso cuando las features
siguen dando pequeñas mejoras.

Para atribuir el resultado, preferí cambiar una dimensión por ensayo: conservá
las features al comparar modelos y el modelo al comparar features. Podés modificar
ambos archivos cuando una única hipótesis lo requiera, explicando el motivo.
El aporte de una feature depende del estimador: que no mejore un modelo no implica
que sea inútil para otra familia. Partí siempre del mejor estado confirmado.

`frame['history']` contiene una lista de registros por fila. El cargador fijo
normaliza los arrays provenientes de Parquet a listas Python; sin sesiones
previas se recibe `[]`. Cada registro incluye los campos normalizados de una
sesión anterior, `purchased` (0/1) y `days_ago`.
El contexto `data.history_fields` enumera los campos disponibles de esos registros.
Puede contener faltantes como `None`. Sólo aparecen sesiones de días estrictamente
anteriores del mismo usuario, también durante validación. Derivá nuevas columnas
manteniendo intactas las listas recibidas. Usá `purchased` de esas sesiones
históricas para construir tasas, contadores o recencias de compra.
Agregá esas listas a columnas escalares y eliminá `history` antes del estimador.
El acceso al pasado es fijo; las variables derivadas son parte del experimento.
El historial de experimentos del contexto es distinto de `frame['history']`:
el primero describe los ensayos anteriores; el segundo contiene sesiones pasadas.

Ejemplo de feature histórica, a implementar dentro de `build_features`:

```python
out['purchases_last_7_days'] = frame.history.map(
    lambda events: sum(event['purchased'] for event in events if event['days_ago'] <= 7)
)
```

Respondé únicamente JSON con esta estructura: un objeto con exactamente
`hypothesis` (string) y `files` (objeto de rutas a contenido Python completo).
En `files` incluí sólo los archivos modificados: `features.py`, `modeling.py` o
ambos, usando las rutas completas relativas al repositorio de los ejemplos.
Una propuesta puede modificar exclusivamente `modeling.py`.

En `hypothesis` explicá el cambio, el efecto esperado sobre log loss y por qué
lo elegís a partir de los resultados anteriores. Una hipótesis debe poder
evaluarse con un solo entrenamiento: elegí una configuración concreta por ensayo.

Los siguientes ejemplos son propuestas independientes y muestran archivos
completos a partir de la baseline suministrada. Usalos como referencia de formato.
Adaptalos al mejor código recibido en `files`: si ya hay
features o preprocesamiento aceptados, conservalos salvo que cambiarlos sea parte
explícita de la hipótesis.

Ejemplo 1: modificar sólo `modeling.py` para comparar Random Forest con el
modelo actual, conservando las features y el preprocesamiento de la baseline.

```json
{
  "hypothesis": "Comparar Random Forest con el modelo actual para capturar interacciones no lineales, manteniendo las features y el preprocesamiento de la baseline. Usar la comparación de familias pendiente en el historial.",
  "files": {
    "src/autoresearch_purchase_propensity/modeling.py": "\"\"\"Example model factory; keep the train-fitted preprocessing.\"\"\"\nfrom sklearn.compose import ColumnTransformer, make_column_selector\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler\nfrom .features import build_features\nfrom sklearn.ensemble import RandomForestClassifier\n\n\ndef build_model(columns, config):\n    categories = Pipeline([\n        ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__')),\n        ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist',\n                                 min_frequency=config['min_category_frequency'])),\n    ])\n    numbers = Pipeline([\n        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),\n        ('scale', StandardScaler()),\n    ])\n    transform = ColumnTransformer([\n        ('categorical', categories, make_column_selector(dtype_exclude='number')),\n        ('numeric', numbers, make_column_selector(dtype_include='number')),\n    ], sparse_threshold=1.0)\n    return Pipeline([\n        ('prepare', FunctionTransformer(build_features, validate=False)),\n        ('features', transform),\n        ('model', RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,\n                                         n_jobs=4, random_state=config['seed'])),\n    ])\n"
  }
}
```

Ejemplo 2: modificar sólo `modeling.py` para probar boosting con XGBoost.

```json
{
  "hypothesis": "Comparar XGBoost con el mejor modelo confirmado manteniendo las features y el preprocesamiento de la baseline. Probar árboles poco profundos y regularizados como otra familia candidata.",
  "files": {
    "src/autoresearch_purchase_propensity/modeling.py": "\"\"\"Example model factory; keep the train-fitted preprocessing.\"\"\"\nfrom sklearn.compose import ColumnTransformer, make_column_selector\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler\nfrom .features import build_features\nfrom xgboost import XGBClassifier\n\n\ndef build_model(columns, config):\n    categories = Pipeline([\n        ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__')),\n        ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist',\n                                 min_frequency=config['min_category_frequency'])),\n    ])\n    numbers = Pipeline([\n        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),\n        ('scale', StandardScaler()),\n    ])\n    transform = ColumnTransformer([\n        ('categorical', categories, make_column_selector(dtype_exclude='number')),\n        ('numeric', numbers, make_column_selector(dtype_include='number')),\n    ], sparse_threshold=1.0)\n    return Pipeline([\n        ('prepare', FunctionTransformer(build_features, validate=False)),\n        ('features', transform),\n        ('model', XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,\n                                subsample=0.8, colsample_bytree=0.8, reg_lambda=5,\n                                objective='binary:logistic', eval_metric='logloss',\n                                tree_method='hist', n_jobs=4, random_state=config['seed'])),\n    ])\n"
  }
}
```

Ejemplo 3: modificar sólo `features.py` para medir el aporte de un contador
histórico, manteniendo el modelo actual.

```json
{
  "hypothesis": "Agregar el número de sesiones anteriores a las features de la baseline para medir su aporte al modelo actual. Comprobar en el historial que esta combinación aún no se evaluó.",
  "files": {
    "src/autoresearch_purchase_propensity/features.py": "\"\"\"Example session features with a historical session count.\"\"\"\nimport numpy as np\n\nCATEGORIES = ['source', 'channelGrouping', 'browser', 'deviceCategory', 'country', 'city', 'day_of_week']\nLOG_COLUMNS = ['visitNumber', 'hits', 'pageviews', 'timeOnSite']\n\n\ndef build_features(frame):\n    out = frame.copy()\n    out['historical_sessions'] = frame['history'].map(len)\n    out = out.drop(columns='history')\n    if 'day_of_week' in out:\n        out['is_weekend'] = out.day_of_week.ge(5).astype('int8')\n    for name in ('pageviews', 'timeOnSite'):\n        if name in out:\n            out[f'{name}_missing'] = out[name].isna().astype('int8')\n    for name in out:\n        if name in CATEGORIES:\n            values = out[name].astype('string')\n            out[name] = values.astype(object).where(values.notna(), np.nan)\n        elif name in LOG_COLUMNS:\n            out[name] = np.log1p(out[name].astype(float))\n    return out\n"
  }
}
```

También podés devolver ambos archivos en el mismo objeto `files` cuando la
hipótesis requiera una adaptación conjunta de features y modelo; incluí los dos
contenidos completos y justificá esa necesidad en `hypothesis`.

Devolvé el objeto JSON directamente, con una hipótesis concreta y el contenido
Python completo de cada archivo modificado, listo para aplicar y entrenar.
