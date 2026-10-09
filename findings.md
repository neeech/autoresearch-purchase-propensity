# Autoresearch LightGBM

Objetivo: minimizar log loss en mayo; Brier Score secundario. Referencia logística: log loss 0,03757.

Metodología: agente conversacional propone hiperparámetros; protocolo y código se commitean antes de entrenar. Features actuales fijas (month excluido), categorías aprendidas sólo con train, desconocidas como faltantes. CPU local, sin early stopping ni pesos de clase. Enero–abril para entrenamiento, mayo para selección, junio reservado.

Presupuesto: 20 entrenamientos o 60 minutos; se detiene al alcanzar cualquiera. Los errores cuentan como intentos. Todos los candidatos se conservan; el mejor se elige estrictamente por log loss. Revisión cada cinco experimentos.

## Hallazgos

Pendientes de la primera ejecución.

### Revisión tras 5 experimentos

El mínimo de 30 sesiones por hoja mejora la referencia: log loss 0,033878 frente a 0,034000, con 31 hojas y 300 árboles. 63 hojas empeoran; 15 no mejora. Un mínimo de 300 sesiones reduce Brier pero no log loss, por lo que no se promueve. Se conserva el mínimo de 30 como candidato actual. Próxima etapa: regularización L2 y duración del boosting, manteniendo la selección por log loss.

### Revisión tras 10 experimentos

Mejor: run_008, 150 árboles, learning_rate=0,05, 31 hojas, mínimo 30 sesiones, L2=1. Log loss 0,033772 y Brier 0,010469. 75 árboles empeoran; 225 y 300 tampoco superan 150. L2=5 mejora Brier ligeramente, pero empeora log loss; sin L2 también empeora. Próxima etapa: menor learning rate con cantidad de árboles ajustada y muestreo, sin early stopping.

### Revisión tras 15 experimentos

Reducir learning_rate a 0,025 y usar 300 árboles mejora ambas métricas frente a 0,05/150. 0,015/500 no aporta mejora. Muestrear features al 80% no mejora; muestrear filas al 60% cada iteración da el mejor log loss (run_015: 0,033508; Brier 0,010396). Estas son diferencias exploratorias en una única validación, no pruebas de superioridad estadística. Próxima etapa: complejidad y número de árboles alrededor del candidato, y confirmación de reproducibilidad.

### Cierre: 20 entrenamientos (19 configuraciones y una repetición)

Ganador por menor log loss: run_017 (300 árboles, learning_rate=0,025, 47 hojas, mínimo 30 sesiones, L2=1, muestreo de filas=0,6 cada iteración). Log loss 0,033507806 y Brier 0,010379287. run_020 reprodujo exactamente sus probabilidades; recargar el modelo nativo también reproduce las guardadas.

Mejora relativa frente a logística: 10,82% en log loss; Brier baja de 0,011327090 a 0,010379287. Run_015 (31 hojas) está prácticamente empatado en log loss (diferencia 0,000000125); no se puede afirmar superioridad estadística de 47 hojas. 450 árboles empeora; 225 queda casi empatado con 300 en la última región. Estos resultados seleccionan un candidato en mayo, no demuestran rendimiento futuro.

Presupuesto agotado por número de entrenamientos; no se inicia otra ejecución. Junio no fue leído para entrenar ni evaluar. No se hizo calibración, búsqueda de features ni parada temprana. El modelo seleccionado sigue entrenado sólo en enero–abril. Próxima evaluación de junio requiere cerrar previamente cualquier decisión de selección y reentrenamiento.

### Evaluación final de junio: proyecto cerrado

Selección y hashes congelados antes de test en commit 5ec76f1. Se evaluaron una sola vez los modelos entrenados en enero–abril, sin reentrenamiento. LightGBM: log loss 0,029803438, Brier 0,009212972. Logística: log loss 0,033255924, Brier 0,009982702. Reducciones relativas: 10,38% y 7,71%. Mejora presente tanto con como sin historia previa.

Esta ejecución queda como referencia de la skill para la futura tool propia en otra branch. Junio ya fue observado; no utilizarlo para dirigir esa búsqueda. Para comparación justa mantener datos, features, presupuesto, recursos y métricas. No se instrumentó el costo en tokens del agente. Detalles en reports/autoresearch.md.
