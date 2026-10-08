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
