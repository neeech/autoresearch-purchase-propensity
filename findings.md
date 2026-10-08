# Autoresearch LightGBM

Objetivo: minimizar log loss en mayo; Brier Score secundario. Referencia logística: log loss 0,03757.

Metodología: agente conversacional propone hiperparámetros; protocolo y código se commitean antes de entrenar. Features actuales fijas (month excluido), categorías aprendidas sólo con train, desconocidas como faltantes. CPU local, sin early stopping ni pesos de clase. Enero–abril para entrenamiento, mayo para selección, junio reservado.

Presupuesto: 20 entrenamientos o 60 minutos; se detiene al alcanzar cualquiera. Los errores cuentan como intentos. Todos los candidatos se conservan; el mejor se elige estrictamente por log loss. Revisión cada cinco experimentos.

## Hallazgos

Pendientes de la primera ejecución.

### Revisión tras 5 experimentos

El mínimo de 30 sesiones por hoja mejora la referencia: log loss 0,033878 frente a 0,034000, con 31 hojas y 300 árboles. 63 hojas empeoran; 15 no mejora. Un mínimo de 300 sesiones reduce Brier pero no log loss, por lo que no se promueve. Se conserva el mínimo de 30 como candidato actual. Próxima etapa: regularización L2 y duración del boosting, manteniendo la selección por log loss.
