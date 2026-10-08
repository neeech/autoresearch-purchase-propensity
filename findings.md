# Autoresearch LightGBM

Objetivo: minimizar log loss en mayo; Brier Score secundario. Referencia logística: log loss 0,03757.

Metodología: agente conversacional propone hiperparámetros; protocolo y código se commitean antes de entrenar. Features actuales fijas (month excluido), categorías aprendidas sólo con train, desconocidas como faltantes. CPU local, sin early stopping ni pesos de clase. Enero–abril para entrenamiento, mayo para selección, junio reservado.

Presupuesto: 20 entrenamientos o 60 minutos; se detiene al alcanzar cualquiera. Los errores cuentan como intentos. Todos los candidatos se conservan; el mejor se elige estrictamente por log loss. Revisión cada cinco experimentos.

## Hallazgos

Pendientes de la primera ejecución.
