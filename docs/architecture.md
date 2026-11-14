# Diseño del proyecto

Dos flujos separados comparten un modelo versionado:

```text
Tasa constante / LightGBM+Optuna / AutoResearch independiente (local, log loss)
  → candidato congelado → evaluación probabilística y costes simulados
  → tracking MLflow local → revisión humana → empaquetado
  → S3 (modelo) + ECR (imagen) → definición SageMaker

Lote de sesiones terminadas en S3 → Batch Transform → probabilidades en S3
                                          ↓
                                     logs CloudWatch
```

IAM limita los accesos del rol de ejecución. Budget y créditos permiten observar
costes; no son un mecanismo de parada exacta. Primer ensayo: una ml.m5.large en
Ohio, 16 sesiones; seguimiento manual, solicitar parada si supera 15 minutos y
confirmar Stopped. Cuota pendiente de aprobación; no se lanzó el job.

La imagen demo-v1 conserva su modelo y contrato de probabilidades. Los umbrales
son una decisión posterior versionada con los escenarios; no están embebidos en
el servidor de inferencia. El candidato demo no equivale a modelo aprobado.
La selección actual vive en configs/candidate.json; enlaza snapshot, métricas,
umbrales y hashes. Ver docs/independent-results.md. El código genérico no se
sobrescribe automáticamente con el ganador y demo-v1 conserva su versión anterior.

## Implementado y pendiente

- Implementado: procesamiento temporal, entrenamiento, AutoResearch, evaluador,
  tests, servidor Docker de inferencia, configuración IAM y batch, evaluación de
  costes y tracking opcional de sus reportes.
- AWS preparado: S3, ECR y definición de modelo. Pendiente ejecutar el primer
  batch y comprobar resultados y permisos reales.
- Pendiente: registrar modelos destacados con sus snapshots en MLflow, enlazar
  aprobación con versión de despliegue, batches simulados con historia coherente,
  reporte diario y orquestación periódica de inferencia con reintentos controlados.
- Optuna y AutoResearch se ejecutan manualmente; ver docs/model-comparison.md.
  El reentrenamiento y la revisión trimestral siguen manuales. No se programa el
  loop de AutoResearch ni se incorpora HPO de SageMaker en paralelo.
