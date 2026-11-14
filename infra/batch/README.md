# Configuración de Batch Transform

[smoke-demo-v1.json](smoke-demo-v1.json) es un ejemplo de job para una sola
instancia ml.m5.large en us-east-2. Configura entrada/salida S3, JSON Lines,
SplitType=Line, AssembleWith=Line y MultiRecord. No crea recursos al estar en Git.

El JSON usa nombres y rutas de una cuenta de demostración: **no es ejecutable
sin preparar primero esos recursos**. Un clon contiene configuración y código,
no los modelos, buckets, permisos ni imágenes remotos.

## Requisitos

1. Probar el modelo con el [servidor compatible](../../inference/README.md).
2. Subir una imagen versionada a ECR y un paquete de modelo a S3.
3. Crear el rol con confianza SageMaker y [permisos IAM](../iam/README.md).
4. Registrar un modelo SageMaker con imagen, rol y paquete correctos.
5. Preparar JSON Lines procesado en S3. No necesita contener etiquetas.
6. Adaptar nombres y URI del JSON; usar nombre de job y salida únicos.
7. Consultar cuota aplicada del tipo de instancia y revisar coste regional.

La preparación histórica de la demo no equivale a verificación del candidato
actual. El primer job aún no se ha validado. El servidor actualizado admite LightGBM, pero la imagen local candidate-v2
aún debe publicarse y asociarse a una definición nueva de modelo para
[el candidato](../../configs/candidate.json). No ejecutar el JSON antiguo
esperando que use automáticamente el candidato nuevo.

## Ejecución y seguimiento

El ejemplo contiene un prefijo S3 que selecciona todas las claves que empiecen
por esa URI; evitar objetos adicionales con ese prefijo. Los resultados agregan
.out al nombre del archivo y mantienen el orden de registros por archivo.

InvocationsTimeoutInSeconds=60 es timeout HTTP, no duración máxima ni tope de
coste del job. Seguir el primer ensayo manualmente; si supera 15 minutos,
solicitar StopTransformJob y confirmar Stopped. La parada no es instantánea.
Consultar estado con DescribeTransformJob y errores con FailureReason/CloudWatch.
Al completar, verificar número de salidas, probabilidades y equivalencia local.

No se mantiene un endpoint de inferencia permanente. S3, ECR y logs conservan
archivos después del job y pueden generar costes. El presupuesto de facturación
no sustituye el seguimiento de ejecución.

Referencia: https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-transform-job.html
