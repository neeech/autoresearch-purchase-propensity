# Rol para Batch Transform

Nombre propuesto: `purchase-propensity-batch-role`.
La confianza permite `sts:AssumeRole` a `sagemaker.amazonaws.com`.

La política `sagemaker-batch-policy.json` es específica de nuestra cuenta y recursos.
Adjuntarla como política inline al rol, con nombre `purchase-propensity-batch-access`,
sin mantener `AmazonSageMakerFullAccess` en ese rol.

Permite leer `models/` y `batches/` del bucket del proyecto, escribir `predictions/`,
descargar la imagen ECR y publicar logs de TransformJobs. No permite borrar objetos,
publicar imágenes, iniciar entrenamientos ni administrar IAM. La autenticación ECR
usa `Resource: "*"` porque GetAuthorizationToken no admite limitar por repositorio;
las operaciones de descarga sí se restringen al repositorio del proyecto.

La identidad humana que envía CreateModel/CreateTransformJob necesita sus propios
permisos, incluido iam:PassRole para este rol. Esta política de ejecución no los otorga.
Usa objetos S3 con cifrado SSE-S3; SSE-KMS requeriría permisos adicionales sobre la clave.
Las llamadas reales aún deben verificar que no existan otras restricciones de cuenta.

Referencia: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html
