# Cierre del proyecto — 14 de noviembre de 2026

## Motivo

El target utilizado fue compra en una sesión terminada, con variables agregadas
de esa misma sesión. Esto permite clasificación retrospectiva, pero no establece
un momento útil para anticipar la compra y decidir una campaña. La actividad puede
incluir eventos posteriores a la transacción y, al terminar la sesión, la compra
normalmente ya es conocida. Ocultar la etiqueta no resuelve esa limitación.

Los resultados registrados describen ese problema retrospectivo; no demuestran
predicción de compra futura ni impacto económico de campañas. Se cierra esta
formulación y no se continúa con su despliegue o automatización.

## Trabajo conservado

Se mantienen EDA, procesamiento e historia temporal, baselines, Optuna,
AutoResearch, evaluación congelada de junio, tracking y registro local MLflow,
paquetes y servidor Docker. Ver [resultados](test-results.md) y
[arquitectura técnica](architecture.md). Los modelos y registros locales no se
incluyen en Git. No hay un modelo aprobado para producción.

La inferencia se verificó localmente. Se publicaron imágenes en ECR, paquetes y
muestras en S3, y definiciones de modelo SageMaker; no se completó un job batch AWS.
La cuota pendiente deja de ser un bloqueo de desarrollo porque el proyecto terminó.

## Recursos remotos

El cierre documental no elimina infraestructura. Revisar los recursos del proyecto:

- Bucket S3 purchase-propensity-demo-7f4c2a y sus objetos.
- Repositorio ECR purchase-propensity-inference y sus imágenes.
- Definiciones SageMaker purchase-propensity-demo-v1 y purchase-propensity-candidate-v2.
- Rol IAM purchase-propensity-batch-role y su política.
- Solicitud de cuota y presupuesto de facturación de la cuenta.

El almacenamiento puede consumir créditos aunque no haya jobs. Eliminar recursos
requiere una decisión explícita sobre qué conservar; no borrar la cuenta ni recursos
compartidos. No se programó inferencia periódica en este proyecto.

## Proyecto posterior, separado

El siguiente repositorio usará Online Retail II para estimar recompra de clientes
con al menos una compra conocida en los próximos 30 días, con un corte de información
explícito. Los visitantes sin compras quedan fuera de ese alcance. Este repositorio
no implementa ese objetivo y sus resultados no se trasladan al nuevo target.
