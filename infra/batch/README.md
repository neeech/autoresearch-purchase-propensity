# Primera prueba batch: configuración preparada, no ejecutada

`smoke-demo-v1.json` define una sola instancia para procesar las 16 sesiones de
validación ya guardadas en S3. No son un batch sintético ni una evaluación de junio.
La salida mantiene el orden de las líneas y agrega `.out` al nombre del archivo.
S3Prefix selecciona todas las claves que comiencen con la URI indicada; conservar
esa ubicación sin objetos adicionales que compartan el prefijo.

Antes de ejecutar: verificar que la cuota aplicada sea al menos 1, revisar el precio
regional vigente y acordar el seguimiento/parada del job. Solicitar una cuota no
es aprobarla. Este JSON no ha sido enviado a AWS.

`InvocationsTimeoutInSeconds=60` limita cada solicitud HTTP, NO la duración total
del job ni el coste. Los reintentos HTTP están desactivados para esta prueba.
Cada nuevo job necesita un nombre único y un prefijo de salida propio. La
inferencia no crea un endpoint persistente.

La definición del modelo, su rol IAM, la imagen ECR y el paquete S3 ya se crearon
siguiendo la práctica. Los permisos y el servidor aún deben verificarse en AWS.

Referencia: https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-transform-job.html
