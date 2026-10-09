# AutoResearch mínimo

El agente usa **GPT-OSS 120B en Groq**, una API remota con plan gratuito sujeto a cuotas. Necesita una API key de una cuenta del plan gratuito; la disponibilidad y los límites se consultan en [Groq](https://console.groq.com/docs/rate-limits). El programa no puede comprobar si tu cuenta tiene facturación habilitada. Usa JSON mode según la [API oficial](https://console.groq.com/docs/structured-outputs).

## Ejecutar

Crear una clave en [Groq Console](https://console.groq.com/keys) y definirla únicamente en el entorno:

```bash
read -rs -p "Groq API key: " GROQ_API_KEY
export GROQ_API_KEY
uv run python autoresearch.py --check-api
uv run python autoresearch.py --experiments 3
```

No requiere Ollama, Docker ni instalar un LLM. El entrenamiento sigue en CPU local. Los Parquet ya están generados; en otra instalación usar `uv sync` y ejecutar el preprocesamiento existente. Hasta 30 propuestas, 180 segundos por entrenamiento y predicción, cuatro threads y semilla 42. Las consultas al LLM tienen su propio timeout de 180 segundos. Un error HTTP de la API se registra y detiene la búsqueda, sin cambiar a un proveedor pago ni reintentos automáticos.

## Archivos esenciales

| Archivo | Responsabilidad |
|---|---|
| `autoresearch.py` | Entrada CLI. |
| `research_agent.py` | Una llamada HTTP al LLM y una respuesta JSON. |
| `research.py` | Validación, loop, timeout, evaluación y registro. |
| `research_train.py` | Entrenamiento y predicción con el pipeline existente. |
| `research_features.py` | Operaciones fijas por fila sobre las columnas. |
| `configs/autoresearch_features.json` | Features a agregar y quitar. |
| `configs/autoresearch_model.json` | Modelo e hiperparámetros. |
| `tests/test_autoresearch.py` | Verificación de los controles y del loop. |

Se eliminó el paquete `autoloop/`, sus seis módulos, las clases Workspace/Memory/Evaluator/Runner y el agente de smoke tests del código de producción. Se mantuvieron los tests y los resultados anteriores. Las propuestas simuladas se usan únicamente para verificar el sistema.

## Método y registros

1. Copiar una vez train y validación; guardar una copia pequeña del código de entrenamiento fijo.
2. Entrenar la baseline con las mismas condiciones que los candidatos.
3. Pedir una hipótesis al LLM, con las configuraciones, columnas disponibles y resultados recientes.
4. Validar los reemplazos y entrenar la configuración candidata.
5. Conservar sólo una reducción estricta de **log loss en mayo**. Brier es secundario.
6. Registrar el resultado y repetir desde la mejor configuración.

Sólo se admiten cambios en los dos JSON. Los originales del repositorio permanecen intactos; el mejor estado vive en memoria y en `best/`. Descartar un candidato consiste en conservar ese mejor estado, por lo que no hace falta revertir archivos del usuario. No se hacen commits automáticos.

En `experiments/autoresearch/<run>/`: `history.jsonl` guarda hipótesis, razonamiento, reemplazos exactos, scores, estado, duración y error; `summary.json` identifica el mejor experimento; `best/` contiene sus configuraciones; cada `run_NNN/` conserva modelo, configuraciones, predicciones y log. `data/` y `code/` mantienen las condiciones de la búsqueda. Los registros anteriores no se eliminaron.

El LLM recibe configuraciones y métricas, **no filas, usuarios ni credenciales**. El entrenamiento no recibe la API key. Las features se limitan a `log1p`, cocientes, productos, sumas, diferencias y descarte de columnas. No hay Python generado, comandos arbitrarios, estadísticas de validación ni un sandbox del sistema operativo.

Modelos definidos: logística, Random Forest, Extra Trees y LightGBM. Se reutilizan `load_data`, `build_model` y las métricas existentes. La historia, el target, los splits y el evaluador quedan protegidos. Junio permanece fuera del loop y se evalúa después de congelar el ganador. Ya se observó en la investigación anterior: no constituye un test completamente ciego.

## Verificación

Pasaron 13 tests de propuestas protegidas, API simulada, cuotas, timeout, mejora, empate, descarte, rollback y features, junto con los tests existentes de datos y baselines. La prueba con entrenamiento real reprodujo baseline 0.03757155 y conservó `hits_per_page` con 0.03754751, usando una propuesta simulada. La API real queda pendiente de configurar la clave; no se enviaron consultas remotas.

El cliente identifica la aplicación con `User-Agent` para evitar el bloqueo 1010 observado con Python urllib. `--check-api` verifica la clave y la disponibilidad del modelo sin entrenar ni solicitar inferencia. Los errores HTTP muestran el detalle del proveedor con la clave redactada. Pasaron 14 tests tras este ajuste.
