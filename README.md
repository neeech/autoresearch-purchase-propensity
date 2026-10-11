# Autoresearch: propensión de compra

POC de un loop con commits: proponer un cambio, entrenar, evaluar en validación
y conservarlo sólo si mejora log loss. Clasifica sesiones terminadas.

## Preparar y ejecutar

Requiere Python >= 3.13, uv y Git con nombre/email configurados. Desde la raíz:

```bash
uv sync
uv run python -m autoresearch_purchase_propensity.preprocess
uv run python -m unittest discover -s tests -v
```

Definir `GROQ_API_KEY` en el entorno y commitear la implementación antes de
iniciar el loop, que requiere un checkout limpio:

```bash
uv run autoresearch --experiments 10 --timeout 180
```

Usa HTTP directamente, sin SDK ni framework de agentes. Groq es el proveedor
predeterminado y usa `openai/gpt-oss-120b`. El modelo se puede cambiar con
`--model`; su disponibilidad depende del proveedor.

Para usar Gemini con una clave de [Google AI Studio](https://aistudio.google.com/apikey),
definir `GEMINI_API_KEY` en el entorno. Ejemplo en PowerShell:

```powershell
$env:GEMINI_API_KEY = 'TU_API_KEY'
uv run autoresearch --provider gemini --model gemini-3.8-flash --experiments 10 --timeout 600
```

Con `--provider gemini`, el modelo predeterminado es `gemini-3.8-flash`, por lo
que `--model` es opcional. Usa la API nativa `generateContent` con respuesta JSON
estructurada y los mismos reintentos que Groq. Ninguna de las dos claves se pasa
al entrenamiento. Commiteá los cambios antes de iniciar una nueva ejecución.

Sólo baselines, sin API:

```bash
uv run autoresearch --experiments 0
```

Propuestas guardadas, sin API:

```bash
uv run autoresearch --experiments 2 --proposals propuestas.json
```

El JSON contiene una lista de propuestas según [program.md](program.md).

## Contrato

| Archivo | Responsabilidad | Editable por el LLM |
|---|---|---|
| `history.py` | Qué sesiones anteriores puede ver cada fila | No |
| `preprocess.py` | Normalización, target y split temporal | No |
| `features.py` | Variables de sesión y agregaciones del pasado | Sí |
| `modeling.py` | Pipeline y modelo | Sí |
| `train.py` y `evaluate.py` | Entrenamiento y métricas | No |
| `research.py` | Propuestas, commits, comparación y registro | No |
| `program.md` | Instrucciones del agente | No |

Train: enero–abril de 2017. Validation: mayo. Test: junio, fuera del loop.
El target es `transactions > 0`; transacciones vacías se interpretan como cero.

Cada fila recibe `history`, una lista de sesiones del mismo usuario de días
estrictamente anteriores, con `purchased` y `days_ago`. El agente elige tasas,
promedios, ventanas u otras agregaciones en `features.py`, preservando filas
y orden. Debe convertir esa lista en variables escalares y quitarla antes del
estimador. Las transformaciones aprendidas se ajustan únicamente en train.

Primero se miden las baselines constante y logística usando la sesión actual.
Después, el prompt pide investigar el aporte del historial. Cada candidato
se commitea y se evalúa en un intérprete nuevo. Sólo se conserva una mejora
estricta; empates, errores y timeouts vuelven al mejor commit.

Cada llamada a la API tiene hasta tres intentos para errores transitorios,
con `Retry-After` si el proveedor lo indica o esperas de 2 y 4 segundos.
También se reintenta un fallo de generación de JSON. Un request inválido no
se repite automáticamente con el mismo contenido. Si se agotan los intentos,
el error aparece en consola, se registra y el loop sigue con el próximo ensayo
desde el mejor commit. Los reintentos internos no consumen ensayos adicionales.

## Resultados

El loop crea una rama `autoresearch/<run_id>` y un worktree separado en
`experiments/autoresearch/<run_id>/`. Allí guarda:

- `history.jsonl`: hipótesis, SHA, métricas, duración y errores de cada intento.
- `summary.json`: mejor commit, score y directorio del mejor ensayo.
- `protocol.json`: configuración y hashes de los archivos fijos.
- `proposal_NNN.json` y `run_NNN/`: propuestas, logs y artefactos.
- `data/` y `worktree/`: datos congelados y código de investigación.

El checkout original no cambia. Las propuestas descartadas quedan guardadas,
aunque Git puede eliminar sus commits sin referencias al hacer garbage collection.
Para incorporar mejoras, revisar y hacer cherry-pick de los commits aceptados.

Los datos procesados y los artefactos se generan localmente y se ignoran en Git.
El CSV original se conserva en `data/raw/`. Una ejecución independiente de
`train.py` guarda sus resultados en `experiments/training/`.
