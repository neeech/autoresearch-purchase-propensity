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
uv run autoresearch --provider gemini --model gemini-3.8-flash --rpm 10 --experiments 10 --timeout 600
```

Con `--provider gemini`, el modelo predeterminado es `gemini-3.8-flash`, por lo
que `--model` es opcional. Usa la API nativa `generateContent` con respuesta JSON
estructurada y los mismos reintentos que Groq. Ninguna de las dos claves se pasa
al entrenamiento. Commiteá los cambios antes de iniciar una nueva ejecución.

`--rpm N` limita las llamadas a la API por minuto, incluidos los reintentos,
para cualquiera de los proveedores. Las llamadas se espacian al menos `60 / N`
segundos: `--rpm 10` deja 6 segundos entre inicios. La primera llamada es inmediata;
el tiempo de respuesta y entrenamiento cuenta como espera. `Retry-After` también
se respeta. Es un máximo, no una cantidad obligatoria; sin `--rpm` no se agrega
esta espera. El límite es local a esta ejecución y no controla tokens ni llamadas
de otros procesos que compartan la clave.

Sólo baselines, sin API:

```bash
uv run autoresearch --experiments 0
```

Propuestas guardadas, sin API:

```bash
uv run autoresearch --experiments 2 --proposals propuestas.json
```

El JSON contiene una lista de propuestas según [program.md](program.md).

## Continuar una investigación

Para agregar 10 intentos a una investigación existente en la misma máquina:

```bash
uv run autoresearch --resume <run_id> --experiments 10 --provider gemini --rpm 10 --timeout 600
```

`<run_id>` es el nombre de la carpeta en `experiments/autoresearch/`. Se reutilizan
los datos congelados, el mejor commit y su score, sin repetir las baselines.
El LLM recibe el historial acumulado y los nuevos ensayos continúan la numeración
sin sobrescribir resultados. Cada candidato se entrena desde cero.

Podés cambiar de proveedor o modelo con `--provider` y `--model`; cada ejecución
y ensayo registra cuál se usó. Las opciones de proveedor, modelo, RPM y timeout
se indican para cada ejecución; no se heredan de la anterior. También se puede
continuar con `--proposals` sin llamadas a una API.

Conservá la carpeta completa, incluido el worktree. La reanudación verifica los
hashes de los datos congelados y archivos protegidos, y que el checkout actual
mantenga el protocolo original. No permite dos ejecuciones simultáneas de la
misma investigación.

Los resultados se guardan después de cada ensayo. Si un intento quedó incompleto,
se registra como interrumpido y se continúa con un número nuevo desde el mejor
confirmado. `Ctrl+C` guarda el estado y detiene el loop. Si el proveedor identifica
explícitamente una cuota diaria agotada en una respuesta 429, también se detiene;
otros límites y fallos transitorios conservan los reintentos habituales.

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
- `executions/`: configuración y estado de cada ejecución, incluidas reanudaciones.
- `pending.json`: intento en curso, usado para recuperar interrupciones.

El checkout original no cambia. Las propuestas descartadas quedan guardadas,
aunque Git puede eliminar sus commits sin referencias al hacer garbage collection.
Para incorporar mejoras, revisar y hacer cherry-pick de los commits aceptados.

Los datos procesados y los artefactos se generan localmente y se ignoran en Git.
El CSV original se conserva en `data/raw/`. Una ejecución independiente de
`train.py` guarda sus resultados en `experiments/training/`.
