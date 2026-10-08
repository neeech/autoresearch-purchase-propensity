# Baselines simples

Entrenamiento: enero–abril. Validación: mayo. Test reservado. Logística con contexto, actividad e historia; imputación y codificación ajustadas sólo en train. Sin calibración ni búsqueda de configuraciones.

| Modelo | Log loss ↓ | Average Precision ↑ |
|---|---:|---:|
| constant | 0.08723 | 0.0171 |
| logistic | 0.03757 | 0.5020 |

Artefactos: `experiments/baselines/20261008T211349197683Z`.
