"""Reproducible descriptive audit and figures; no model selection."""
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='propensity-mpl-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_report(data, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    first_day = data.groupby('fullVisitorID').date.transform('min')
    audit = data.assign(month_label=data.date.dt.strftime('%Y-%m'),
                        prior_day_history=data.date.gt(first_day))
    monthly = audit.groupby('month_label').agg(
        sessions=('target', 'size'), visitors=('fullVisitorID', 'nunique'),
        purchases=('target', 'sum'), purchase_rate=('target', 'mean'),
        prior_day_history_rate=('prior_day_history', 'mean'))
    visitor_first_month = data.groupby('fullVisitorID').date.min().dt.strftime('%Y-%m')
    monthly['new_observed_visitors'] = visitor_first_month.value_counts().reindex(monthly.index, fill_value=0)
    monthly['new_observed_visitor_share'] = monthly.new_observed_visitors / monthly.visitors
    monthly.to_csv(output / 'monthly.csv')
    missing = data.isna().mean().sort_values(ascending=False)
    missing.rename('missing_fraction').to_csv(output / 'missingness.csv')
    numeric = data.select_dtypes('number').describe(percentiles=[.5, .9, .99]).T
    numeric.to_csv(output / 'numeric_summary.csv')
    categories = {}
    for col in ['source', 'channelGrouping', 'browser', 'deviceCategory', 'country', 'city']:
        categories[col] = int(data[col].nunique())
        data.groupby(col, dropna=False).target.agg(['size', 'sum', 'mean']).sort_values('size', ascending=False).to_csv(output / f'category_{col}.csv')
    raw_cols = [c for c in data if c not in ['session_row_id', 'target'] and not c.endswith('_missing')]
    audit_stats = {'rows': len(data), 'visitors': int(data.fullVisitorID.nunique()),
                   'positive_sessions': int(data.target.sum()), 'purchase_rate': float(data.target.mean()),
                   'duplicate_session_summaries': int(data[raw_cols].duplicated().sum()),
                   'revenue_without_purchase': int((data.transactionRevenue.fillna(0).gt(0) & data.target.eq(0)).sum()),
                   'purchase_without_revenue': int((data.target.eq(1) & data.transactionRevenue.isna()).sum()),
                   'same_day_multi_session_rows': int(data.duplicated(['fullVisitorID', 'date'], keep=False).sum()),
                   'categorical_cardinality': categories}
    (output / 'eda_summary.json').write_text(json.dumps(audit_stats, indent=2))
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .15, 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    monthly.sessions.plot.bar(ax=axes[0], color='#0072B2', rot=30)
    axes[0].set_title('Sesiones por mes')
    (monthly.purchase_rate * 100).plot(ax=axes[1], marker='o', color='#D55E00')
    axes[1].set_title('Sesiones con compra (%)')
    (monthly.new_observed_visitor_share * 100).plot.bar(ax=axes[2], color='#009E73', rot=30)
    axes[2].set_title('Visitantes nuevos observados (%)')
    for ax in axes:
        ax.set_xlabel('Mes')
    fig.savefig(figures / 'monthly.png', dpi=300)
    fig.savefig(figures / 'monthly.pdf')
    plt.close(fig)
    rows = '\n'.join(f'| {month} | {int(row.sessions):,} | {int(row.purchases):,} | {row.purchase_rate:.2%} | {row.new_observed_visitor_share:.2%} | {row.prior_day_history_rate:.2%} |' for month, row in monthly.iterrows())
    report = f'''# Análisis exploratorio de sesiones

Generado por `eda.py` a partir de los datos normalizados, antes de construir historia.

{len(data):,} sesiones, {data.fullVisitorID.nunique():,} visitantes y {data.target.sum():,} sesiones con compra ({data.target.mean():.2%}). Fechas: {data.date.min().date()} a {data.date.max().date()}.

## Volumen, compras e historia

| Mes | Sesiones | Con compra | Tasa de compra | Visitantes nuevos observados | Sesiones con historia de días anteriores |
|---|---:|---:|---:|---:|---:|
{rows}

Un visitante nuevo observado no necesariamente es nuevo para el sitio: no tenemos historia anterior al inicio del archivo. La proporción de visitantes nuevos y la proporción de sesiones sin historia tienen denominadores diferentes.

![Evolución mensual](figures/monthly.png)

## Calidad y decisiones

- El target es `transactions > 0`. Se asume explícitamente que un valor vacío significa cero compras; debe confirmarse con la fuente. Hay {audit_stats['revenue_without_purchase']} filas con ingresos positivos sin compra y {audit_stats['purchase_without_revenue']} compras sin ingresos registrados.
- `transactions` y `transactionRevenue` se excluyen del modelo. Los totales de actividad describen sesiones terminadas y pueden incluir consecuencias de la compra; el benchmark no demuestra anticipación online.
- `fullVisitorID` se conserva como texto para historia y auditoría, pero se excluye de las features.
- `timeOnSite` falta en {data.timeOnSite.isna().mean():.2%}; `bounces` en {data.bounces.isna().mean():.2%}. Duración conserva faltantes e indicador; bounces ya se normalizó a 0/1 según la definición proporcionada.
- Los textos `(not set)` y `not available in demo dataset` se normalizan como faltantes en categorías. La imputación y codificación se ajustarán sólo con entrenamiento.
- Hay {audit_stats['same_day_multi_session_rows']:,} filas de usuarios con múltiples sesiones el mismo día. La historia usa únicamente días anteriores, evitando inventar un orden intradía.
- Resúmenes de sesión duplicados: {audit_stats['duplicate_session_summaries']}. No se eliminan automáticamente: no hay un identificador único de sesión en la fuente. `session_row_id` identifica la fila original.
- Historia sin observaciones: contadores en cero, promedios/recencia desconocidos como faltantes y `has_history=0`.

## Evaluación propuesta

Entrenar enero–abril, validar mayo y reservar junio para test. La historia se actualiza secuencialmente con días anteriores, incluso dentro de validación/test, simulando disponibilidad de sesiones ya terminadas. Esto no ajusta parámetros del modelo con esos períodos.

Comparar baseline sin historia y candidato con historia. Usar log loss como objetivo y Average Precision, Brier score y calibración como métricas complementarias; reportar nuevos observados y recurrentes por separado. No seleccionar modelos ni features con resultados del test. Este EDA describe todo el período y no constituye un test estadístico completamente ciego.

Los CSV adjuntos contienen faltantes, distribuciones numéricas y tasas por categoría. Las asociaciones son descriptivas; no prueban causalidad. La tabla mensual permite revisar desbalance y cambios temporales antes de entrenar.
'''
    (output / 'eda.md').write_text(report)
