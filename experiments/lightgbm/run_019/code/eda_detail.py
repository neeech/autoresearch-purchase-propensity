"""Deep raw/normalized audit. All statistics descriptive, no model fitting."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .preprocess import load_sessions, CATEGORICAL
from .features import add_history
from .eda import generate_report, plt

NUMERIC = ['visitNumber', 'bounces', 'hits', 'pageviews', 'timeOnSite', 'transactions', 'transactionRevenue']
DESCRIPTIONS = {
 'fullVisitorID': ('Identificador de visitante observado; no garantiza una persona real.', 'Texto; sólo auditoría e historia.'),
 'visitNumber': ('Número de sesión del usuario; 1 indica primera sesión.', 'Contador; puede reflejar visitas anteriores al archivo.'),
 'date': ('Día de la sesión, YYYYMMDD; sin hora ni zona horaria declarada.', 'Ordenar por fecha; no inventar orden intradía.'),
 'bounces': ('Indicador de rebote: sólo 1 o vacío en este archivo.', 'Definición confirmada por el usuario: vacío=0, rebote=1.'),
 'hits': ('Acciones registradas, como visitas a páginas o clicks en botones.', 'Actividad agregada de sesión; probar log1p.'),
 'pageviews': ('Total de páginas vistas; puede contar vistas repetidas.', 'Preservar faltantes; probar log1p.'),
 'timeOnSite': ('Duración total registrada en segundos, confirmada por el usuario.', 'No mide necesariamente todo el tiempo real; conservar faltantes.'),
 'transactions': ('Cantidad de transacciones registradas en la sesión.', 'Sólo target; nunca feature ni indicador de faltante en X.'),
 'transactionRevenue': ('Ingreso total de transacciones; moneda y escala no declaradas. El CSV contiene decimales pese al tipo INTEGER del diccionario.', 'Excluir de X; no inferir moneda ni aplicar escala automáticamente.'),
 'source': ('Origen de tráfico informado, con nombres y dominios.', 'Categoría; no equivale directamente a channelGrouping.'),
 'channelGrouping': ('Agrupación del canal de adquisición.', 'Categoría de baja cardinalidad.'),
 'browser': ('Navegador informado.', 'Categoría con cola de valores raros.'),
 'deviceCategory': ('Tipo de dispositivo informado.', 'Categoría de baja cardinalidad.'),
 'country': ('País informado; posiblemente derivado de IP.', 'Categoría; geografía observada, no nacionalidad.'),
 'city': ('Ciudad informada, con valores ocultos/no definidos.', 'Preservar motivo de ausencia para estudiar su efecto.')}


class Report:
    def __init__(self):
        self.md, self.htm = [], []
    def heading(self, title, level=2):
        self.md.append('#' * level + ' ' + title)
        self.htm.append(f'<h{level}>{html.escape(title)}</h{level}>')
    def paragraph(self, text):
        self.md.append(text)
        self.htm.append('<p>' + html.escape(text) + '</p>')
    def table(self, frame):
        f = frame.copy().fillna('—')
        def fmt(v):
            if isinstance(v, (float, np.floating)):
                return f'{v:.4f}'
            return str(v).replace('|', '/')
        f = f.map(fmt)
        self.md.append('| ' + ' | '.join(f.columns) + ' |\n|' + '|'.join(['---'] * len(f.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in f.values))
        self.htm.append(f.to_html(index=False, escape=True, border=0))
    def image(self, path, title):
        self.md.append(f'![{title}]({path})')
        self.htm.append(f'<img src="{path}" alt="{html.escape(title)}">')
    def save(self, output):
        (output / 'eda.md').write_text('\n\n'.join(self.md) + '\n')
        css = 'body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#243345}h1,h2{color:#125879}h2{margin-top:48px}table{display:block;overflow-x:auto;border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dce3ea;text-align:left}th{background:#edf4f7}img{max-width:100%;height:auto}p{max-width:1000px}'
        (output / 'eda.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EDA de sesiones</title><style>' + css + '</style><body>' + '\n'.join(self.htm) + '</body></html>')


def rates(frame, by):
    t = frame.groupby(by, dropna=False, observed=True).target.agg(sessions='size', purchases='sum', purchase_rate='mean').reset_index()
    # Wilson 95% interval: descriptive iid approximation, users may be correlated.
    n, p, z = t.sessions, t.purchase_rate, 1.96
    center = (p + z*z/(2*n))/(1+z*z/n)
    width = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    t['rate_pct'] = 100*p
    t['ci95_low_pct'], t['ci95_high_pct'] = 100*(center-width), 100*(center+width)
    return t


def savefig(fig, output, name):
    for ext in ['png', 'pdf']:
        fig.savefig(output / 'figures' / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run(config):
    output = Path(config['report_dir'])
    raw = pd.read_csv(config['raw_path'], dtype={'fullVisitorID': 'string'}, low_memory=False)
    data = load_sessions(config['raw_path'], config['missing_transactions_mean_zero'])
    generate_report(data, output)
    r = Report()
    r.heading('Análisis exploratorio profundo: sesiones y compras', 1)
    r.paragraph(f'Fuente: {config["raw_path"]}. {len(raw):,} filas, {raw.shape[1]} columnas originales, {raw.fullVisitorID.nunique():,} visitantes observados. Período {data.date.min().date()} a {data.date.max().date()}. Target: transactions > 0; {int(data.target.sum()):,} sesiones positivas ({data.target.mean():.2%}). Se asume que transactions vacío implica ausencia de compra registrada.')
    r.paragraph('Este reporte describe todo el período, incluido junio. Las relaciones con el target para orientar baselines se calculan sólo en entrenamiento. No se entrenan modelos ni se eligen configuraciones con el test. No existe un test completamente ciego después de este EDA.')
    r.heading('1. Qué representa una fila y qué información falta')
    r.paragraph('Una fila representa un resumen de sesión, según la descripción del proyecto. No hay visitId, hora de inicio, eventos, URLs, carrito, productos ni orden de clicks. session_row_id identifica la posición en el CSV y no una sesión en el sistema de origen. El orden físico de filas no es un orden temporal fiable.')
    r.paragraph('Los nombres y patrones coinciden con un extracto aplanado de Universal Analytics. Es una inferencia, no una procedencia certificada. Su documentación define hits como registros de interacción, duración en segundos, bounces=1 para rebote y null para no rebote. El CSV podría haber sufrido transformaciones; transactionRevenue contiene decimales y su moneda/escala no puede reconstruirse sin el proceso de extracción.')
    r.md.append('Referencia semántica: [esquema oficial de Universal Analytics](https://support.google.com/analytics/answer/3437719?hl=en).')
    r.htm.append('<p>Referencia: <a href="https://support.google.com/analytics/answer/3437719?hl=en">Esquema oficial de Universal Analytics</a>.</p>')
    dictionary = []
    for c in raw:
        desc, action = DESCRIPTIONS[c]
        dictionary.append({'variable': c, 'tipo leído': str(raw[c].dtype), 'únicos observados': raw[c].nunique(),
                           'vacíos originales %': 100*raw[c].isna().mean(), 'ausentes normalizados %': 100*data[c].isna().mean(),
                           'interpretación': desc, 'tratamiento': action})
    dictionary = pd.DataFrame(dictionary)
    dictionary.to_csv(output / 'data_dictionary.csv', index=False)
    r.table(dictionary)
    r.paragraph('Los identificadores se leen como texto; los ceros iniciales son relevantes. Ninguna categoría identifica de forma fiable a una persona. Los campos numéricos describen agregados y no una secuencia de actividad.')

    r.heading('2. Faltantes: ausencia, semántica y ocultamiento')
    sentinels = []
    for c in CATEGORICAL:
        for value in ['(not set)', 'not available in demo dataset', '(direct)', '(none)', 'unknown', '']:
            count = int(raw[c].eq(value).sum())
            if count:
                sentinels.append({'variable': c, 'texto': value, 'filas': count, 'porcentaje': 100*count/len(raw),
                                  'acción actual': 'faltante' if value in ['(not set)', 'not available in demo dataset'] else 'se conserva'})
    sentinel_table = pd.DataFrame(sentinels)
    sentinel_table.to_csv(output / 'sentinels.csv', index=False)
    r.table(sentinel_table)
    r.paragraph('Los vacíos de transactions y revenue coinciden exactamente en este archivo. Su ausencia revela directamente el target y sus indicadores deben excluirse. bounces, con sólo 1 o vacío, no es una medición continua incompleta: si el origen es UA, el vacío es un cero semántico. La definición proporcionada confirma vacío=sin rebote; se codifica 0/1 sin indicador de faltante redundante. (direct) es una categoría de origen, no un faltante.')
    patterns = data.assign(bounce_state=np.where(data.bounces.eq(0), 'sin rebote (0)', 'rebote (1)'),
                           duration_state=np.where(data.timeOnSite.isna(), 'vacía', 'observada'),
                           hit_state=np.where(data.hits.eq(1), '1 hit', '>1 hit'))
    pattern_table = rates(patterns.loc[data.date.lt(pd.Timestamp(config['validation_start']))], ['bounce_state', 'duration_state', 'hit_state'])
    pattern_table.to_csv(output / 'missing_patterns.csv', index=False)
    r.paragraph('Patrones conjuntos de rebote, duración y hits, sólo en entrenamiento:')
    r.table(pattern_table[['bounce_state', 'duration_state', 'hit_state', 'sessions', 'purchases', 'rate_pct']])
    r.paragraph('En entrenamiento, todas las sesiones de un hit carecen de duración registrada. También hay sesiones de más de un hit sin duración: no conviene convertir indiscriminadamente la ausencia en cero segundos. Casi todas las compras tienen duración observada. Esto es señal descriptiva de medición/actividad, no prueba causal ni garantía de utilidad online.')
    missing_month = data.assign(month=data.date.dt.strftime('%Y-%m')).groupby('month')[['bounces','timeOnSite','pageviews','city','country']].agg(lambda x: 100*x.isna().mean())
    missing_month.to_csv(output / 'missingness_monthly.csv')
    r.table(missing_month.reset_index())
    city_reason = np.where(raw.city.eq('not available in demo dataset'), 'oculta en demo', np.where(raw.city.eq('(not set)'), 'no definida', np.where(raw.city.isna(), 'vacía', 'informada')))
    train_mask = data.date.lt(pd.Timestamp(config['validation_start']))
    train = data.loc[train_mask].copy()
    city_rates = rates(train.assign(city_status=city_reason[train_mask]), 'city_status')
    city_rates.to_csv(output / 'city_missingness_train.csv', index=False)
    r.table(city_rates[['city_status','sessions','purchases','rate_pct']])
    r.paragraph('La normalización actual combina ciudad oculta y ciudad no definida. Son mecanismos distintos: conservar su motivo original podría ayudar a auditar y probar una feature explícita. La disponibilidad de ciudad puede reflejar políticas del dataset demo, no comportamiento del usuario; probar una ablación sin city y sus indicadores.')

    r.heading('3. Duplicados, claves y consistencia')
    checks = []
    def check(name, mask, interpretation):
        checks.append({'control': name, 'filas afectadas': int(mask.sum()), 'interpretación': interpretation})
    check('Duplicado exacto de las 15 columnas originales (excedentes)', raw.duplicated(), 'Sin identificador de sesión no se deduplica automáticamente.')
    check('Repetición usuario + día (todas las filas del grupo)', data.duplicated(['fullVisitorID','date'], keep=False), 'Varias sesiones por día son legítimas; no es una clave.')
    check('Repetición usuario + visitNumber (excedentes)', data.duplicated(['fullVisitorID','visitNumber']), 'Investigar repetición entre fechas; no basta como clave.')
    check('Repetición usuario + día + visitNumber (excedentes)', data.duplicated(['fullVisitorID','date','visitNumber']), 'Unicidad observada no certifica una clave universal.')
    check('ID con ceros iniciales', raw.fullVisitorID.str.startswith('0'), 'Leer como texto para no alterar identidad.')
    check('ID no compuesto exclusivamente por dígitos', ~raw.fullVisitorID.str.fullmatch(r'\d+').fillna(False), 'Revisar anomalías de identificadores.')
    check('pageviews > hits', data.pageviews.gt(data.hits), 'Inconsistencia esperable si se cuentan en el mismo universo.')
    check('hits < 1 o visitNumber < 1', data.hits.lt(1) | data.visitNumber.lt(1), 'Valores incompatibles con sesiones observadas.')
    check('Compra con ingresos ausentes', data.target.eq(1) & data.transactionRevenue.isna(), 'Consistencia del outcome.')
    check('Ingreso positivo sin compra', data.transactionRevenue.fillna(0).gt(0) & data.target.eq(0), 'Consistencia del outcome.')
    check('Rebote con compra', data.bounces.eq(1) & data.target.eq(1), 'Relación determinista observada, no regla universal.')
    for c in NUMERIC:
        check(f'{c}: negativos o infinitos', data[c].lt(0) | np.isinf(data[c]), 'No se corrigen silenciosamente.')
    for c in ['visitNumber','hits','pageviews','transactions','timeOnSite','bounces']:
        check(f'{c}: valores no enteros', data[c].notna() & data[c].mod(1).ne(0), 'Contadores o resolución temporal; revisar fuente.')
    checks = pd.DataFrame(checks)
    checks.to_csv(output / 'quality_checks.csv', index=False)
    r.table(checks)
    day_numbers = data.groupby(['fullVisitorID','date']).visitNumber.agg(['min','max']).reset_index().sort_values(['fullVisitorID','date'])
    prior_max = day_numbers.groupby('fullVisitorID')['max'].cummax().groupby(day_numbers.fullVisitorID).shift()
    nonincreasing = day_numbers['min'].le(prior_max)
    r.paragraph(f'{int(nonincreasing.sum()):,} grupos usuario-día tienen visitNumber mínimo menor o igual a un máximo de días previos. No se usa ese contador para ordenar intradía. Una caída podría indicar reinicio, extracción parcial o semántica distinta; requiere fuente para decidir.')
    dates = pd.date_range(data.date.min(), data.date.max())
    gaps = dates.difference(pd.DatetimeIndex(data.date.unique()))
    r.paragraph(f'Cobertura diaria: {data.date.nunique()} días observados, {len(gaps)} fechas ausentes dentro del período. Que todas las fechas estén presentes no demuestra que la extracción incluya todas las sesiones.')
    r.paragraph('No se eliminan filas por pertenecer al mismo usuario, compartir fecha o coincidir en un resumen de actividad. No hay evidencia suficiente para distinguir eventos repetidos de sesiones reales más allá de estos controles.')

    r.heading('4. Distribuciones numéricas y extremos')
    dist = raw[NUMERIC].describe(percentiles=[.01,.25,.5,.75,.9,.95,.99,.999]).T.reset_index(names='variable')
    dist.to_csv(output / 'numeric_quantiles.csv', index=False)
    r.table(dist[['variable','count','min','50%','90%','99%','99.9%','max']])
    r.paragraph('Las distribuciones tienen colas largas: la media no representa la sesión típica. Los extremos no se etiquetan automáticamente como bots, errores o fraude. Probar log1p para contadores/duración; cualquier recorte de extremos debe aprender sus umbrales sólo con train y justificarse con una ablación. Revenue es descriptivo del outcome y no debe entrar en X.')
    fig, axes = plt.subplots(1, 3, figsize=(12,3.5), constrained_layout=True)
    for ax,c in zip(axes,['hits','pageviews','timeOnSite']):
        edges = np.linspace(np.log1p(train[c].dropna()).min(), np.log1p(train[c].dropna()).max(), 41)
        for label,color in [(0,'#0072B2'),(1,'#D55E00')]:
            values = train.loc[train.target.eq(label),c].dropna()
            ax.hist(np.log1p(values), bins=edges, density=True, alpha=.5, color=color, label=f'target={label}')
        ax.set_title(c + ' (train, observados)')
        ax.set_xlabel('log(1 + valor)'); ax.set_ylabel('Densidad'); ax.legend()
    savefig(fig,output,'activity_distributions')
    r.image('figures/activity_distributions.png','Distribuciones de actividad')
    by_target = train.groupby('target')[['visitNumber','hits','pageviews','timeOnSite']].agg(['count','median','mean'])
    by_target.to_csv(output / 'numeric_by_target_train.csv')
    for c,bins in [('hits',[0,1,2,5,10,20,50,np.inf]),('timeOnSite',[-1,30,120,600,1800,np.inf])]:
        grouped = train.assign(band=pd.cut(train[c],bins).astype('string').fillna('faltante'))
        table = rates(grouped,'band')
        table.to_csv(output / f'{c}_bands_train.csv',index=False)
        r.paragraph(f'Tasas por bandas de {c}, exclusivamente entrenamiento:')
        r.table(table[['band','sessions','purchases','rate_pct','ci95_low_pct','ci95_high_pct']])
    r.paragraph('Los intervalos Wilson al 95% son aproximaciones descriptivas por sesión; no corrigen dependencia entre sesiones de un usuario ni comparaciones múltiples. No usarlos como prueba de superioridad de una feature.')
    corr = train[['visitNumber','hits','pageviews','timeOnSite']].corr(method='spearman')
    corr.to_csv(output / 'spearman_train.csv')
    r.paragraph('Correlaciones de Spearman entre variables de actividad (pares con valores observados):')
    r.table(corr.reset_index(names='variable'))
    r.paragraph('Una correlación alta entre hits y pageviews puede representar redundancia. No se eliminan automáticamente: medir su aporte incremental con validación. Ratios entre ambos y duración por página pueden probarse después; controlar divisiones, faltantes y el alcance de sesión terminada.')

    r.heading('5. Categorías, concentración y valores nuevos')
    cat_summary = []
    val = data[data.date.ge(pd.Timestamp(config['validation_start'])) & data.date.lt(pd.Timestamp(config['test_start']))]
    test = data[data.date.ge(pd.Timestamp(config['test_start']))]
    for c in CATEGORICAL:
        counts = train[c].fillna('__MISSING__').value_counts()
        known = set(train[c].dropna())
        row = {'variable':c,'categorías train':train[c].nunique(),'categorías con <20 sesiones':int(counts.lt(20).sum()),
               'sesiones train en categorías <20 %':100*counts[counts.lt(20)].sum()/len(train),
               'categoría mayor %':100*counts.iloc[0]/len(train),
               'categorías nuevas validation':val.loc[val[c].notna() & ~val[c].isin(known),c].nunique(),
               'sesiones validation categoría nueva %':100*(val[c].notna() & ~val[c].isin(known)).mean(),
               'sesiones test categoría nueva %':100*(test[c].notna() & ~test[c].isin(known)).mean()}
        cat_summary.append(row)
        table = rates(train,c).sort_values('sessions',ascending=False)
        table.to_csv(output / f'category_{c}_train.csv',index=False)
        r.paragraph(f'{c}: las ocho categorías con mayor volumen en train. Tasas sin ajuste por país, dispositivo o actividad:')
        r.table(table.head(8)[[c,'sessions','purchases','rate_pct','ci95_low_pct','ci95_high_pct']])
    cats = pd.DataFrame(cat_summary)
    cats.to_csv(output / 'category_audit.csv',index=False)
    r.table(cats)
    r.paragraph('Agrupar categorías raras y resolver desconocidas usando sólo train. Evitar rankings de tasas con soporte diminuto. No hacer target encoding sobre todo el dataset; si se prueba, debe ser temporal y fuera de muestra en train. Las tasas marginales pueden reflejar mezcla geográfica y de adquisición.')
    cross = pd.crosstab(train.source,train.channelGrouping)
    cross.to_csv(output / 'source_channel_train.csv')
    direct = train.loc[train.source.eq('(direct)')].channelGrouping.value_counts().rename_axis('canal').reset_index(name='sesiones source=(direct)')
    r.table(direct)
    r.paragraph('source=(direct) aparece en varios canales. Esto no demuestra un error: source y channelGrouping pueden tener reglas de atribución diferentes. No reconstruir un canal desde source ni fusionar categorías sin consultar la extracción.')

    r.heading('6. Visitantes, recurrencia y cobertura de historia')
    counts = data.groupby('fullVisitorID').size()
    user_table = pd.DataFrame({'medida':['usuarios con una sola fila','usuarios con >1 fila','sesiones de usuarios con una sola fila','máximo de sesiones observadas por usuario'],
                               'valor':[int(counts.eq(1).sum()),int(counts.gt(1).sum()),int(counts.eq(1).sum()),int(counts.max())]})
    r.table(user_table)
    first = day_numbers.groupby('fullVisitorID', sort=False).head(1)
    r.paragraph(f'{first["min"].gt(1).mean():.2%} de los visitantes tienen visitNumber mínimo>1 en su primer día observado. Por lo tanto, sin historia en el archivo no equivale necesariamente a primera visita real. El ID puede cambiar entre dispositivos/cookies y no prueba identidad personal.')
    history = add_history(data)
    train_history = history[history.date.lt(pd.Timestamp(config['validation_start']))]
    history_rates = rates(train_history.assign(history_group=np.where(train_history.has_history.eq(1),'con días previos','sin días previos')), 'history_group')
    history_rates.to_csv(output / 'history_groups_train.csv',index=False)
    r.table(history_rates[['history_group','sessions','purchases','rate_pct','ci95_low_pct','ci95_high_pct']])
    history_cols = [c for c in history if c.startswith('history_') or c.startswith('days_since_') or c=='has_history']
    history[history_cols].describe(percentiles=[.5,.9,.99]).T.to_csv(output / 'history_summary.csv')
    counts.describe(percentiles=[.5,.9,.99]).to_csv(output / 'sessions_per_visitor.csv')
    r.table(pd.DataFrame([
        ['history_sessions', 'Número de sesiones en días anteriores', '0'],
        ['history_purchase_sessions', 'Sesiones con al menos una compra en días anteriores; no número total de transacciones', '0'],
        ['history_purchase_rate', 'Sesiones previas con compra / sesiones previas', 'faltante'],
        ['history_mean_hits', 'Suma de hits previos / sesiones previas', 'faltante'],
        ['history_mean_pageviews', 'Media de pageviews sobre sesiones previas con valor observado', 'faltante'],
        ['history_mean_time_on_site', 'Media de duración sobre sesiones previas con duración observada', 'faltante'],
        ['days_since_last_session', 'Días desde el último día previo con sesión', 'faltante'],
        ['days_since_last_purchase', 'Días desde el último día previo con sesión positiva', 'faltante'],
        ['has_history', 'Indicador de existencia de sesiones en días anteriores', '0'],
        ['day_of_week / month / is_weekend', 'Calendario de la sesión actual; lunes=0, domingo=6', 'no aplica']
    ], columns=['feature derivada', 'definición', 'sin historia']))
    r.paragraph('La historia se limita a días estrictamente anteriores. Excluye todas las sesiones del día actual aunque algunas puedan haber terminado antes. Es una aproximación conservadora por falta de timestamps. Compras previas sólo son features válidas si sus resultados ya están disponibles operacionalmente al calcular la historia.')
    r.paragraph('El historial atraviesa fronteras del split y se actualiza día a día; una fila de validación puede usar sesiones de días anteriores de validación, sin entrenar con ellas. Esta política simula disponibilidad secuencial. Comparar features con/sin historia y distinguir usuarios sin historia, recurrentes y usuarios conocidos desde train.')

    r.heading('7. Cambios temporales y diseño de evaluación')
    monthly = pd.read_csv(output / 'monthly.csv')
    r.table(monthly)
    r.image('figures/monthly.png','Volumen y evolución mensual')
    d = data.assign(month=data.date.dt.strftime('%Y-%m'))
    monthly_activity = d.groupby('month')[['visitNumber','hits','pageviews','timeOnSite']].median()
    monthly_activity.to_csv(output / 'monthly_activity.csv')
    r.table(monthly_activity.reset_index())
    drift=[]
    fig, axes=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
    for ax,c in zip(axes,['deviceCategory','channelGrouping']):
        shares=pd.crosstab(d.month,d[c],normalize='index')*100
        shares.to_csv(output / f'monthly_share_{c}.csv')
        shares.plot(ax=ax,marker='o'); ax.set_title(c); ax.set_ylabel('% de sesiones'); ax.legend(fontsize=7)
    savefig(fig,output,'category_mix')
    r.image('figures/category_mix.png','Mezcla temporal de categorías')
    for c in CATEGORICAL:
        train_dist=train[c].fillna('__MISSING__').value_counts(normalize=True)
        for name,part in [('validation',val),('test',test)]:
            other=part[c].fillna('__MISSING__').value_counts(normalize=True)
            union=train_dist.index.union(other.index)
            tv=.5*(train_dist.reindex(union,fill_value=0)-other.reindex(union,fill_value=0)).abs().sum()
            drift.append({'feature':c,'partición':name,'distancia variación total':tv})
    drift=pd.DataFrame(drift)
    drift.to_csv(output / 'categorical_drift.csv',index=False)
    r.table(drift)
    r.paragraph('La distancia de variación total compara proporciones de categorías con train: 0 significa igualdad y 1 distribuciones disjuntas. Es descriptiva, no un test de significancia ni evidencia automática de degradación. El cambio de tasa de compra exige evaluar calibración; una partición aleatoria mezclaría estos períodos.')
    parts={'train':train,'validation':val,'test':test}
    train_ids=set(train.fullVisitorID)
    overlap=[]
    for name,part in parts.items():
        overlap.append({'partición':name,'filas':len(part),'compras':int(part.target.sum()),'tasa compra %':100*part.target.mean(),
                        'visitantes':part.fullVisitorID.nunique(),'filas usuario visto en train %':100*part.fullVisitorID.isin(train_ids).mean()})
    r.table(pd.DataFrame(overlap))
    r.paragraph('Permitir visitantes repetidos entre períodos representa el uso temporal propuesto. Separar por usuario sería otro escenario: generalización exclusivamente a usuarios ajenos al entrenamiento. El ID no entra en el modelo; siempre informar resultados por presencia de historia y por usuario visto en train.')

    r.heading('8. Decisiones para los baselines y pendientes')
    actions = pd.DataFrame([
        ['Target','transactions > 0; vacío como cero','Confirmar contrato con fuente; los patrones internos lo respaldan, no lo certifican.'],
        ['Exclusiones','transactions, revenue, IDs y sus indicadores de ausencia','Evitar filtración directa del outcome.'],
        ['bounces','Indicador 0/1; null original se convierte a 0','Definición confirmada; sin imputación ni indicador redundante.'],
        ['Duración','Conservar vacío + indicador','No equiparar ausencia con cero real; imputar sólo dentro del pipeline train.'],
        ['Ciudad','Separar ocultamiento demo de no definida en futura variante','Ablación sin ciudad para medir dependencia del mecanismo de extracción.'],
        ['Contadores','Imputación aprendida en train; log1p como candidato','No recortar extremos sin prueba de error o comparación validada.'],
        ['Categorías','Codificación con manejo de desconocidos','Agrupación rara aprendida sólo en train.'],
        ['Historia','Sólo días anteriores; contadores cero y medias/recencias faltantes sin historia','Comparar baseline sin historia vs con historia.'],
        ['Evaluación','Train enero–abril; validation mayo; test junio','Optimizar log loss y reportar AP, Brier, calibración y segmentos.'],
        ['Alcance','Clasificación de sesión completa','La actividad y rebote pueden reflejar consecuencias de compra; no anticipación online.'],
        ['Procedencia','Solicitar SQL/exportación, moneda, timezone y esquema de sesión','No inventar la semántica de lo que el CSV no especifica.']
    ],columns=['tema','decisión propuesta','motivo / pendiente'])
    r.table(actions)
    r.paragraph('Orden de baselines: (1) probabilidad constante aprendida en train; (2) logística con contexto y actividad; (3) misma logística con historia. Añadir una ablación sólo contexto para entender cuánto depende el resultado de la actividad agregada. Las hipótesis salen del EDA de train; las decisiones se toman en validation y el test se ejecuta al cierre.')
    r.heading('9. Reproducibilidad y archivos adjuntos')
    r.paragraph('Ejecutar desde la raíz: PYTHONPATH=src python -m autoresearch_purchase_propensity.eda_detail --config configs/data.json. Se regeneran eda.md, eda.html, figuras PNG/PDF y tablas CSV; no se modifican datos raw ni particiones Parquet. El procesamiento también ejecuta este reporte antes de construir las particiones.')
    r.paragraph('Tablas: data_dictionary.csv, quality_checks.csv, sentinels.csv, missing_patterns.csv, missingness_monthly.csv, numeric_quantiles.csv, numeric_by_target_train.csv, *_bands_train.csv, category_*_train.csv, category_audit.csv, source_channel_train.csv, history_groups_train.csv, history_summary.csv, categorical_drift.csv y monthly*.csv. Los CSV contienen detalles que exceden las tablas resumidas.')
    compact_report(data, config, output, history_rates, overlap)
    audit_manifest={'raw_sha256':hashlib.sha256(Path(config['raw_path']).read_bytes()).hexdigest(), 'config':config,
                    'association_scope':'train only', 'descriptive_scope':'all periods',
                    'versions':{'pandas':pd.__version__,'numpy':np.__version__},'checks':checks.to_dict(orient='records')}
    (output / 'eda_manifest.json').write_text(json.dumps(audit_manifest,indent=2))
    print(f'Reporte compacto generado en {output / "eda.html"}')


def compact_figures(data, output, history_rates):
    """Two complementary summaries, with explicit populations and denominators."""
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    columns = ['city', 'bounces', 'timeOnSite', 'country', 'source', 'pageviews', 'browser']
    missing = (data[columns].isna().mean() * 100).sort_values()
    axes[0].barh(missing.index, missing.values, color='#0072B2')
    axes[0].set_xlim(0, 75)
    axes[0].set_xlabel('% de sesiones con valor ausente')
    axes[0].set_title('Faltantes normalizados · todo el período')
    for y, value in enumerate(missing.values):
        axes[0].text(value + .8, y, f'{value:.2f}%', va='center', fontsize=8)
    h = history_rates.set_index('history_group').reindex(['sin días previos', 'con días previos'])
    bars = axes[1].bar(['Sin historia', 'Con historia'], h.rate_pct, color=['#0072B2', '#D55E00'], width=.55)
    axes[1].set_ylabel('% de sesiones con compra')
    axes[1].set_ylim(0, h.rate_pct.max() * 1.35)
    axes[1].set_title('Compra según historia · sólo train')
    for bar, (_, row) in zip(bars, h.iterrows()):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + .1,
                     f'{row.rate_pct:.2f}%\nn={int(row.sessions):,}', ha='center', fontsize=9)
    savefig(fig, output, 'missingness_history')


def compact_report(data, config, output, history_rates, overlap):
    """Keep the main report actionable; retain CSVs as supporting evidence."""
    r = Report()
    r.heading('Dataset de sesiones: hallazgos y decisiones', 1)
    r.paragraph(f'{len(data):,} sesiones completas, {data.fullVisitorID.nunique():,} visitantes y 15 variables originales. Período: {data.date.min().date()} a {data.date.max().date()}. Hay {int(data.target.sum()):,} sesiones con compra ({data.target.mean():.2%}). El objetivo actual es clasificar sesiones terminadas; los agregados pueden incluir consecuencias de la compra y no permiten validar anticipación online.')
    r.heading('Variables y tratamiento')
    r.table(pd.DataFrame([
        ['Identidad y fecha', 'fullVisitorID, date, visitNumber', 'ID como texto para historia; no como predictor. Fecha para split y calendario. visitNumber es un contador, no una clave.'],
        ['Actividad', 'hits, pageviews, timeOnSite, bounces', 'Totales de sesión. Conservar faltantes e indicadores; probar log1p en contadores y duración. hits cuenta acciones, incluyendo páginas y clicks; duración en segundos.'],
        ['Contexto', 'source, channelGrouping, browser, deviceCategory, country, city', 'Categorías; codificar y resolver valores nuevos usando sólo entrenamiento.'],
        ['Resultado', 'transactions, transactionRevenue', 'Target: transactions > 0. Excluir ambas columnas y sus indicadores de ausencia de las features.']
    ], columns=['grupo', 'variables', 'tratamiento']))
    compact_figures(data, output, history_rates)
    r.heading('Calidad: lo esencial')
    sentinels = pd.read_csv(output / 'sentinels.csv')
    def city_count(value):
        return int(sentinels.loc[sentinels.variable.eq('city') & sentinels.texto.eq(value), 'filas'].sum())
    raw_checks = pd.read_csv(output / 'quality_checks.csv').set_index('control')['filas afectadas']
    r.paragraph(f'No hay duplicados exactos, valores numéricos negativos/infinitos ni inconsistencias detectadas entre compras e ingresos. Hay {int(raw_checks.loc["Repetición usuario + visitNumber (excedentes)"]):,} repeticiones de usuario + visitNumber entre fechas y {int(raw_checks.loc["Repetición usuario + día (todas las filas del grupo)"]):,} filas en grupos usuario-día con varias sesiones. No deduplicar por usuario ni ordenar sesiones del mismo día sin timestamps.')
    r.table(pd.DataFrame([
        ['timeOnSite', f'{data.timeOnSite.isna().mean():.2%} vacío', 'No asumir cero segundos. Conservar indicador y ajustar imputación sólo con train.'],
        ['bounces', f'{data.bounces.eq(1).mean():.2%} con rebote; resto sin rebote', 'Definición confirmada: null original se codifica como 0; no es un faltante desconocido.'],
        ['pageviews', f'{int(data.pageviews.isna().sum())} vacíos', 'Conservar e imputar dentro del pipeline.'],
        ['city', f'{city_count("not available in demo dataset"):,} ocultas en demo; {city_count("(not set)"):,} no definidas', 'Son motivos distintos; el procesamiento actual los combina. Probar una variante sin ciudad.'],
        ['transactions / revenue', 'Ambas vacías en las mismas sesiones', 'Se asume que transactions vacío significa cero compras registradas; confirmar con la fuente.']
    ], columns=['variable', 'hallazgo', 'decisión']))
    r.paragraph('El diccionario proporcionado confirma las definiciones de las variables, incluida duración en segundos y rebote null=0. Revenue contiene decimales en el CSV pese a declararse INTEGER; no se fuerza a entero. Su moneda/escala y la zona horaria siguen sin documentarse. No se eliminan extremos automáticamente: las variables de actividad tienen colas largas.')
    r.md.append('Referencia semántica: [esquema oficial de Universal Analytics](https://support.google.com/analytics/answer/3437719?hl=en).')
    r.htm.append('<p>Referencia: <a href="https://support.google.com/analytics/answer/3437719?hl=en">Esquema oficial de Universal Analytics</a>.</p>')
    r.image('figures/missingness_history.png', 'Faltantes normalizados e historia en entrenamiento')
    r.paragraph('Izquierda: ausencias después de normalizar categorías, sobre todo el período. Derecha: tasas de compra en train; cada barra indica el volumen de sesiones. bounces ya está normalizado a 0/1: los null originales significan sin rebote.')
    r.heading('Historia y señales para investigar')
    monthly = pd.read_csv(output / 'monthly.csv')
    new_share = monthly.loc[monthly.month_label.ne(monthly.month_label.min()), 'new_observed_visitor_share']
    r.paragraph(f'Entre los meses posteriores al primero, {new_share.min():.1%}–{new_share.max():.1%} de los visitantes son nuevos en el archivo. Sin historia observada no equivale a primera visita real. El modelo debe funcionar bien para usuarios sin antecedentes disponibles.')
    h = history_rates[['history_group', 'sessions', 'purchases', 'rate_pct']].copy()
    h.columns = ['historia en entrenamiento', 'sesiones', 'compras', 'tasa de compra %']
    r.table(h)
    r.paragraph('La diferencia es una asociación descriptiva, no causal. La historia incluye número de sesiones y sesiones con compra, tasa de compra, medias de actividad y días desde última sesión/compra. Usa sólo días anteriores; sin historia, contadores=0 y medias/recencias=faltantes. Comparar modelos con y sin historia.')
    r.paragraph('Las sesiones con compra muestran más actividad y casi siempre duración observada. Esto puede reflejar tanto intención como efectos posteriores a la compra. La disponibilidad de ciudad también se asocia con el resultado y podría reflejar el mecanismo del dataset demo.')
    r.image('figures/activity_distributions.png', 'Actividad por resultado de compra, sólo entrenamiento')
    r.paragraph('Las distribuciones se normalizan por separado para comparar formas pese al desbalance. El eje usa log(1 + valor); duración incluye sólo valores observados. La separación no demuestra anticipación de compras.')
    r.heading('Evaluación y próximos baselines')
    split = pd.DataFrame(overlap)[['partición', 'filas', 'compras', 'tasa compra %']]
    r.table(split)
    r.paragraph(f'Validación desde {config["validation_start"]}; test desde {config["test_start"]}; fechas anteriores para entrenamiento. Cambian la tasa de compra y la mezcla de fuentes/canales: mantener el split temporal. La historia se actualiza día a día, también dentro de validación/test, sin ajustar el modelo con esas filas.')
    r.paragraph('Baselines: probabilidad constante aprendida en train; regresión logística con contexto y actividad; misma logística con historia. Añadir comparaciones sólo contexto y sin ciudad para entender de dónde viene la señal. Optimizar log loss y acompañar con Average Precision, Brier y calibración, globales y según presencia de historia. Reservar resultados de modelos en test para el cierre.')
    r.image('figures/monthly.png', 'Volumen, tasa de compra y visitantes nuevos por mes')
    r.paragraph('El primer mes muestra 100% de visitantes nuevos por inicio del registro; no implica que todos visitaran el sitio por primera vez.')
    r.heading('Referencia y reproducción')
    r.paragraph('Las asociaciones con compra se calcularon sólo en train; los controles de calidad y cobertura describen todo el período. El test ya fue observado descriptivamente. Los CSV conservan el detalle para consultas puntuales.')
    for label, path in [('Diccionario completo', 'data_dictionary.csv'), ('Controles de calidad', 'quality_checks.csv'), ('Distribuciones numéricas', 'numeric_quantiles.csv'), ('Categorías y desconocidas', 'category_audit.csv'), ('Evolución mensual', 'monthly.csv')]:
        r.md.append(f'[{label}]({path})')
        r.htm.append(f'<p><a href="{path}">{label}</a></p>')
    r.paragraph('Reproducir desde la raíz: PYTHONPATH=src python -m autoresearch_purchase_propensity.eda_detail --config configs/data.json. No modifica el CSV ni los Parquet.')
    r.save(output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/data.json')
    args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()))


if __name__=='__main__':
    main()
