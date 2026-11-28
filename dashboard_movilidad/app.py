import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from pyarrow import ArrowException

from data_model import (
    FIELDS, MIN_CELL, PUBLIC, PUBLIC_FILTERS, ROOT, class_counts, correlations,
    counts, cross_counts, load_trips,
)
from prediction import CLASS_LABELS, prediction_page

ASSETS = ROOT / "assets"
CLASSES = CLASS_LABELS
COLORS = {"Colectivo": "#384252", "Individual motorizado": "#e95813"}
PAGES = [
    "Resumen ejecutivo", "Modelo de datos", "Análisis exploratorio",
    "Evaluación de modelos", "Hallazgos para DiDi", "Predicción interactiva",
]
# Tabla de MLflow guardada en la sección 6.1 del notebook Entrega_3.ipynb.
METRICS = pd.DataFrame(
    {"Regresión logística": [0.8266, 0.7833, 0.8356, 0.8086, 0.9009, 0.8902],
     "Random Forest": [0.8453, 0.8464, 0.7906, 0.8176, 0.9137, 0.9030]},
    index=["Accuracy", "Precisión", "Recall", "F1", "AUC-ROC", "AUC-ROC ponderado"],
)


@st.cache_data(show_spinner=False)
def read_public():
    return (
        pd.read_parquet(PUBLIC / "distribuciones.parquet"),
        pd.read_csv(PUBLIC / "correlacion.csv", index_col=0),
        pd.read_parquet(PUBLIC / "cruce_estrato_auto.parquet"),
    )


read_private = st.cache_data(show_spinner=False)(load_trips)


def number(value):
    return f"{value:,}".replace(",", ".")


def percent(value):
    return f"{value:.1%}".replace(".", ",")


def show_asset(name, caption):
    file = ASSETS / name
    if file.is_file():
        st.image(str(file), caption=caption, width="stretch")
    else:
        st.info(f"Falta el archivo {name}.")


def show_chart(figure, height=390, filename="movilidad-bogota"):
    figure.update_layout(
        height=height, margin=dict(l=12, r=20, t=24, b=28),
        legend=dict(title=None, orientation="h", y=-0.2),
        hoverlabel=dict(namelength=-1), font=dict(color="#18202c"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", config={
        "displaylogo": False, "displayModeBar": True,
        "toImageButtonOptions": {"format": "svg", "filename": filename},
    })


def split_chart(table, field, proportion):
    if field == "ingreso_grupo":
        table = table.loc[~table.categoria.str.contains("No reporta|Categorías agrupadas", case=False)]
    if table.empty:
        st.info("La selección no tiene categorías visibles para esta gráfica.")
        return
    table = table.assign(Transporte=table.es_individual.map(CLASSES))
    table["proporcion"] = table.viajes / table.groupby("categoria").viajes.transform("sum")
    axis = "proporcion" if proportion else "viajes"
    figure = px.bar(
        table, x=axis, y="categoria", color="Transporte", barmode="stack",
        color_discrete_map=COLORS, category_orders={"Transporte": list(CLASSES.values())},
        labels={"categoria": "", "viajes": "Viajes", "proporcion": "Proporción"},
        hover_data={"viajes": ":,d", "proporcion": ":.1%"},
    )
    figure.update_yaxes(autorange="reversed", categoryorder="category descending")
    if proportion:
        figure.update_xaxes(tickformat=".0%", range=[0, 1])
    show_chart(figure, max(350, 29 * table.categoria.nunique() + 120), field)


def summary(totals, income, baseline, auto):
    total = totals.sum()
    share = totals[1] / total
    difference = share - baseline
    for card, name, value in zip(
        st.columns(4),
        ("Viajes analizados", "Colectivo", "Individual motorizado", "AUC-ROC · Random Forest"),
        (number(total), percent(totals.get(0, 0) / total),
         percent(totals.get(1, 0) / total), "0,9137"),
    ):
        delta = points(difference) if name == "Individual motorizado" else None
        card.metric(name, value, delta=delta, delta_color="off", border=True)
    st.caption("Variación de participación frente a la base completa · "
               "AUC-ROC fijo del conjunto de prueba de la Entrega 3.")
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Distribución de viajes")
        data = totals.rename(index=CLASSES).rename_axis("Transporte").reset_index(name="Viajes")
        figure = px.pie(data, names="Transporte", values="Viajes", color="Transporte",
                        hole=0.66, color_discrete_map=COLORS)
        figure.update_traces(textposition="inside", textinfo="percent")
        show_chart(figure, 365)
        st.caption("Conteos sin ponderar de la base analítica de la Encuesta de Movilidad 2023.")
    with right:
        st.subheader("¿Qué segmentos vale la pena explorar?")
        by_auto = auto.pivot(index="categoria", columns="es_individual", values="viajes")
        if {"No", "Sí"}.issubset(by_auto.index):
            by_auto = by_auto.reindex(columns=[0, 1]).fillna(0)
            shares = by_auto[1].div(by_auto.sum(axis=1))
            auto_gap = shares["Sí"] - shares["No"]
            st.markdown(f"**Tenencia de auto.** En la selección, el {percent(shares['Sí'])} de los viajes "
                        f"de hogares con auto es individual, frente al {percent(shares['No'])} de hogares "
                        f"sin auto ({points(auto_gap)}). Compara ambos grupos por estrato antes de definir un piloto.")
        if income is not None:
            by_income = income.pivot(index="categoria", columns="es_individual", values="viajes")
            by_income = by_income.reindex(columns=[0, 1]).fillna(0)
        else:
            by_income = pd.DataFrame()
        if {"1. Hasta $1,16 M", "4. Más de $4,9 M"}.issubset(by_income.index):
            share = by_income[1].div(by_income.sum(axis=1))
            low, high = share.loc[["1. Hasta $1,16 M", "4. Más de $4,9 M"]]
            st.markdown(f"**Ingreso.** El transporte individual representa el {percent(low)} de "
                        f"los viajes en el tramo de menor ingreso y el {percent(high)} en el mayor.")
        st.markdown(
            "**Contexto del viaje.** Explorar motivos, franjas y localidades ayuda a formular "
            "hipótesis para campañas y cobertura.\n\n"
            "**Alcance.** El resultado distingue dos tipos de transporte. La demanda de DiDi "
            "requiere datos comerciales propios."
        )


def restrict(frame, selections):
    for field, labels in selections.items():
        if labels:
            frame = frame[frame[f"label_{field}"].isin(labels)]
    return frame


def filter_sidebar(source, fields, local):
    prefix = "local_filter" if local else "public_filter"
    keys = {field: f"{prefix}_{field}" for field in fields}

    def clear():
        for key in keys.values():
            st.session_state[key] = []

    with st.sidebar:
        st.subheader("Filtros globales")
        st.button("Restablecer filtros", on_click=clear, icon=":material/restart_alt:")
        return {field: st.multiselect(FIELDS[field], sorted(source[f"label_{field}"].unique()),
                                      key=keys[field], placeholder="Todas las categorías")
                for field in fields}


def points(value):
    return f"{value * 100:+.1f} pp".replace(".", ",")


def profile_counts(frame, field):
    if "viajes" not in frame:
        return counts(frame, field)
    column = f"label_{field}"
    return (frame.groupby([column, "es_individual"], as_index=False)
            .viajes.sum().rename(columns={column: "categoria"}))


def distributions(frame, table, fields):
    field = st.selectbox("Variable", fields, format_func=FIELDS.get, key="distribution_field")
    if table is None or field in PUBLIC_FILTERS:
        series = profile_counts(frame, field)
    else:
        series = table.loc[table.variable.eq(field)]
    selected = st.multiselect("Categorías visibles", sorted(series.categoria.unique()),
                              key=f"visible_{field}", placeholder="Todas las categorías")
    if selected:
        series = series[series.categoria.isin(selected)]
    if field == "ingreso_grupo":
        st.caption("«No reporta» se conserva en la base y se oculta en la gráfica de ingreso.")
    scale = st.radio("Escala", ("Viajes", "Proporción por categoría"), horizontal=True)
    split_chart(series, field, scale == "Proporción por categoría")


def segment_comparison(frame, fields):
    st.caption("Cada segmento se define dentro de la selección global. Los segmentos pueden solaparse.")
    if "viajes" in frame:
        st.caption("La vista pública compara segmentos preagregados de estrato y tenencia de automóvil.")
    segments = {}
    for pane, name in zip(st.columns(2), ("Segmento A", "Segmento B")):
        with pane:
            st.subheader(name)
            conditions = {}
            if "viajes" in frame:
                dimensions = fields
            else:
                dimensions = st.multiselect("Variables del segmento", fields, format_func=FIELDS.get,
                                            key=f"fields_{name}", default=["tiene_auto"])
            for field in dimensions:
                options = sorted(frame[f"label_{field}"].unique())
                conditions[field] = st.multiselect(FIELDS[field], options, key=f"{name}_{field}",
                                                    placeholder="Todas las categorías")
            segments[name] = class_counts(restrict(frame, conditions))
    totals = pd.DataFrame(segments).T
    volumes = totals.sum(axis=1)
    cols = st.columns(3)
    for col, name in zip(cols, totals.index):
        col.metric(f"Viajes · {name}", number(volumes[name]), border=True)
    if volumes.eq(0).any():
        st.info("Se requieren viajes en ambos segmentos para calcular la diferencia.")
        return
    shares = totals.div(volumes, axis=0)[1]
    cols[2].metric("Diferencia B − A", points(shares.iloc[1] - shares.iloc[0]), border=True)
    shares = shares.rename("Individual motorizado").rename_axis("Segmento").reset_index()
    chart = px.bar(shares, x="Individual motorizado", y="Segmento", orientation="h",
                   text_auto=".1%", color_discrete_sequence=[COLORS[CLASSES[1]]])
    chart.update_traces(hovertemplate="%{y}<br>Individual motorizado: %{x:.1%}<extra></extra>")
    chart.update_xaxes(range=[0, 1], tickformat=".0%")
    show_chart(chart, 330, "comparacion-segmentos")


def cross_chart(frame, fields):
    if "viajes" in frame:
        first, second = PUBLIC_FILTERS
        st.caption("Cruce público disponible entre estrato y tenencia de automóvil.")
    else:
        left, right = st.columns(2)
        first = left.selectbox("Variable de filas", fields, format_func=FIELDS.get)
        second = right.selectbox("Variable de columnas", [f for f in fields if f != first],
                                 format_func=FIELDS.get)
    x, y = f"label_{second}", f"label_{first}"
    data = cross_counts(frame, first, second)
    if "ingreso_grupo" in (first, second):
        data = data.loc[~data["label_ingreso_grupo"].str.contains("No reporta", case=False)]
        st.caption("La gráfica omite «No reporta».")
    pivot = data.pivot(index=[y, x], columns="es_individual", values="viajes")
    if pivot.empty:
        st.info("La selección no tiene categorías visibles para este cruce.")
        return
    pivot = pivot.reindex(columns=[0, 1]).fillna(0)
    volume = pivot.sum(axis=1)
    share = pivot[1].div(volume).unstack()
    chart = px.imshow(share, text_auto=".1%", aspect="auto", zmin=0, zmax=1,
                      color_continuous_scale=["#fff3e9", "#b6440b"],
                      labels={"x": FIELDS[second], "y": FIELDS[first], "color": "% individual"})
    chart.update_xaxes(tickangle=-35 if share.shape[1] > 5 else 0, automargin=True)
    chart.update_yaxes(automargin=True)
    chart.update_coloraxes(colorbar_tickformat=".0%")
    chart.update_traces(customdata=volume.unstack().reindex_like(share).to_numpy(),
                        hovertemplate="%{y} · %{x}<br>Individual: %{z:.1%}<br>Viajes: %{customdata:,}<extra></extra>")
    show_chart(chart, max(380, 28 * len(share) + 140), f"cruce-{first}-{second}")


def exploration(frame, table, matrix, filtered):
    local = "viajes" not in frame
    fields = list(FIELDS if local or not filtered else PUBLIC_FILTERS)
    tabs = st.tabs(["Perfiles", "Comparar segmentos", "Cruzar variables", "Correlación"])
    with tabs[0]:
        st.subheader("¿Cómo varía la elección de transporte?")
        if not local and filtered:
            st.caption("Con filtros globales activos se muestran las variables del cruce publicado.")
        distributions(frame, table, fields)
    with tabs[1]:
        segment_comparison(frame, list(FIELDS if local else PUBLIC_FILTERS))
    with tabs[2]:
        cross_chart(frame, fields)
    with tabs[3]:
        correlation_chart(frame if local else None, matrix)
    with st.expander("Figuras académicas de referencia · Entrega 3"):
        st.caption("Figuras originales de la entrega, con la selección de datos del notebook.")
        show_asset("edad.png", "Distribución de edad por tipo de transporte")
        show_asset("franja_motivo.png", "Franja de salida y motivo")
        show_asset("correlacion.png", "Correlación de las variables codificadas en entrenamiento")


def correlation_chart(trips, matrix):
    st.subheader("Matriz de correlación")
    st.caption("Pearson sobre variables numéricas y binarias de "
               + ("la selección local." if trips is not None else "la base completa en la vista pública."))
    if trips is not None and len(trips) < MIN_CELL:
        st.info(f"Se requieren al menos {MIN_CELL} viajes para calcular la correlación.")
        return
    if trips is not None:
        matrix = correlations(trips)
    show_chart(px.imshow(matrix, zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                         color_continuous_scale=["#384252", "#f4f5f7", "#ec6321"]), 525,
               "correlacion-movilidad")


def evaluation():
    st.caption("Conjunto de prueba de la Entrega 3 · clase positiva: individual motorizado.")
    auc, precision, recall = st.columns(3)
    auc.metric("AUC-ROC · Random Forest", "0,9137", border=True)
    precision.metric("Precisión · Random Forest", "84,64 %", border=True)
    recall.metric("Recall · Regresión logística", "83,56 %", border=True)
    st.dataframe(METRICS.style.format("{:.4f}"), width="stretch")
    st.markdown("Se seleccionó **Random Forest** por AUC-ROC. También obtuvo mayor precisión "
                "y F1, mientras que la regresión logística obtuvo mayor recall al umbral de 0,50.")
    st.caption("Métricas de la sección 6.1 y figuras exportadas de MLflow. "
               "Solo la fila marcada como ponderada usa el factor de expansión.")
    artifacts = (
        ("Matrices de confusión", "Figura_1_Matrices_confusion.png"),
        ("Curvas ROC", "Figura_2_Curvas_ROC.png"),
        ("Curvas precisión-recall", "Figura_3_Precision_Recall.png"),
    )
    for tab, (label, name) in zip(st.tabs([label for label, _ in artifacts]), artifacts):
        with tab:
            show_asset(name, label + " · MLflow")
            file = ASSETS / name
            if file.is_file():
                st.download_button("Descargar figura original", file.read_bytes(), name, "image/png",
                                   key=f"download_{name}", icon=":material/download:")
    st.markdown("**Decisión comercial.** La mayor precisión de Random Forest reduce la proporción "
                "de falsos positivos entre los viajes que clasifica como individuales. Si interesa recuperar "
                "más de esos viajes, la regresión logística ofrece mayor recall al umbral de 0,50. "
                "La curva precisión-recall permite evaluar ese intercambio antes de elegir un umbral para un piloto.")


def findings():
    chart, notes = st.columns([1.35, 1])
    with chart:
        show_asset("importancia_rf.png", "Importancia por reducción de impureza · Entrega 3 / MLflow")
    with notes:
        st.markdown("""
1. **Tenencia de automóvil.** Comparar motivos y franjas de viaje entre hogares con y sin auto.
2. **Licencia de motocicleta.** Explorar si los perfiles con licencia presentan patrones distintos por motivo.
3. **Licencia de automóvil.** Cruzarla con tenencia en el análisis interno para distinguir acceso y habilitación.
4. **Tenencia de motocicleta.** Comparar perfiles por franja y localidad para orientar pruebas segmentadas.
5. **Número de etapas.** Estudiar trayectos con transbordos como hipótesis para servicios de conexión.
""")
    st.caption("El ranking mide contribución predictiva. Las decisiones comerciales propuestas son hipótesis. "
               "Para usar el número de etapas en una predicción operativa, hay que confirmar que esté disponible antes del viaje.")
    st.markdown("Para un piloto, compara los segmentos de tenencia por motivo y franja. "
                "Valida las hipótesis con solicitudes, conversión y costo de adquisición de DiDi, "
                "usando un grupo de comparación para medir el efecto de la campaña.")
    with st.expander("Modelo de referencia · regresión logística"):
        show_asset("importancia_rl.png", "Coeficientes de regresión logística · Entrega 3 / MLflow")


def main():
    st.set_page_config(page_title="Movilidad Bogotá · DiDi", page_icon="🚕", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 1480px; padding-top: 2.5rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {background: #fff; border-radius: 14px; border-top: 3px solid #ef641f;
                                  box-shadow: 0 4px 18px #18202c0a;}
        [data-testid="stSidebar"] {border-right: 1px solid #e7e8ed;}
        [data-testid="stHeading"] h1 {letter-spacing: -.04em;}
        .st-key-executive_header {background: #fbeee5; padding: 1rem 1.3rem 0.65rem; border-radius: 16px;}
        .source {font-size: .78rem; color: #a13b10; font-weight: 750; letter-spacing: .08em;}
        </style>""", unsafe_allow_html=True)
    private_dir = os.environ.get("MOVILIDAD_PARQUET_DIR")
    with st.sidebar:
        st.title("Movilidad Bogotá", anchor=False)
        st.caption("Encuesta de Movilidad 2023")
        page = st.radio("Navegación", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Parquet locales" if private_dir else "Datos agregados · vista pública")
    with st.container(key="executive_header"):
        st.markdown('<div class="source">BOGOTÁ 2023 · ESTRATEGIA COMERCIAL DIDI</div>',
                    unsafe_allow_html=True)
        st.title("¿Quién elige el transporte individual?" if page == PAGES[0] else page)
    data_error = None
    try:
        if private_dir:
            source, table, matrix = read_private(Path(private_dir)), None, None
            fields = list(FIELDS)
        else:
            table, matrix, source = read_public()
            fields = list(PUBLIC_FILTERS)
        filters = filter_sidebar(source, fields, bool(private_dir))
        frame = restrict(source, filters)
        filtered = any(filters.values())
        if not private_dir:
            st.sidebar.caption(f"Filtros públicos por estrato y automóvil. "
                               f"Cada celda del cruce contiene al menos {MIN_CELL} viajes.")
    except (OSError, ValueError, duckdb.Error, ArrowException) as exc:
        data_error = str(exc)
    if page in (PAGES[0], PAGES[2]):
        if data_error:
            st.error(f"No se pudieron cargar los datos. {data_error}")
            return
        totals = class_counts(frame)
        if not totals.sum():
            st.info("La combinación de filtros no contiene viajes. Ajusta o restablece los filtros.")
            return
        if page == PAGES[0]:
            baseline = class_counts(source)
            income = (counts(frame, "ingreso_grupo") if private_dir
                      else None if filtered else table[table.variable.eq("ingreso_grupo")])
            summary(totals, income, baseline[1] / baseline.sum(), profile_counts(frame, "tiene_auto"))
        else:
            st.caption(f"{number(totals.sum())} viajes · {percent(totals[1] / totals.sum())} individuales"
                       " · Conteos sin ponderar · Descarga las figuras con el icono de cámara.")
            exploration(frame, table, matrix, filtered)
    elif page == PAGES[1]:
        st.subheader("Esquema de copo de nieve · DuckDB")
        show_asset("esquema_copo.png", "Diagrama del proyecto académico")
        st.markdown("**Fact_Viaje** contiene un registro por viaje. Se une a **Dim_Persona**, "
                    "**Dim_Modo**, **Dim_Motivo** y **Dim_Franja** mediante sus claves "
                    "sustitutas. **Dim_Persona** se une a **Dim_Hogar** por `sk_hogar`. "
                    "Cada dimensión puede describir varios registros de la tabla relacionada.")
    elif page == PAGES[3]:
        evaluation()
    elif page == PAGES[4]:
        findings()
    else:
        prediction_page()


if __name__ == "__main__":
    main()
