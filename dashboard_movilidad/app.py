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
    "Evaluación de modelos", "Oportunidades comerciales para DiDi", "Predicción interactiva",
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


def download_table(frame, filename, label="Descargar resumen CSV"):
    """Descarga un agregado ya publicado, sin exponer registros individuales."""
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), filename,
                       "text/csv", icon=":material/download:")


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
    total = class_counts(frame)
    with st.container(border=True):
        cards = st.columns(3)
        cards[0].metric("Viajes en la seleccion", number(total.sum()))
        cards[1].metric("Individual motorizado", percent(total.get(1, 0) / total.sum()))
        cards[2].metric("Alcance de la evidencia", "Encuesta, no demanda DiDi")
        st.caption("Estos indicadores describen la muestra seleccionada. No son usuarios, solicitudes ni conversiones de DiDi.")
    tabs = st.tabs(["Perfiles", "Comparar segmentos", "Cruzar variables", "Correlación"])
    with tabs[0]:
        st.subheader("¿Cómo varía la elección de transporte?")
        if not local and filtered:
            st.caption("Con filtros globales activos se muestran las variables del cruce publicado.")
        distributions(frame, table, fields)
        if table is not None:
            public_summary = table.assign(Transporte=table.es_individual.map(CLASSES))
            download_table(public_summary, "resumen_distribucion_transporte.csv")
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


def legacy_findings():
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


def share_table(table, field):
    """Calcula participaciones desde agregados publicados, no desde filas individuales."""
    data = table.loc[table.variable.eq(field)].copy()
    pivot = data.pivot_table(index="categoria", columns="es_individual", values="viajes", aggfunc="sum")
    pivot = pivot.reindex(columns=[0, 1], fill_value=0).fillna(0)
    pivot["total"] = pivot.sum(axis=1)
    pivot["share_individual"] = pivot[1].div(pivot["total"].where(pivot["total"].ne(0)))
    return pivot.reset_index()


def best_category(table, field, contains=None):
    data = share_table(table, field)
    if contains:
        data = data[data.categoria.astype(str).str.contains(contains, case=False, na=False, regex=True)]
    data = data.dropna(subset=["share_individual"])
    return None if data.empty else data.sort_values("share_individual", ascending=False).iloc[0]


def opportunity_card(title, question, evidence, action, validation):
    with st.container(border=True):
        st.subheader(title)
        st.caption(question)
        evidence_col, action_col, validation_col = st.columns(3)
        with evidence_col:
            st.markdown("**Hallazgo observado**")
            st.write(evidence)
        with action_col:
            st.markdown("**Posible accion de DiDi**")
            st.write(action)
        with validation_col:
            st.markdown("**Como validar**")
            st.write(validation)


def commercial_opportunities(frame, table, matrix):
    st.info("El transporte individual motorizado incluye automovil particular, motocicleta, taxi y plataformas. "
            "Esta pagina identifica patrones de movilidad, pero no estima usuarios ni demanda de DiDi.")
    st.subheader("¿Qué puede hacer DiDi con esta herramienta?")
    value_cards = st.columns(3)
    value_cards[0].metric("1 · Entender", "Mercado de movilidad")
    value_cards[0].caption("Leer patrones de Bogota mas alla de los usuarios actuales de la plataforma.")
    value_cards[1].metric("2 · Comparar", "Segmentos")
    value_cards[1].caption("Priorizar perfiles para investigaciones o pilotos comerciales.")
    value_cards[2].metric("3 · Complementar", "Datos propios")
    value_cards[2].caption("Cruzar la encuesta con demanda, tarifas, disponibilidad y campanas de DiDi.")
    if table is None:
        table = pd.concat([counts(frame, field).assign(variable=field) for field in FIELDS], ignore_index=True)
    explorer_table = table
    public_total = table.loc[table.variable.eq(PUBLIC_FILTERS[0]), "viajes"].sum()
    if "viajes" in frame and frame.viajes.sum() < public_total:
        explorer_table = pd.concat(
            [profile_counts(frame, field).assign(variable=field) for field in PUBLIC_FILTERS],
            ignore_index=True,
        )
        st.caption("Con filtros publicos activos, el explorador se limita a estrato y tenencia de automovil, que son los agregados disponibles para esa seleccion.")
    available = set(explorer_table.variable)
    st.subheader("Explorador de oportunidades")
    st.caption("Explora categorias agregadas de la encuesta. Las comparaciones publicas respetan el umbral de "
               f"{MIN_CELL} viajes por celda y no permiten reconstruir registros individuales.")
    selectable = [field for field in FIELDS if field in available]
    field = st.selectbox("Pregunta comercial", selectable, format_func=FIELDS.get, key="opportunity_field")
    data = share_table(explorer_table, field)
    visible = st.multiselect("Categorias a comparar", data.categoria.tolist(),
                             default=data.categoria.tolist()[:6], key=f"opportunity_categories_{field}")
    selected = data[data.categoria.isin(visible)].copy()
    if selected.empty:
        st.info("Selecciona al menos una categoria para comparar.")
    else:
        selected["Colectivo"] = selected[0].div(selected["total"])
        selected["Individual motorizado"] = selected["share_individual"]
        chart_data = selected.melt(id_vars="categoria", value_vars=["Colectivo", "Individual motorizado"],
                                   var_name="Transporte", value_name="Proporcion")
        chart = px.bar(chart_data, x="Proporcion", y="categoria", color="Transporte", barmode="stack",
                       orientation="h", text_auto=".1%", color_discrete_map=COLORS,
                       labels={"categoria": "", "Proporcion": "Participacion"})
        chart.update_xaxes(range=[0, 1], tickformat=".0%")
        chart.update_yaxes(autorange="reversed")
        show_chart(chart, max(360, 30 * len(selected) + 130), "oportunidades-comerciales")
        download_table(selected[["categoria", 0, 1, "total", "share_individual"]],
                       f"oportunidades_{field}.csv")
    st.caption("Participacion dentro de cada categoria de la encuesta. No equivale a probabilidad de usar DiDi.")

    st.subheader("Hipótesis respaldadas por la evidencia disponible")
    auto = best_category(table, "tiene_auto", "^No$")
    auto_yes = best_category(table, "tiene_auto", "^S")
    motive = best_category(table, "motivo_grupo")
    time = best_category(table, "franja")
    locality = best_category(table, "localidad")
    cards = st.tabs(["Sin auto", "Multiples etapas", "Motivos", "Franja", "Localidad"])
    with cards[0]:
        if auto is not None and auto_yes is not None:
            auto_chart = share_table(table, "tiene_auto")
            auto_chart["Colectivo"] = auto_chart[0].div(auto_chart["total"])
            auto_chart["Individual motorizado"] = auto_chart["share_individual"]
            chart_data = auto_chart.melt(id_vars="categoria", value_vars=["Colectivo", "Individual motorizado"],
                                         var_name="Transporte", value_name="Proporcion")
            chart = px.bar(chart_data, x="categoria", y="Proporcion", color="Transporte", barmode="group",
                           text_auto=".1%", color_discrete_map=COLORS,
                           labels={"categoria": "Tenencia de automovil", "Proporcion": "Participacion"})
            chart.update_yaxes(range=[0, 1], tickformat=".0%")
            show_chart(chart, 340, "hogares-con-sin-auto")
            opportunity_card(
                "Movilidad individual ocasional en hogares sin automovil",
                "\u00bfQue viajes individuales realizan hogares sin auto y en que contextos podria requerirse una alternativa?",
                f"En la encuesta, los viajes individuales representan {percent(auto.share_individual)} en hogares sin auto, frente a {percent(auto_yes.share_individual)} en hogares con auto.",
                "Investigar uso ocasional por segmento, sin asumir que el transporte individual observado ya es demanda de DiDi.",
                "Medir solicitudes nuevas y recurrentes por tenencia de auto, origen-destino, hora, motivo y tarifa; comparar con hogares con auto.")
        else:
            st.info("No hay suficiente agregado publico para esta oportunidad.")
    with cards[1]:
        corr = matrix.at["Etapas", "Viaje individual"] if matrix is not None and "Etapas" in matrix.index else None
        evidence = (f"La correlacion publicada entre numero de etapas y viaje individual es {corr:.2f}; la asociacion es descriptiva y no prueba causalidad." if corr is not None else "El numero de etapas no esta publicado como distribucion publica; solo puede revisarse internamente.")
        if corr is not None:
            st.metric("Correlacion publicada con viaje individual", f"{corr:.2f}", border=True)
            stage_chart = px.bar(pd.DataFrame({"Relacion": ["Etapas · individual"], "Correlacion": [corr]}),
                                 x="Correlacion", y="Relacion", orientation="h",
                                 color_discrete_sequence=[COLORS["Individual motorizado"]],
                                 labels={"Correlacion": "Pearson", "Relacion": ""})
            stage_chart.update_xaxes(range=[-1, 1], zeroline=True, zerolinewidth=2, zerolinecolor="#384252")
            show_chart(stage_chart, 260, "etapas-correlacion")
        opportunity_card("Viajes colectivos con multiples etapas",
                         "\u00bfDonde aparecen viajes con varias etapas y que conexion podria complementar esa necesidad?",
                         evidence,
                         "Explorar conexiones o ultima milla donde los datos de ruta y disponibilidad indiquen friccion.",
                         "Usar datos de rutas, tiempos, solicitudes y abandono para medir si una oferta de conexion reduce friccion o aumenta viajes completados.")
    with cards[2]:
        if motive is not None:
            st.metric("Mayor participacion individual observada", percent(motive.share_individual), border=True)
            st.caption(f"Categoria destacada: {motive.categoria}")
            opportunity_card(f"Necesidad asociada al motivo: {motive.categoria}",
                             "\u00bfQue motivos concentran mayor participacion individual y en que momentos conviene investigar el servicio?",
                             f"Es la categoria de motivo con mayor participacion individual observada: {percent(motive.share_individual)}.",
                             "Priorizar mensajes o pilotos exploratorios por motivo, sin convertir el indicador en intencion de compra.",
                             "Contrastar motivo declarado con busquedas, solicitudes, conversion, hora y distancia en una prueba controlada.")
    with cards[3]:
        if time is not None:
            st.metric("Mayor participacion individual observada", percent(time.share_individual), border=True)
            st.caption(f"Franja destacada: {time.categoria}")
            opportunity_card(f"Ventana horaria: {time.categoria}",
                             "\u00bfQue ventanas horarias muestran mayor movilidad individual para orientar cobertura?",
                             f"Es la franja con mayor participacion individual observada: {percent(time.share_individual)}.",
                             "Evaluar cobertura o comunicacion en esa franja, sin asumir que toda la movilidad individual es direccionable a DiDi.",
                             "Comparar disponibilidad, ETA, tarifa, conversion y cancelacion por franja antes de definir una campana.")
    with cards[4]:
        if locality is not None:
            st.metric("Mayor participacion individual observada", percent(locality.share_individual), border=True)
            st.caption(f"Localidad destacada: {locality.categoria}")
            opportunity_card(f"Localidad para exploracion: {locality.categoria}",
                             "\u00bfQue localidades ameritan una investigacion comercial prioritaria?",
                             f"Es la localidad agregada con mayor participacion individual observada: {percent(locality.share_individual)}.",
                             "Priorizar una prueba de cobertura o investigacion zonal, no estimar tamano de mercado.",
                             "Validar cobertura, origen-destino, densidad de solicitudes y costo de adquisicion dentro de la localidad.")

    st.subheader("Información adicional que DiDi necesitaría")
    st.dataframe(pd.DataFrame([
        ["Demanda real", "Solicitudes, viajes completados, conversion y frecuencia", "Comprobar si el patron de encuesta se convierte en uso."],
        ["Contexto del viaje", "Origen-destino, distancia, duracion, motivo y hora", "Separar necesidad ocasional, commuting y ultima milla."],
        ["Economia", "Tarifa, promociones, costo de adquisicion y margen", "Medir viabilidad comercial, no solo prevalencia."],
        ["Experimento", "Grupo de comparacion y exposicion a la campana", "Estimar efecto incremental de una accion comercial."],
    ], columns=["Pregunta", "Dato requerido", "Para que sirve"]), hide_index=True, width="stretch")

    st.subheader("De inteligencia de movilidad a inteligencia comercial integrada")
    st.caption("Propuesta futura: no hay conexion con datos de DiDi ni indicadores comerciales simulados en esta version.")
    future = st.columns(4)
    future[0].markdown("### 1\nEncuesta Bogotá 2023")
    future[0].caption("Patrones de movilidad y segmentos")
    future[1].markdown("### +\nSolicitudes DiDi")
    future[1].caption("Demanda efectiva y viajes completados")
    future[2].markdown("### +\nTarifas y disponibilidad")
    future[2].caption("Oferta, ETA y condiciones de servicio")
    future[3].markdown("### →\nCampanas evaluadas")
    future[3].caption("Incrementalidad, conversion y margen")
    st.success("Evolucion objetivo: pasar de comprender patrones a evaluar demanda y acciones comerciales con evidencia propia.")

    st.subheader("Ranking predictivo: Random Forest")
    st.divider()
    st.caption("La importancia indica contribucion predictiva dentro del modelo entrenado. No es causalidad, elasticidad ni ranking de clientes.")
    ranking = pd.DataFrame([
        [1, "Tenencia de automovil", "Acceso al vehiculo"],
        [2, "Licencia de motocicleta", "Habilitacion para conducir moto"],
        [3, "Licencia de automovil", "Habilitacion para conducir auto"],
        [4, "Tenencia de motocicleta", "Acceso a moto"],
        [5, "Numero de etapas", "Complejidad del trayecto"],
    ], columns=["Posicion", "Caracteristica", "Lectura comercial prudente"])
    st.dataframe(ranking, hide_index=True, width="stretch")
    chart, notes = st.columns([1.25, 1])
    with chart:
        show_asset("importancia_rf.png", "Importancia por reduccion de impureza · Entrega 3 / MLflow")
    with notes:
        st.markdown("**Como usarlo.** Prioriza preguntas para segmentar y validar; no conviertas una variable importante en una causa ni en una audiencia de DiDi.")
        st.markdown("**Limite.** El modelo clasifica transporte colectivo frente a individual motorizado; no identifica taxi, auto particular, moto o plataforma por separado.")
    with st.expander("Figuras academicas de referencia · Entrega 3"):
        show_asset("importancia_rl.png", "Coeficientes de regresion logistica · Entrega 3 / MLflow")


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
        if page in (PAGES[0], PAGES[2], PAGES[4]):
            st.caption("Los filtros globales actualizan indicadores y descriptivos compatibles.")
        else:
            st.caption("Resultados fijos del entrenamiento o entradas independientes.")
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
        commercial_opportunities(frame, table, matrix)
    else:
        prediction_page()


if __name__ == "__main__":
    main()
