import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from pyarrow import ArrowException

from data_model import FIELDS, MIN_CELL, PUBLIC, ROOT, correlations, counts, load_trips

ASSETS = ROOT / "assets"
CLASSES = {0: "Colectivo", 1: "Individual motorizado"}
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


def show_chart(figure, height=390):
    figure.update_layout(
        height=height, margin=dict(l=12, r=20, t=24, b=12),
        legend=dict(title=None, orientation="h", y=-0.2),
        hoverlabel=dict(namelength=-1),
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


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
    show_chart(figure, max(350, 29 * table.categoria.nunique() + 120))


def summary(totals, income):
    total = totals.sum()
    for card, name, value in zip(
        st.columns(4),
        ("Viajes analizados", "Colectivo", "Individual motorizado", "AUC-ROC · Random Forest"),
        (number(total), percent(totals.get(0, 0) / total),
         percent(totals.get(1, 0) / total), "0,9137"),
    ):
        card.metric(name, value, border=True)
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
        st.subheader("Oportunidades comerciales")
        by_income = income.pivot(index="categoria", columns="es_individual", values="viajes")
        if {"1. Hasta $1,16 M", "4. Más de $4,9 M"}.issubset(by_income.index):
            share = by_income[1].div(by_income.sum(axis=1))
            low, high = share.loc[["1. Hasta $1,16 M", "4. Más de $4,9 M"]]
            st.markdown(f"**Ingreso.** El transporte individual representa el {percent(low)} de "
                        f"los viajes en el tramo de menor ingreso y el {percent(high)} en el mayor.")
        st.markdown(
            "**Acceso a vehículos.** La tenencia de automóvil encabeza el ranking de Random Forest. "
            "Cruzar tenencia y licencias en el análisis interno permite comparar perfiles de movilidad.\n\n"
            "**Contexto del viaje.** Explorar motivos, franjas y localidades ayuda a formular "
            "hipótesis para campañas y cobertura.\n\n"
            "**Alcance.** El resultado distingue dos tipos de transporte. La demanda de DiDi "
            "requiere datos comerciales propios."
        )


def exploration(trips, table, matrix):
    with st.sidebar:
        st.subheader("Filtros de exploración")
        if trips is not None:
            source = trips
            for field, label in FIELDS.items():
                column = f"label_{field}"
                options = sorted(source[column].unique())
                selected = st.multiselect(label, options, key=field, placeholder="Todas las categorías")
                if selected:
                    trips = trips[trips[column].isin(selected)]
    fields = list(FIELDS if trips is not None else [f for f in FIELDS if f in table.variable.values])
    field = st.selectbox("Variable", fields, format_func=FIELDS.get)
    series = counts(trips, field) if trips is not None else table.loc[table.variable.eq(field)]
    if trips is None:
        with st.sidebar:
            selected = st.multiselect("Categorías visibles", sorted(series.categoria.unique()),
                                      key=f"public_{field}", placeholder="Todas las categorías")
        if selected:
            series = series[series.categoria.isin(selected)]
        st.caption(f"Vista pública con una variable por vez. Se agrupan categorías pequeñas "
                   f"para publicar al menos {MIN_CELL} viajes por celda.")
    else:
        st.caption(f"{number(len(trips))} viajes en la selección local. Conteos sin ponderar.")
    if field == "ingreso_grupo":
        st.caption("«No reporta» se conserva en la base y se oculta en la gráfica de ingreso.")
    scale = st.radio("Escala", ("Viajes", "Proporción por categoría"), horizontal=True)
    split_chart(series, field, scale == "Proporción por categoría")
    st.subheader("Matriz de correlación")
    st.caption("Pearson sobre variables numéricas y binarias de "
               + ("la selección local." if trips is not None else "la base completa en la vista pública."))
    if trips is not None and len(trips) < MIN_CELL:
        st.info(f"Se requieren al menos {MIN_CELL} viajes para calcular la correlación.")
        return
    if trips is not None:
        matrix = correlations(trips)
    show_chart(px.imshow(matrix, zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                         color_continuous_scale=["#384252", "#f4f5f7", "#ec6321"]), 525)


def evaluation():
    st.caption("Conjunto de prueba de la Entrega 3 · clase positiva: individual motorizado.")
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


def main():
    st.set_page_config(page_title="Movilidad Bogotá · DiDi", page_icon="🚕", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 1340px; padding-top: 2.5rem;}
        [data-testid="stMetric"] {background: #fff; border-radius: 14px; border-top: 3px solid #ef641f;}
        [data-testid="stSidebar"] {border-right: 1px solid #e7e8ed;}
        [data-testid="stHeading"] h1 {letter-spacing: -.035em;}
        .source {font-size: .78rem; color: #a13b10; font-weight: 750; letter-spacing: .08em;}
        </style>""", unsafe_allow_html=True)
    private_dir = os.environ.get("MOVILIDAD_PARQUET_DIR")
    with st.sidebar:
        st.title("Movilidad Bogotá")
        st.caption("Encuesta de Movilidad 2023")
        page = st.radio("Navegación", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Parquet locales" if private_dir else "Datos agregados · vista pública")
    st.markdown('<div class="source">BOGOTÁ 2023 · ESTRATEGIA COMERCIAL DIDI</div>',
                unsafe_allow_html=True)
    st.title("¿Quién elige el transporte individual?" if page == PAGES[0] else page)
    if page in (PAGES[0], PAGES[2]):
        try:
            trips = read_private(Path(private_dir)) if private_dir else None
            table, matrix = read_public() if trips is None else (None, None)
        except (OSError, ValueError, duckdb.Error, ArrowException) as exc:
            st.error(f"No se pudieron cargar los datos. {exc}")
            return
        if page == PAGES[0]:
            totals = (trips.groupby("es_individual").size() if trips is not None
                      else table[table.variable.eq("sexo")].groupby("es_individual").viajes.sum())
            income = (counts(trips, "ingreso_grupo") if trips is not None
                      else table[table.variable.eq("ingreso_grupo")])
            summary(totals, income)
        else:
            exploration(trips, table, matrix)
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
        st.info("Pendiente de exportación del pipeline entrenado.")
        st.markdown("La corrida de Random Forest guardó en MLflow el pipeline con el "
                    "preprocesador ajustado y el clasificador. Para habilitar inferencia hay que "
                    "recuperar ese pipeline, su entorno y la firma de entrada con las 17 variables. "
                    "La exportación pública debe excluir los ejemplos de prueba registrados en MLflow.")


if __name__ == "__main__":
    main()
