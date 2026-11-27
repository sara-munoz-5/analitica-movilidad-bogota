"""Contract for the fitted Random Forest in Entrega_3.ipynb, section 5."""

import json
import math
import os
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import streamlit as st

CATEGORIES = {
    "estrato_grupo": ("Estrato", ("Bajo (1-2)", "Medio (3-4)", "Alto (5-6)")),
    "ingreso_grupo": ("Ingreso", (
        "1. Hasta $1,16 M", "2. $1,16 M - $2,5 M", "3. $2,5 M - $4,9 M", "4. Más de $4,9 M",
        "5. No reporta",
    )),
    "localidad": ("Localidad", (
        "Antonio Nariño", "Barrios Unidos", "Bosa", "Chapinero", "Ciudad Bolívar", "Engativá",
        "Fontibón", "Kennedy", "La Candelaria", "Los Mártires", "Puente Aranda",
        "Rafael Uribe Uribe", "San Cristóbal", "Santa Fe", "Suba", "Sumapaz",
        "Teusaquillo", "Tunjuelito", "Usaquén", "Usme",
    )),
    "rango_edad": ("Rango de edad", ("05-17", "18-25", "26-35", "36-45", "46-60", "61+")),
    "sexo": ("Sexo", ("Hombre", "Mujer", "Intersexual")),
    "nivel_educativo": ("Nivel educativo", (
        "1. Ninguno o primaria", "2. Secundaria o media", "3. Técnico o tecnológico",
        "4. Universitario", "5. Posgrado",
    )),
    "ocupacion": ("Ocupación", (
        "Empleado", "Estudiante", "Hogar", "Independiente o informal",
        "Pensionado o rentista", "Sin ocupación u otra",
    )),
    "motivo_grupo": ("Motivo", (
        "Acompañar o llevar", "Compras", "Estudio", "Ocio, social o religioso",
        "Otro", "Regreso al hogar", "Salud o trámites", "Trabajo",
    )),
    "franja": ("Franja horaria", (
        "1. Madrugada (0-5 h)", "2. Pico AM (5-9 h)", "3. Valle (9-16 h)",
        "4. Pico PM (16-20 h)", "5. Noche (20-24 h)",
    )),
}
BINARY = {
    "tiene_auto": "Tenencia de automóvil", "tiene_moto": "Tenencia de motocicleta",
    "tiene_bici": "Tenencia de bicicleta", "licencia_auto": "Licencia de automóvil",
    "licencia_moto": "Licencia de motocicleta", "celular_datos": "Celular con datos",
}
FEATURES = (*CATEGORIES, "edad", "num_etapas", *BINARY)
ARTIFACT = "random_forest.joblib"
MANIFEST = "entorno.json"
RUNTIME_PACKAGES = ("scikit-learn", "numpy", "scipy", "pandas", "joblib")


def age_band(age):
    for maximum, band in zip((17, 25, 35, 45, 60, 100), CATEGORIES["rango_edad"][1]):
        if age <= maximum:
            return band
    raise ValueError("Edad fuera del rango de la encuesta.")


@st.cache_resource(show_spinner=False)
def load_pipeline(directory, artifact_revision, manifest_revision):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    expected = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    mismatches = [f"{pkg}: instalada {version(pkg)}, exportada {expected.get(pkg)}"
                  for pkg in RUNTIME_PACKAGES if expected.get(pkg) != version(pkg)]
    python = platform.python_version()
    if str(expected.get("python")).split(".")[:2] != python.split(".")[:2]:
        mismatches.append(f"Python: instalado {python}, exportado {expected.get('python')}")
    if mismatches:
        raise ValueError("Versiones incompatibles. " + " | ".join(mismatches))
    pipeline = joblib.load(directory / ARTIFACT)
    if (not isinstance(pipeline, Pipeline)
            or not isinstance(pipeline.named_steps.get("modelo"), RandomForestClassifier)
            or tuple(pipeline.feature_names_in_) != FEATURES or set(pipeline.classes_) != {0, 1}):
        raise ValueError("El artefacto no tiene el pipeline, las 17 variables y las clases de Entrega 3.")
    prep = pipeline.named_steps["prep"]
    cat_columns = next(columns for name, _, columns in prep.transformers_ if name == "cat")
    if tuple(cat_columns) != tuple(CATEGORIES):
        raise ValueError("Las variables categóricas difieren de Entrega 3.")
    categories = {field: tuple(values) for field, values in
                  zip(cat_columns, prep.named_transformers_["cat"].categories_, strict=True)}
    return pipeline, categories


def infer(pipeline, values, categories):
    if set(values) != set(FEATURES) or any(values[field] is None for field in FEATURES):
        raise ValueError("Completa todos los campos antes de calcular.")
    if (not 5 <= values["edad"] <= 100 or not 1 <= values["num_etapas"] <= 6
            or any(values[field] % 1 for field in ("edad", "num_etapas"))):
        raise ValueError("Ingresa una edad de 5 a 100 años y entre 1 y 6 etapas.")
    if values["rango_edad"] != age_band(values["edad"]):
        raise ValueError("El rango de edad debe corresponder a la edad.")
    if any(values[field] not in allowed for field, allowed in categories.items()):
        raise ValueError("Alguna categoría no pertenece al vocabulario del pipeline.")
    if any(values[field] not in (0, 1) for field in BINARY):
        raise ValueError("Las variables de acceso solo admiten Sí o No.")
    row = pd.DataFrame([values], columns=FEATURES).astype({"edad": float, "num_etapas": float})
    probabilities = dict(zip(pipeline.classes_, pipeline.predict_proba(row)[0], strict=True))
    individual = float(probabilities[1])
    if not math.isfinite(individual) or not 0 <= individual <= 1:
        raise ValueError("El pipeline devolvió una probabilidad inválida.")
    selected = int(individual >= 0.5)
    return selected, float(probabilities[selected]), individual


def prediction_page():
    st.subheader("¿Qué tipo de transporte estima el modelo?")
    st.caption("La probabilidad corresponde al tipo de transporte. El número de etapas "
               "debe conocerse para interpretar el resultado como una estimación previa al viaje.")
    pipeline = None
    options = {field: values for field, (_, values) in CATEGORIES.items()}
    configured = os.environ.get("MOVILIDAD_MODELO_DIR")
    if not configured:
        st.info("Formulario disponible. Para calcular, configura MOVILIDAD_MODELO_DIR "
                "con random_forest.joblib y entorno.json exportados desde Colab.")
    else:
        directory = Path(configured)
        try:
            pipeline, options = load_pipeline(directory, (directory / ARTIFACT).stat().st_mtime_ns,
                                              (directory / MANIFEST).stat().st_mtime_ns)
        except Exception as exc:
            st.error(f"No se pudo cargar el pipeline de Random Forest. {exc}")
    age = st.number_input("Edad", min_value=5, max_value=100, value=None, step=1)
    values = {"edad": age, "rango_edad": age_band(age) if age is not None else None}
    with st.form("rf_prediction"):
        profile, trip, access = st.columns(3)
        with profile:
            st.markdown("**Perfil**")
            st.text_input("Rango de edad calculado", value=values["rango_edad"] or "", disabled=True)
            for field in ("estrato_grupo", "ingreso_grupo", "localidad", "sexo", "nivel_educativo", "ocupacion"):
                values[field] = st.selectbox(CATEGORIES[field][0], options[field], index=None, key=f"rf_{field}")
        with trip:
            st.markdown("**Viaje**")
            for field in ("motivo_grupo", "franja"):
                values[field] = st.selectbox(CATEGORIES[field][0], options[field], index=None, key=f"rf_{field}")
            values["num_etapas"] = st.number_input("Número de etapas", min_value=1, max_value=6,
                                                   value=None, step=1)
        with access:
            st.markdown("**Acceso a vehículos y conectividad**")
            for field, label in BINARY.items():
                values[field] = st.selectbox(label, (0, 1), format_func={0: "No", 1: "Sí"}.get,
                                             index=None, key=f"rf_{field}")
        submit = st.form_submit_button("Calcular clasificación", type="primary", disabled=pipeline is None)
    if submit:
        try:
            label, probability, individual = infer(pipeline, values, options)
            left, right = st.columns(2)
            left.metric("Clase estimada", ("Colectivo", "Individual motorizado")[label], border=True)
            right.metric("Probabilidad de la clase", f"{probability:.1%}".replace(".", ","), border=True)
            st.caption(f"Probabilidad de individual motorizado: {individual:.1%} · Umbral: 0,50.")
        except (ValueError, TypeError, RuntimeError) as exc:
            st.error(f"No se pudo estimar la clasificación. {exc}")
