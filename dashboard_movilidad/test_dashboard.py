import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from data_model import FIELDS, MIN_CELL, PUBLIC, PUBLIC_FILTERS, ROOT, class_counts
from prediction import FEATURES, age_band, infer


class DashboardTests(unittest.TestCase):
    def test_pipeline_classifies_transport_type(self):
        class FakePipeline:
            classes_ = [0, 1]

            def predict_proba(self, row):
                self.row = row
                return [[0.25, 0.75]]

        values = {
            "estrato_grupo": "Bajo (1-2)", "ingreso_grupo": "1. Hasta $1,16 M",
            "localidad": "Bosa", "rango_edad": "26-35", "sexo": "Hombre",
            "nivel_educativo": "4. Universitario", "ocupacion": "Empleado",
            "motivo_grupo": "Trabajo", "franja": "2. Pico AM (5-9 h)",
            "edad": 35, "num_etapas": 1, "tiene_auto": 1, "tiene_moto": 0,
            "tiene_bici": 0, "licencia_auto": 1, "licencia_moto": 0,
            "celular_datos": 1,
        }
        categories = {field: (value,) for field, value in values.items()
                      if field in {"estrato_grupo", "ingreso_grupo", "localidad", "rango_edad",
                                   "sexo", "nivel_educativo", "ocupacion", "motivo_grupo", "franja"}}
        label, probability, individual = infer(FakePipeline(), values, categories)
        self.assertEqual(label, 1)
        self.assertEqual(probability, 0.75)
        self.assertEqual(individual, 0.75)

    def test_public_counts(self):
        data = pd.read_parquet(PUBLIC / "distribuciones.parquet")
        self.assertEqual(set(data.columns), {"variable", "categoria", "es_individual", "viajes"})
        self.assertEqual(set(data.variable), set(FIELDS))
        self.assertTrue(data.viajes.ge(MIN_CELL).all())
        class_counts = data.pivot_table(index="variable", columns="es_individual",
                                        values="viajes", aggfunc="sum")
        self.assertTrue(class_counts.nunique().eq(1).all())
        self.assertTrue(class_counts.sum(axis=1).eq(40529).all())

    def test_public_cube_is_a_partition(self):
        cube = pd.read_parquet(PUBLIC / "cruce_estrato_auto.parquet")
        self.assertEqual(set(cube), {"viajes", "es_individual", *(f"label_{f}" for f in PUBLIC_FILTERS)})
        counts = cube.pivot(index=[f"label_{f}" for f in PUBLIC_FILTERS],
                            columns="es_individual", values="viajes")
        self.assertFalse(counts.fillna(0).lt(MIN_CELL).any().any())
        self.assertEqual(class_counts(cube).sum(), 40529)

    @patch.dict("os.environ", {"MOVILIDAD_PARQUET_DIR": "", "MOVILIDAD_MODELO_DIR": ""})
    def test_pages_and_figures(self):
        app = AppTest.from_file(ROOT / "app.py").run()
        self.assertEqual(app.metric[0].value, "40.529")
        for page in app.sidebar.radio[0].options:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, page)
            self.assertFalse(app.error, page)
            if page == "Evaluación de modelos":
                self.assertEqual(len(app.get("image")), 3)
            if page == "Análisis exploratorio":
                app.selectbox[0].set_value("ingreso_grupo").run()
                self.assertEqual(len(app.get("plotly_chart")), 4)
                self.assertNotIn("No reporta", app.get("plotly_chart")[0].proto.spec)
                app.multiselect(key="visible_ingreso_grupo").set_value(["5. No reporta"]).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get("plotly_chart")), 3)
            if page == "Predicción interactiva":
                self.assertEqual(len(FEATURES), 17)
                self.assertEqual(len(app.selectbox), 14)
                self.assertEqual(len(app.number_input), 2)
                self.assertTrue(next(button for button in app.button if button.label.startswith("Calcular")).disabled)
                app.number_input[0].set_value(35).run()
                self.assertEqual(app.text_input[0].value, age_band(35))

    @patch.dict("os.environ", {"MOVILIDAD_PARQUET_DIR": ""})
    def test_public_filters_and_segments(self):
        app = AppTest.from_file(ROOT / "app.py").run()
        app.sidebar.multiselect(key="public_filter_estrato_grupo").set_value(["Bajo (1-2)"]).run()
        self.assertEqual(app.metric[0].value, "19.078")
        app.sidebar.radio[0].set_value("Análisis exploratorio").run()
        self.assertEqual(app.selectbox[0].options, ["Estrato", "Tenencia de automóvil"])
        app.multiselect(key="Segmento A_tiene_auto").set_value(["No"]).run()
        app.multiselect(key="Segmento B_tiene_auto").set_value(["Sí"]).run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.metric[2].value, "+0,0 pp")
        app.sidebar.radio[0].set_value("Resumen ejecutivo").run()
        self.assertEqual(app.metric[0].value, "19.078")
        next(button for button in app.sidebar.button if button.label == "Restablecer filtros").click().run()
        self.assertEqual(app.metric[0].value, "40.529")


if __name__ == "__main__":
    unittest.main()
