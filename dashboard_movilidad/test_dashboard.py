import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from data_model import FIELDS, MIN_CELL, PUBLIC, ROOT


class DashboardTests(unittest.TestCase):
    def test_public_counts(self):
        data = pd.read_parquet(PUBLIC / "distribuciones.parquet")
        self.assertEqual(set(data.columns), {"variable", "categoria", "es_individual", "viajes"})
        self.assertEqual(set(data.variable), set(FIELDS))
        self.assertTrue(data.viajes.ge(MIN_CELL).all())
        class_counts = data.pivot_table(index="variable", columns="es_individual",
                                        values="viajes", aggfunc="sum")
        self.assertTrue(class_counts.nunique().eq(1).all())
        self.assertTrue(class_counts.sum(axis=1).eq(40529).all())

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
                self.assertEqual(len(app.get("plotly_chart")), 2)
                self.assertNotIn("No reporta", app.get("plotly_chart")[0].proto.spec)
                app.sidebar.multiselect[0].set_value(["5. No reporta"]).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get("plotly_chart")), 1)


if __name__ == "__main__":
    unittest.main()
