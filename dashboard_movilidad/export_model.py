"""Export the trained Random Forest defined in Entrega_3.ipynb.

This intentionally executes only the notebook cells required to build the cleaned
dataset and fit the model; exploratory plots and MLflow registration are skipped.
"""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path, help="Folder containing datos/ and the dictionary XLSX")
    parser.add_argument("--notebook", type=Path, default=Path(__file__).parents[1] / "Entrega_3.ipynb")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "modelo")
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "display": lambda *args, **kwargs: None}
    required = {3, 6, 7, 9, 14, 16, 19, 20, 25, 26, 27, 31, 33, 35, 37, 39,
                41, 43, 45, 47, 49, 53, 56, 59, 61, 64, 66, 68, 69, 72, 73,
                74, 75, 79, 80, 83, 86, 87, 90, 103, 104, 105}
    source_dir = args.source_dir.resolve()
    dictionary = source_dir / "g. Diccionario base de datos.xlsx"
    data_dir = source_dir / "datos"

    for index, cell in enumerate(notebook["cells"]):
        if index not in required or cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if index == 3:
            source = "import pandas as pd\nimport duckdb\nimport glob\nimport os\n"
        if index == 6:
            source = 'con = duckdb.connect(":memory:")\n'
        if index == 7:
            source = source.replace(
                'RUTA_DICC = "https://raw.githubusercontent.com/sara-munoz-5/analitica-movilidad-bogota/DuclkDB/g.%20Diccionario%20base%20de%20datos.xlsx"',
                f"RUTA_DICC = r'{dictionary}'",
            )
        elif index == 9:
            source = f"""CARPETA_DATOS = r'{data_dir}'
SEP = ';'
for ruta in sorted(glob.glob(os.path.join(CARPETA_DATOS, '*.csv'))):
    nombre_base = os.path.splitext(os.path.basename(ruta))[0].lower()
    nombre_limpio = nombre_base.replace(' ', '_').replace('-', '_').replace('.', '_')
    nombre = f'raw_{{nombre_limpio}}'
    tmp_df = pd.read_csv(ruta, sep=SEP, encoding='latin1', dtype=str)
    con.execute(f'CREATE OR REPLACE TABLE {{nombre}} AS SELECT * FROM tmp_df')
"""
        elif index == 105:
            source = source.replace(
                "gs_rf.fit(X_train, y_train, groups=g_train, modelo__sample_weight=w_train)",
                "gs_rf.fit(X_train, y_train, groups=g_train, modelo__sample_weight=w_train)",
            )
        exec(compile(source, f"{args.notebook} cell {index}", "exec"), namespace)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import joblib

    pipeline = namespace["gs_rf"].best_estimator_
    joblib.dump(pipeline, output / "random_forest.joblib", compress=3)
    packages = ("scikit-learn", "numpy", "scipy", "pandas", "joblib")
    manifest = {"python": platform.python_version(), **{pkg: version(pkg) for pkg in packages}}
    (output / "entorno.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Modelo exportado en {output}")
    print(f"Mejores parámetros: {namespace['gs_rf'].best_params_}")


if __name__ == "__main__":
    main()
