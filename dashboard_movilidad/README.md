# ¿Quién elige el transporte individual?

Dashboard de la Encuesta de Movilidad Bogotá 2023 para la estrategia comercial de DiDi. La base analítica contiene 40.529 viajes y distingue transporte colectivo (0) e individual motorizado (1).

## Ejecución local

Desde la raíz del repositorio, con Python 3.14:

```bash
python -m pip install -r dashboard_movilidad/requirements.txt
python -m streamlit run dashboard_movilidad/app.py
python -m unittest discover -s dashboard_movilidad -p "test_*.py"
```

La vista predeterminada usa datos agregados. Para explorar cruces libres en una sesión local privada, configura la ruta de los seis Parquet de la Entrega 3 y reinicia Streamlit.

```powershell
$env:MOVILIDAD_PARQUET_DIR = "C:\ruta\privada\data"
python -m streamlit run dashboard_movilidad/app.py
```

La carpeta privada debe contener `Fact_Viaje.parquet`, `Dim_Persona.parquet`, `Dim_Hogar.parquet`, `Dim_Modo.parquet`, `Dim_Motivo.parquet` y `Dim_Franja.parquet`. DuckDB une las tablas y valida el nivel de viaje.

## Qué se puede explorar

| Función | Vista pública | Vista local privada |
| --- | --- | --- |
| Filtros globales y KPI | Estrato y tenencia de automóvil, combinables | Estrato, ingreso, edad por rango, sexo, localidad, vehículos, licencias, motivo y franja |
| Perfiles | Distribuciones marginales de todas las variables sin filtros globales. Con filtros activos, estrato y tenencia de automóvil | Todas las distribuciones se recalculan |
| Comparación A/B | Segmentos por estrato y tenencia de automóvil | Segmentos con filtros independientes sobre todas las variables de exploración |
| Cruce y correlación | Cruce estrato × automóvil y correlación global | Cualquier par distinto de variables y correlación sobre la selección |
| Descargas | Figuras Plotly como SVG desde la cámara y figuras de evaluación como PNG | Las mismas opciones sobre la selección local |

En la comparación, B − A mide la diferencia en puntos porcentuales de participación del transporte individual. Las proporciones usan como denominador los viajes de cada categoría o segmento. Las figuras académicas conservan sus muestras originales. «No reporta» se conserva en los datos y se oculta en las gráficas descriptivas de ingreso.

## Despliegue en Streamlit Community Cloud

1. En Community Cloud, crea una aplicación y conecta el repositorio de GitHub.
2. Selecciona `dashboard-inteligencia-comercial` para revisar la rama o `main` después de integrar el pull request.
3. En **Main file path**, introduce `dashboard_movilidad/app.py`. En **Advanced settings**, selecciona Python 3.14.
4. Despliega con `MOVILIDAD_PARQUET_DIR` y `MOVILIDAD_MODELO_DIR` sin configurar. El repositorio aporta los agregados, figuras y dependencias. La interfaz predictiva se abre con el cálculo deshabilitado.

[Guía oficial de despliegue de Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Fuentes y estructura

- `app.py` contiene la navegación, visualizaciones y tabla de métricas de prueba de la sección 6.1 de `Entrega_3.ipynb`.
- `data_model.py` carga los Parquet privados y genera los agregados públicos. Ejecuta `python dashboard_movilidad/data_model.py RUTA_PRIVADA` desde la raíz para regenerarlos.
- `public_data/` contiene distribuciones por variable y clase, una correlación de Pearson global redondeada a dos decimales y un único cruce de estrato × automóvil × clase. Cada celda publicada contiene al menos 20 viajes. Las consultas públicas filtran o suman esas mismas celdas disjuntas. Las distribuciones marginales quedan separadas del cruce.
- `assets/` contiene el esquema dimensional, la figura original de importancia de Random Forest y las matrices y curvas de `Figuras_1_2_3.zip`, exportadas desde MLflow en la Entrega 3. Los conteos descriptivos son sin ponderar.
- `test_dashboard.py` prueba navegación, gráficos, filtros y propiedades de los datos públicos.

## Habilitar inferencia con el modelo ya entrenado

En la sección 6 de `Entrega_3.ipynb`, MLflow registra `gs_rf.best_estimator_` con los pasos ajustados `prep` y `modelo`. También registra `X_test.head(5)` como ejemplo de entrada. Usa un Colab aparte o una celda temporal en la sesión original y exporta solo el objeto ajustado.

Si la sesión original sigue activa, empieza con `pipeline = gs_rf.best_estimator_`. Si solo conservas el almacenamiento de MLflow, recupera tanto `mlflow.db` como el directorio de artefactos al que apunta esa base. Abre la corrida padre `random_forest` con AUC-ROC de prueba 0,9137, toma el ID del modelo registrado como `modelo` y ejecuta lo siguiente en el entorno original:

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:////content/mlflow.db")
pipeline = mlflow.sklearn.load_model("models:/m-ID_DEL_MODELO")
```

[Formato de URI y modelos guardados en MLflow 3](https://www.mlflow.org/docs/latest/tracking/)

Con `pipeline` recuperado por cualquiera de las dos vías, ejecuta esta celda. No ejecuta `fit` ni accede a `X_train` o `X_test`.

```python
import json
import platform
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import joblib
from google.colab import files

folder = Path("/content/dashboard_rf")
folder.mkdir(exist_ok=True)
packages = ("scikit-learn", "numpy", "scipy", "pandas", "joblib")
versions = {name: version(name) for name in packages}
joblib.dump(pipeline, folder / "random_forest.joblib", compress=3)
(folder / "entorno.json").write_text(
    json.dumps({"python": platform.python_version(), **versions}, indent=2), encoding="utf-8"
)
with ZipFile("/content/dashboard_rf.zip", "w") as archive:
    for name in ("random_forest.joblib", "entorno.json"):
        archive.write(folder / name, arcname=name)
files.download("/content/dashboard_rf.zip")
```

Descomprime en una carpeta privada y configura el entorno local con el mismo Python y las cinco versiones de `entorno.json`. En un entorno virtual, instala primero los requisitos del dashboard. Luego, en PowerShell:

```powershell
$env:MOVILIDAD_MODELO_DIR = "C:\ruta\privada\dashboard_rf"
$versions = Get-Content "$env:MOVILIDAD_MODELO_DIR/entorno.json" -Raw | ConvertFrom-Json
$packages = @("scikit-learn", "numpy", "scipy", "pandas", "joblib")
$pins = $packages | ForEach-Object { "$_==$($versions.$_)" }
python -m pip install $pins
python -m streamlit run dashboard_movilidad/app.py
```

El formulario toma las categorías admitidas del codificador ajustado, deriva el rango de edad e invoca `predict_proba` con umbral 0,50. Usa exclusivamente un archivo `joblib` propio y de confianza. Para habilitarlo en un futuro despliegue, proporciona al servidor solo la exportación limpia y alinea también los requisitos de Community Cloud con ese entorno. Mantén privados el directorio original de MLflow, su ejemplo de entrada y los Parquet fuente. Si la sesión terminó y su almacenamiento no fue conservado, el notebook por sí solo no contiene el pipeline entrenado.
