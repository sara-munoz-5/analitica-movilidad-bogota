# ¿Quién elige el transporte individual?

Dashboard de la Encuesta de Movilidad Bogotá 2023 para la estrategia comercial de DiDi. La base analítica contiene 40.529 viajes y distingue transporte colectivo (0) e individual motorizado (1).

## Ejecución local

Desde la raíz del repositorio, con Python 3.14:

```bash
python -m pip install -r dashboard_movilidad/requirements.txt
python -m streamlit run dashboard_movilidad/app.py
python -m unittest discover -s dashboard_movilidad -p "test_*.py"
```

La vista predeterminada usa únicamente resultados agregados. Para explorar cruces en una sesión local privada, configura la ruta de los seis Parquet de la Entrega 3 y reinicia Streamlit.

```powershell
$env:MOVILIDAD_PARQUET_DIR = "C:\ruta\privada\data"
python -m streamlit run dashboard_movilidad/app.py
```

La carpeta privada debe contener `Fact_Viaje.parquet`, `Dim_Persona.parquet`, `Dim_Hogar.parquet`, `Dim_Modo.parquet`, `Dim_Motivo.parquet` y `Dim_Franja.parquet`. DuckDB une las tablas y valida el nivel de viaje. Los filtros locales se combinan y recalculan las gráficas descriptivas. «No reporta» se conserva en los datos y se oculta en la gráfica de ingresos.

## Despliegue en Streamlit Community Cloud

1. En Community Cloud, crea una aplicación y conecta el repositorio de GitHub.
2. Selecciona `dashboard-streamlit` para revisar la rama una vez disponible en GitHub o `main` después de integrar el pull request.
3. En **Main file path**, introduce `dashboard_movilidad/app.py`. En **Advanced settings**, selecciona Python 3.14.
4. Despliega con `MOVILIDAD_PARQUET_DIR` sin configurar. El repositorio aporta los agregados, figuras y dependencias.

[Guía oficial de despliegue de Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Fuentes y estructura

- `app.py` contiene la navegación, visualizaciones y tabla de métricas de prueba de la sección 6.1 de `Entrega_3.ipynb`.
- `data_model.py` carga los Parquet privados y genera los agregados públicos. Ejecuta `python dashboard_movilidad/data_model.py RUTA_PRIVADA` desde la raíz para regenerarlos.
- `public_data/` contiene distribuciones por una variable y clase, así como correlaciones de Pearson globales, redondeadas a dos decimales. La publicación agrupa categorías pequeñas y exige al menos 20 viajes por celda de distribución. Los cruces de filtros quedan disponibles en la sesión privada.
- `assets/` contiene el esquema dimensional, la figura original de importancia de Random Forest y las matrices y curvas de `Figuras_1_2_3.zip`, exportadas desde MLflow en la Entrega 3. Los conteos descriptivos son sin ponderar.
- `test_dashboard.py` prueba navegación, gráficos, filtros y propiedades de los datos públicos.

## Inferencia pendiente

Hace falta recuperar el pipeline ajustado de Random Forest del registro MLflow de la sección 6, con el preprocesador, el clasificador, las versiones de dependencias y la firma de las 17 variables. La Entrega 3 registró `X_test.head(5)` como ejemplo de entrada. Para un despliegue público, exporta una copia del pipeline sin ese ejemplo ni otros registros de prueba. La aplicación funciona con los resultados guardados de evaluación y no ejecuta entrenamiento.
