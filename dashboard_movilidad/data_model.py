import argparse
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public_data"
TABLES = ("Fact_Viaje", "Dim_Persona", "Dim_Hogar", "Dim_Motivo", "Dim_Franja", "Dim_Modo")
FIELDS = {
    "estrato_grupo": "Estrato", "ingreso_grupo": "Ingreso", "rango_edad": "Edad",
    "sexo": "Sexo", "localidad": "Localidad", "tiene_auto": "Tenencia de automóvil",
    "tiene_moto": "Tenencia de motocicleta", "tiene_bici": "Tenencia de bicicleta",
    "motivo_grupo": "Motivo", "franja": "Franja horaria",
    "licencia_auto": "Licencia de automóvil", "licencia_moto": "Licencia de motocicleta",
}
NUMERIC = {
    "edad": "Edad", "num_etapas": "Etapas", "tiene_auto": "Automóvil",
    "tiene_moto": "Motocicleta", "tiene_bici": "Bicicleta",
    "licencia_auto": "Licencia auto", "licencia_moto": "Licencia moto",
    "es_individual": "Viaje individual",
}
MIN_CELL = 20
PUBLIC_FILTERS = ("estrato_grupo", "tiene_auto")


def load_trips(directory):
    paths = {table: directory / f"{table}.parquet" for table in TABLES}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Parquet faltantes: " + ", ".join(missing))
    with duckdb.connect() as connection:
        for table, path in paths.items():
            connection.read_parquet(str(path)).create_view(table)
        trips = connection.execute("""
            SELECT f.es_individual, f.num_etapas, p.edad, p.rango_edad, p.sexo,
                   h.estrato_grupo, h.ingreso_grupo, h.localidad, h.tiene_auto,
                   h.tiene_moto, h.tiene_bici, p.licencia_auto, p.licencia_moto,
                   m.motivo_grupo, t.franja
            FROM Fact_Viaje f
            LEFT JOIN Dim_Persona p USING (sk_persona)
            LEFT JOIN Dim_Hogar h ON p.sk_hogar = h.sk_hogar
            LEFT JOIN Dim_Motivo m USING (sk_motivo)
            LEFT JOIN Dim_Franja t USING (sk_franja)
            LEFT JOIN Dim_Modo d USING (sk_modo)
        """).df()
        count, distinct_trips = connection.execute(
            "SELECT count(*), count(DISTINCT sk_viaje) FROM Fact_Viaje"
        ).fetchone()
    if not count or not count == distinct_trips == len(trips) or not trips.es_individual.isin([0, 1]).all():
        raise ValueError("Los Parquet no conservan un registro por viaje con clase 0 o 1.")
    for field in FIELDS:
        values = trips[field]
        if field in NUMERIC:
            values = values.map({0: "No", 1: "Sí"})
        trips[f"label_{field}"] = values.astype("string").fillna("Sin dato")
    return trips


def counts(trips, field):
    return (trips.groupby([f"label_{field}", "es_individual"]).size()
            .rename("viajes").reset_index().rename(columns={f"label_{field}": "categoria"}))


def class_counts(frame):
    groups = frame.groupby("es_individual")
    return (groups.viajes.sum() if "viajes" in frame else groups.size()).reindex([0, 1], fill_value=0)


def cross_counts(frame, first, second):
    columns = [f"label_{first}", f"label_{second}", "es_individual"]
    groups = frame.groupby(columns, as_index=False)
    return groups.viajes.sum() if "viajes" in frame else groups.size().rename(columns={"size": "viajes"})


def correlations(trips):
    return trips[list(NUMERIC)].rename(columns=NUMERIC).corr(min_periods=MIN_CELL).round(2)


def export_public(directory):
    trips = load_trips(directory)
    frames = []
    for field in FIELDS:
        table = counts(trips, field)
        pivot = table.pivot(index="categoria", columns="es_individual", values="viajes")
        small = pivot.fillna(0).min(axis=1) < MIN_CELL
        if small.sum() == 1 and (~small).any():
            small.loc[pivot.loc[~small].sum(axis=1).idxmin()] = True
        table.loc[table.categoria.isin(small[small].index), "categoria"] = "Categorías agrupadas"
        table = table.groupby(["categoria", "es_individual"], as_index=False).viajes.sum()
        protected = table.pivot(index="categoria", columns="es_individual", values="viajes")
        if protected.fillna(0).min().min() >= MIN_CELL:
            frames.append(table.assign(variable=field))
    cube = cross_counts(trips, *PUBLIC_FILTERS)
    cells = cube.pivot(index=[f"label_{field}" for field in PUBLIC_FILTERS],
                       columns="es_individual", values="viajes").reindex(columns=[0, 1])
    if cells.fillna(0).lt(MIN_CELL).any().any():
        raise ValueError("El cruce público contiene celdas inferiores al umbral de publicación.")

    PUBLIC.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(PUBLIC / "distribuciones.parquet", index=False)
    correlations(trips).to_csv(PUBLIC / "correlacion.csv")
    cube.to_parquet(PUBLIC / "cruce_estrato_auto.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera los Parquet agregados del dashboard público.")
    parser.add_argument("directory", type=Path)
    export_public(parser.parse_args().directory)
