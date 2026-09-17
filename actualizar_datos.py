#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga/lee los datos mundiales definidos en config_dacel.json y arma
datos/datos_mundo.csv para el programa de validación.

Las series A (ansiedad) y Dp (depresión) se leen desde archivos locales
descargados directamente de IHME GBD 2023 y guardados en datos_fuente/.
Si una fuente falla, conserva los datos anteriores de esa variable y lo anota.
"""
import glob
import json
import os
import urllib.request

import pandas as pd

AGENTE = {"User-Agent": "TeoriaDacel/2.2 (edazelfernandez.com)"}
CARPETA = "datos"
ARCHIVO = os.path.join(CARPETA, "datos_mundo.csv")
ESTADO = os.path.join(CARPETA, "estado_fuentes.json")


def banco_mundial(codigo):
    url = (f"https://api.worldbank.org/v2/country/WLD/indicator/{codigo}"
           "?format=json&per_page=500")
    req = urllib.request.Request(url, headers=AGENTE)
    with urllib.request.urlopen(req, timeout=60) as r:
        datos = json.load(r)
    filas = [(int(d["date"]), d["value"]) for d in datos[1] if d["value"] is not None]
    return pd.Series(dict(filas), dtype=float).sort_index()


def _banco_mundial_paginas(url_base):
    """Descarga todas las páginas de una consulta del API del Banco Mundial."""
    filas, pagina, paginas = [], 1, 1
    while pagina <= paginas:
        sep = "&" if "?" in url_base else "?"
        req = urllib.request.Request(f"{url_base}{sep}page={pagina}", headers=AGENTE)
        with urllib.request.urlopen(req, timeout=120) as r:
            datos = json.load(r)
        if not isinstance(datos, list) or len(datos) < 2 or datos[1] is None:
            raise ValueError(f"Respuesta vacía del Banco Mundial: {str(datos)[:150]}")
        paginas = int(datos[0].get("pages", 1))
        filas.extend(datos[1])
        pagina += 1
    return filas


def _paises_reales():
    """Códigos ISO3 de países (excluye agregados como WLD, regiones o grupos de ingreso)."""
    filas = _banco_mundial_paginas("https://api.worldbank.org/v2/country?format=json&per_page=400")
    return {f["id"] for f in filas
            if f.get("region", {}).get("id") != "NA"
            and f.get("region", {}).get("value") != "Aggregates"}


def _por_pais(codigo, paises):
    url = (f"https://api.worldbank.org/v2/country/all/indicator/{codigo}"
           "?format=json&per_page=20000&date=1960:2100")
    filas = _banco_mundial_paginas(url)
    registros = [(f["countryiso3code"], int(f["date"]), float(f["value"]))
                 for f in filas
                 if f.get("value") is not None and f.get("countryiso3code") in paises]
    return pd.DataFrame(registros, columns=["pais", "anio", "valor"])


def banco_mundial_ponderado(codigo, cobertura_min=0.80, peso="SP.POP.TOTL"):
    """
    Promedio mundial ponderado por población, calculado con los datos de cada país.
    Se usa cuando el agregado mundial oficial (WLD) empieza tarde.

    - Mismo método para todos los años (no se mezcla con el agregado oficial).
    - Un año solo cuenta si los países con dato suman al menos `cobertura_min`
      de la población mundial; si no, queda vacío. No se inventan datos.
    """
    paises = _paises_reales()
    valores = _por_pais(codigo, paises)
    poblacion = _por_pais(peso, paises).rename(columns={"valor": "pob"})
    pob_mundo = banco_mundial(peso)
    t = valores.merge(poblacion, on=["pais", "anio"], how="inner")
    t["producto"] = t["valor"] * t["pob"]
    g = t.groupby("anio").agg(producto=("producto", "sum"), pob=("pob", "sum"),
                              n_paises=("pais", "nunique"))
    g["media"] = g["producto"] / g["pob"]
    g["cobertura"] = g["pob"] / pob_mundo.reindex(g.index)
    g = g[g["cobertura"] >= cobertura_min].dropna()
    if g.empty:
        raise ValueError("ningún año alcanza la cobertura mínima de población")
    serie = g["media"].astype(float).sort_index()
    serie.attrs["metodo"] = (f"promedio de países ponderado por población ({peso}); "
                             f"cobertura mínima {int(cobertura_min*100)}%")
    serie.attrs["cobertura_min_observada"] = round(float(g["cobertura"].min()), 3)
    serie.attrs["paises_min"] = int(g["n_paises"].min())
    # Comparación con el agregado oficial en los años que coinciden (transparencia)
    try:
        oficial = banco_mundial(codigo)
        comunes = serie.index.intersection(oficial.index)
        if len(comunes):
            serie.attrs["diferencia_max_vs_oficial"] = round(
                float((serie.loc[comunes] - oficial.loc[comunes]).abs().max()), 3)
    except Exception:
        pass
    return serie


def owid(slug, palabra):
    url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=false"
    df = pd.read_csv(url, storage_options=AGENTE)
    df = df[df["Entity"] == "World"]
    col = [c for c in df.columns if palabra.lower() in c.lower()
           and c not in ("Entity", "Code", "Year")]
    if not col:
        raise ValueError(f"No encontré una columna con '{palabra}' en {slug}")
    s = df.set_index("Year")[col[0]].dropna()
    return s.astype(float).sort_index()


def owid_github_energia(columna):
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    df = pd.read_csv(url, usecols=["country", "year", columna])
    s = df[df["country"] == "World"].set_index("year")[columna].dropna()
    return s.astype(float).sort_index()


def ihme_local(archivo):
    """Lee una descarga del GBD Results Tool (IHME).

    Espera las columnas 'year' y 'val'. En GBD, la métrica Porcentaje puede
    venir codificada como proporción 0–1; si es así, se convierte a 0–100.
    """
    ruta = archivo

    # Respaldo para nombres visualmente traducidos/acentuados en algunos móviles.
    if not os.path.exists(ruta):
        carpeta = os.path.dirname(ruta) or "."
        nombre = os.path.basename(ruta)
        if "depresion" in nombre:
            patron = os.path.join(carpeta, nombre.replace("depresion", "depres*"))
            candidatos = glob.glob(patron)
            if len(candidatos) == 1:
                ruta = candidatos[0]

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe el archivo IHME: {archivo}")

    df = pd.read_csv(ruta)
    faltan = {"year", "val"} - set(df.columns)
    if faltan:
        raise ValueError(f"Archivo IHME sin columnas requeridas: {sorted(faltan)}")

    sub = df[["year", "val"]].copy()
    sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
    sub["val"] = pd.to_numeric(sub["val"], errors="coerce")
    sub = sub.dropna().drop_duplicates(subset=["year"]).sort_values("year")

    if sub.empty:
        raise ValueError("Archivo IHME sin datos utilizables")

    # IHME GBD 2023 entrega 'Porcentaje' como fracción (p. ej. 0.0348 = 3.48%).
    if sub["val"].abs().max() <= 1.0:
        sub["val"] = sub["val"] * 100.0

    return pd.Series(sub["val"].values,
                     index=sub["year"].astype(int).values,
                     dtype=float).sort_index()


def main():
    with open("config_dacel.json", encoding="utf-8") as f:
        cfg = json.load(f)
    os.makedirs(CARPETA, exist_ok=True)
    anterior = pd.read_csv(ARCHIVO).set_index("anio") if os.path.exists(ARCHIVO) else pd.DataFrame()

    series, estado = {}, {}
    for var, fuente in cfg["fuentes"].items():
        try:
            if fuente["tipo"] == "banco_mundial":
                s = banco_mundial(fuente["codigo"])
            elif fuente["tipo"] == "owid":
                s = owid(fuente["slug"], fuente["palabra"])
            elif fuente["tipo"] == "owid_github_energia":
                s = owid_github_energia(fuente["columna"])
            elif fuente["tipo"] == "banco_mundial_ponderado":
                s = banco_mundial_ponderado(fuente["codigo"],
                                            fuente.get("cobertura_min", 0.80))
            elif fuente["tipo"] == "ihme_local":
                s = ihme_local(fuente["archivo"])
            else:
                raise ValueError(f"tipo desconocido {fuente['tipo']}")

            if len(s) < 5:
                raise ValueError("menos de 5 años de datos")

            series[var] = s
            estado[var] = {
                "nombre": fuente["nombre"],
                "ok": True,
                "anios": [int(s.index.min()), int(s.index.max())]
            }
            if s.attrs:
                estado[var]["metodo"] = dict(s.attrs)
        except Exception as e:
            if var in anterior and anterior[var].notna().any():
                series[var] = anterior[var].dropna()
                nota = "se usaron los datos guardados de la corrida anterior"
            else:
                nota = "sin datos"
            estado[var] = {
                "nombre": fuente["nombre"],
                "ok": False,
                "error": str(e)[:200],
                "nota": nota
            }

        print(var, "OK" if estado[var]["ok"] else f"FALLÓ ({estado[var]['nota']})")

    df = pd.DataFrame(series)
    df.index = df.index.astype(int)
    df.index.name = "anio"
    df = df.sort_index()

    eventos = {int(a) for a in cfg["perturbaciones"]}
    df["P"] = [1.0 if a in eventos else 0.0 for a in df.index]

    df.reset_index().to_csv(ARCHIVO, index=False)
    with open(ESTADO, "w", encoding="utf-8") as f:
        json.dump({
            "fuentes": estado,
            "perturbaciones": cfg["perturbaciones"],
            "cambios": cfg["cambios"]
        }, f, ensure_ascii=False, indent=2)

    print(f"Guardado {ARCHIVO}: {len(df)} años")


if __name__ == "__main__":
    main()
