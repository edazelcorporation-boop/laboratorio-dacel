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

AGENTE = {"User-Agent": "TeoriaDacel/2.1 (edazelfernandez.com)"}
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
