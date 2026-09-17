#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEORÍA DAÇEL — Motor multiescala + H3-v6 multicausal/H3-v7 replicación — Programa de validación empírica (v4.1)
Autor de la teoría: Edazel Fernández (Edazel Corporation)

Qué hace este programa
----------------------
No "demuestra" la teoría: la SOMETE A PRUEBA. Toma series históricas,
calcula los índices CIDI y HDF con fórmulas explícitas y evalúa las
hipótesis H1–H4 y la Ecuación General contra modelos rivales más simples.
Si Daçel no supera a los rivales, el programa lo dice.

Uso
---
  python dacel_validacion.py --demo              # dos mundos sintéticos de prueba
  python dacel_validacion.py --datos mis_datos.csv   # tus datos reales

Columnas del CSV (ver plantilla_datos.csv):
  anio  E  I  X  K  A  S  D  T  conexion  P  L
  E = energía disponible        I = información procesada
  X = externalización cognitiva K = producción de conocimiento
  A = ansiedad  S = soledad  Dp = depresión (opcional)  D = dependencia digital
  T = exposición informacional  conexion = hiperconectividad (p. ej. % usuarios de internet)
  P = perturbación (0 = año normal, 1 = evento disruptivo, o intensidad)
  L = pérdidas sistémicas (opcional)
Las celdas vacías se permiten; cada prueba usa los años que tengan datos.
"""

import argparse
import json
import os
import sys
import warnings
import re
import glob
import io
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

VEREDICTO_OK = "SUPERA LA PRUEBA"
VEREDICTO_NO = "NO SUPERA LA PRUEBA"
VEREDICTO_NC = "NO CONCLUYENTE"
ALFA = 0.05


# ============================================================
# 1. ÍNDICES DAÇEL (fórmulas explícitas, ya no f(...))
# ============================================================

def calcular_cidi(df, pesos=None, anio_base=None):
    """
    CIDI = Π (V_i / V_i,base) ^ w_i      con Σ w_i = 1
    Media geométrica ponderada de E, I, X, K, cada una relativa a su valor
    en el año base. CIDI(año base) = 1. Si CIDI = 50, la civilización
    procesa ~50 veces más información (en sentido compuesto) que en el año base.
    """
    cols = ["E", "I", "X", "K"]
    pesos = pesos or {c: 0.25 for c in cols}
    sub = df[["anio"] + cols].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    if anio_base is None:
        anio_base = sub["anio"].iloc[0]
    base = sub[sub["anio"] == anio_base].iloc[0]
    log_cidi = sum(pesos[c] * np.log(sub[c] / base[c]) for c in cols)
    return pd.Series(np.exp(log_cidi.values), index=sub["anio"].values, name="CIDI")


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def calcular_hdf(df, pesos=None):
    """
    HDF = Σ w_j · z(V_j)    con V ∈ {A, S, Dp, D, T} disponibles,  Σ w_j = 1
    (Dp = depresión, opcional)
    z = puntuación estándar. HDF = 0 es el promedio del periodo;
    HDF = +1 es una desviación estándar por encima (más desajuste).
    """
    cols = [c for c in ["A", "S", "Dp", "D", "T"] if c in df and df[c].notna().sum() >= 5]
    if not cols:
        return pd.Series(dtype=float)
    pesos = pesos or {c: 1 / len(cols) for c in cols}
    sub = df[["anio"] + cols].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    hdf = sum(pesos[c] * zscore(sub[c]) for c in cols)
    return pd.Series(hdf.values, index=sub["anio"].values, name="HDF")


def calcular_hdf_psicologico(df):
    """Solo componentes psicológicos (A, S, Dp). Se usa en H2 para evitar
    circularidad: D y T YA son medidas de conectividad."""
    cols = [c for c in ["A", "S", "Dp"] if c in df and df[c].notna().sum() >= 5]
    if not cols:
        return pd.Series(dtype=float)
    sub = df[["anio"] + cols].dropna()
    h = sum(zscore(sub[c]) for c in cols) / len(cols)
    return pd.Series(h.values, index=sub["anio"].values, name="HDF_psico")


# ============================================================
# 2. HERRAMIENTAS ESTADÍSTICAS
# ============================================================

def modelo_lineal(t, a, b):
    return a + b * t

def modelo_exponencial_log(t, a, r):
    return a + r * t                     # log y = a + r t

def modelo_logistico_log(t, K, r, t0):
    return np.log(K) - np.log1p(np.exp(-r * (t - t0)))

def modelo_gompertz_log(t, K, b, c):
    return np.log(K) - b * np.exp(-c * t)


def aic(resid, k):
    n = len(resid)
    rss = max(np.sum(resid ** 2), 1e-12)
    return n * np.log(rss / n) + 2 * k


def ajustar_modelos_crecimiento(t, y):
    """Ajusta 4 modelos en escala log. Devuelve dict nombre -> (función, parámetros)."""
    ly = np.log(y)
    ymax = y.max()
    res = {}
    candidatos = {
        "lineal": (lambda tt, a, b: np.log(np.clip(modelo_lineal(tt, a, b), 1e-9, None)),
                   [y[0], (y[-1] - y[0]) / max(t[-1], 1)], None),
        "exponencial": (modelo_exponencial_log, [ly[0], 0.02], None),
        "logistico": (modelo_logistico_log, [ymax * 1.5, 0.05, t.mean()],
                      ([ymax * 0.5, 1e-4, -1e3], [ymax * 1e3, 2.0, 1e4])),
        "gompertz": (modelo_gompertz_log, [ymax * 1.5, 3.0, 0.02],
                     ([ymax * 0.5, 1e-3, 1e-5], [ymax * 1e3, 1e3, 2.0])),
    }
    for nombre, (f, p0, lim) in candidatos.items():
        try:
            if lim:
                p, _ = curve_fit(f, t, ly, p0=p0, bounds=lim, maxfev=50000)
            else:
                p, _ = curve_fit(f, t, ly, p0=p0, maxfev=50000)
            res[nombre] = (f, p)
        except Exception:
            pass
    return res


def iaaft(x, rng, iteraciones=100):
    """Surrogate IAAFT: conserva distribución y espectro, destruye la relación con otra serie."""
    x = np.asarray(x, float)
    ordenado = np.sort(x)
    amplitudes = np.abs(np.fft.rfft(x))
    s = rng.permutation(x)
    for _ in range(iteraciones):
        fase = np.angle(np.fft.rfft(s))
        s = np.fft.irfft(amplitudes * np.exp(1j * fase), n=len(x))
        s = ordenado[np.argsort(np.argsort(s))]
    return s


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ============================================================
# 3. PRUEBAS
# ============================================================

def prueba_colinealidad(df, reporte):
    """El riesgo que ya habíamos detectado: que el PCA solo recupere el tiempo."""
    cols = ["E", "I", "X", "K"]
    sub = df[["anio"] + cols].dropna()
    if len(sub) < 8:
        reporte.append("## Control previo — Colinealidad temporal\n\n- Pocos años completos; se omite.\n")
        return np.nan
    Z = np.log(sub[cols].values)
    Z = (Z - Z.mean(0)) / Z.std(0)
    _, sv, vt = np.linalg.svd(Z, full_matrices=False)
    var_exp = sv ** 2 / np.sum(sv ** 2)
    pc1 = Z @ vt[0]
    r_tiempo = abs(stats.pearsonr(pc1, sub["anio"])[0])

    # Mismo análisis sobre tasas de crecimiento (quita la tendencia común)
    G = np.diff(np.log(sub[cols].values), axis=0)
    G = (G - G.mean(0)) / G.std(0)
    _, sv2, _ = np.linalg.svd(G, full_matrices=False)
    var_exp2 = sv2 ** 2 / np.sum(sv2 ** 2)

    reporte.append("## Control previo — Colinealidad temporal\n")
    reporte.append(f"- PC1 en niveles explica {var_exp[0]*100:.1f}% de la varianza; "
                   f"su correlación con el año es |r| = {r_tiempo:.3f}.")
    reporte.append(f"- PC1 en tasas de crecimiento explica {var_exp2[0]*100:.1f}%.")
    if r_tiempo > 0.95:
        reporte.append("- ADVERTENCIA: en niveles, el PCA está midiendo básicamente el paso del tiempo. "
                       "Por eso todas las pruebas causales de este programa trabajan con "
                       "tasas de cambio (series sin tendencia), no con niveles.\n")
    else:
        reporte.append("- La señal en niveles no es solo tiempo.\n")
    return r_tiempo


def prueba_h1(cidi, reporte, carpeta, etiqueta):
    """H1: el CIDI crece exponencialmente. Rivales: lineal, logístico, Gompertz."""
    reporte.append("## H1 — El CIDI crece exponencialmente\n")
    if len(cidi) < 15:
        reporte.append(f"**Veredicto H1: {VEREDICTO_NC}.** Solo {len(cidi)} años con E, I, X y K completos; se necesitan al menos 15.\n")
        return VEREDICTO_NC
    t = (cidi.index.values - cidi.index.values[0]).astype(float)
    y = cidi.values
    n_ent = int(len(t) * 0.8)

    ajustes = ajustar_modelos_crecimiento(t, y)
    ajustes_ent = ajustar_modelos_crecimiento(t[:n_ent], y[:n_ent])
    filas = []
    for nombre, (f, p) in ajustes.items():
        r = np.log(y) - f(t, *p)
        fila = {"modelo": nombre, "AIC": aic(r, len(p)), "RMSE_fuera": np.nan}
        if nombre in ajustes_ent:
            f2, p2 = ajustes_ent[nombre]
            fila["RMSE_fuera"] = rmse(np.log(y[n_ent:]), f2(t[n_ent:], *p2))
        filas.append(fila)
    tabla = pd.DataFrame(filas).sort_values("AIC")
    reporte.append("Escala logarítmica. Entrenamiento: primer 80% de los años; "
                   "validación fuera de muestra: último 20%.\n")
    reporte.append(tabla.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n")

    # Veredicto: ¿la tasa de crecimiento es positiva y NO va en descenso?
    # Exponencial = tasa estable. Logístico/Gompertz = tasa que cae con el tiempo.
    g = np.diff(np.log(y))
    tg = t[1:]
    media_g = g.mean()
    p_media = stats.ttest_1samp(g, 0, alternative="greater").pvalue
    reg = stats.linregress(tg, g)
    p_caida = reg.pvalue / 2 if reg.slope < 0 else 1 - reg.pvalue / 2
    reporte.append("Tabla descriptiva. El veredicto no depende de ella, porque con series tan "
                   "largas un logístico con techo lejano es casi idéntico a un exponencial.\n")
    reporte.append(f"- Crecimiento medio del CIDI: {media_g*100:.2f}% anual (p = {p_media:.4f})")
    reporte.append(f"- Cambio de esa tasa por década: {reg.slope*1000:+.3f} pp (p de descenso = {p_caida:.4f})")
    if p_media < ALFA and p_caida >= ALFA:
        v = VEREDICTO_OK
        txt = "El CIDI crece a una tasa positiva que no muestra desaceleración: comportamiento exponencial."
    elif p_media >= ALFA:
        v = VEREDICTO_NO
        txt = "No hay crecimiento sostenido del CIDI."
    else:
        v = VEREDICTO_NO
        txt = "El CIDI crece, pero cada vez más lento: es saturación (logístico/Gompertz), no exponencial."
    reporte.append(f"\n**Veredicto H1: {v}.** {txt}")
    reporte.append("Nota: H1 es una hipótesis débil. Casi cualquier medida tecnológica crece así, "
                   "con o sin Daçel. Superarla no distingue a Daçel de otras teorías.\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(cidi.index, y, "k.", label="CIDI")
    for nombre, (f, p) in ajustes.items():
        ax.semilogy(cidi.index, np.exp(f(t, *p)), label=nombre)
    ax.axvline(cidi.index[n_ent], color="gray", ls="--", lw=1, label="inicio validación")
    ax.set_title(f"H1 — CIDI y modelos de crecimiento ({etiqueta})")
    ax.set_xlabel("Año"); ax.set_ylabel("CIDI (año base = 1)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(carpeta, "H1_CIDI.png"), dpi=130); plt.close(fig)
    return v


def prueba_h2(df, reporte, carpeta, etiqueta, rng, n_sur=1000):
    """H2: el vacío humano aumenta con la hiperconectividad (versión sin tendencia + IAAFT)."""
    reporte.append("## H2 / F4 — El vacío humano aumenta con la hiperconectividad\n")
    h = calcular_hdf_psicologico(df)
    if h.empty:
        reporte.append(f"**Veredicto H2: {VEREDICTO_NC}.** No hay datos psicológicos.\n")
        return VEREDICTO_NC
    c = df.set_index("anio")["conexion"].dropna()
    comunes = h.index.intersection(c.index)
    if len(comunes) < 15:
        reporte.append(f"**Veredicto H2: {VEREDICTO_NC}.** Solo {len(comunes)} años con datos; se necesitan al menos 15.\n")
        return VEREDICTO_NC
    h, c = h.loc[comunes].values, c.loc[comunes].values

    r_niveles = stats.pearsonr(h, c)[0]
    dh, dc = np.diff(h), np.diff(c)
    r_obs = stats.pearsonr(dh, dc)[0]
    nulos = np.array([stats.pearsonr(iaaft(dh, rng), dc)[0] for _ in range(n_sur)])
    p = (np.sum(nulos >= r_obs) + 1) / (n_sur + 1)

    reporte.append("Se usa solo la parte psicológica del HDF (ansiedad, soledad y/o depresión). Incluir D y T "
                   "sería circular, porque ya son medidas de conectividad.\n")
    reporte.append(f"- Correlación en niveles: r = {r_niveles:.3f} "
                   "(engañosa: dos series que suben juntas siempre correlacionan).")
    reporte.append(f"- Correlación en cambios año a año: r = {r_obs:.3f}")
    reporte.append(f"- Significancia contra {n_sur} surrogates IAAFT: p = {p:.4f}")
    if r_obs > 0 and p < ALFA:
        v, txt = VEREDICTO_OK, "Cuando la conectividad sube más rápido, el desajuste psicológico también."
    elif p >= ALFA:
        v, txt = VEREDICTO_NO, "La relación desaparece al quitar la tendencia compartida."
    else:
        v, txt = VEREDICTO_NO, "La relación es significativa pero en sentido contrario."
    reporte.append(f"\n**Veredicto H2: {v}.** {txt}\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(nulos, bins=40, color="lightgray", label="surrogates (sin relación)")
    ax.axvline(r_obs, color="red", lw=2, label=f"observado r={r_obs:.2f}")
    ax.set_title(f"H2 — Cambios en vacío humano vs conectividad ({etiqueta})")
    ax.set_xlabel("correlación"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(carpeta, "H2_vacio_humano.png"), dpi=130); plt.close(fig)
    return v



def _crecimiento_anual(df, columna):
    """Tasa logarítmica anual. Si hay huecos, anualiza por la distancia entre años."""
    sub = df[["anio", columna]].dropna().sort_values("anio")
    sub = sub[sub[columna] > 0]
    if len(sub) < 3:
        return pd.Series(dtype=float)
    anios = sub["anio"].to_numpy(dtype=int)
    vals = sub[columna].to_numpy(dtype=float)
    dt = np.diff(anios).astype(float)
    g = np.diff(np.log(vals)) / dt
    return pd.Series(g, index=anios[1:], name=columna)


def _efecto_variable(g, evento, ventana, escala):
    """Cambio post-pre en crecimiento, expresado en desviaciones estándar."""
    if g.empty or not np.isfinite(escala) or escala <= 0:
        return np.nan
    pre_idx = np.arange(evento - ventana, evento, dtype=int)
    post_idx = np.arange(evento + 1, evento + ventana + 1, dtype=int)
    pre = g.reindex(pre_idx)
    post = g.reindex(post_idx)
    if pre.isna().any() or post.isna().any():
        return np.nan
    return float((post.mean() - pre.mean()) / escala)


def _efectos_dominios_en_anio(anio, composicion, crecimientos, escalas, ventana):
    """Devuelve el cambio estandarizado post-pre de cada dominio disponible."""
    efectos = {}
    for dominio, vars_dom in composicion.items():
        efectos_vars = []
        for var in vars_dom:
            e = _efecto_variable(crecimientos[var], anio, ventana, escalas[var])
            if np.isfinite(e):
                efectos_vars.append(e)
        if efectos_vars:
            efectos[dominio] = float(np.mean(efectos_vars))
    return efectos


def _efecto_sistemico_en_anio(anio, composicion, crecimientos, escalas, ventana,
                               modo="magnitud"):
    """
    Intensidad de reorganización entre el régimen pre y post.

    modo="magnitud": promedio de |cambio estandarizado| entre dominios.
    Esta es la métrica primaria de H3: reorganizar puede significar acelerar,
    contraer o sustituir un dominio por otro; el signo no define si hubo
    reconfiguración, sino su magnitud.

    modo="direccion": promedio con signo (diagnóstico de aceleración neta).
    """
    efectos = _efectos_dominios_en_anio(
        anio, composicion, crecimientos, escalas, ventana
    )
    if len(efectos) < 2:
        return np.nan
    vals = np.array(list(efectos.values()), dtype=float)
    if modo == "direccion":
        return float(np.mean(vals))
    return float(np.mean(np.abs(vals)))


def _composicion_evento(evento, dominios, crecimientos, escalas, ventana):
    """Fija qué evidencia estaba realmente disponible para cada perturbación."""
    comp = {}
    for dominio, vars_dom in dominios.items():
        validas = []
        for var in vars_dom:
            e = _efecto_variable(crecimientos[var], evento, ventana, escalas[var])
            if np.isfinite(e):
                validas.append(var)
        if validas:
            comp[dominio] = validas
    return comp


def prueba_h3_eventos_v2(df, reporte, carpeta, etiqueta, rng, ventana=5, n_perm=5000):
    """
    H3 sistémica: una perturbación aumenta la intensidad de reorganización del sistema,
    no solo el crecimiento de I.

    Dominios Daçel:
      - energía: E
      - información procesada: I
      - externalización/conectividad: X_fija, X y conexion (un solo dominio)
      - conocimiento: K_patentes y K (un solo dominio)

    Cada variable se convierte a crecimiento logarítmico anual y el efecto post-pre
    se estandariza por la desviación estándar histórica de esa variable. Después se
    promedian DOMINIOS para que X+conexion no dupliquen el peso de conectividad.

    Para cada perturbación se conserva exactamente la composición de evidencia
    disponible en ese año. La métrica primaria es la MAGNITUD del cambio de régimen
    post-pre (valor absoluto estandarizado), porque una reorganización puede incluir
    expansiones, contracciones o sustituciones entre dominios. El promedio con signo
    se informa solo como diagnóstico de aceleración neta.

    El placebo usa años donde esa misma composición tiene ventanas completas.
    Se requieren >=2 dominios por evento y >=3 eventos.
    """
    reporte.append("## H3 / F2 — Las perturbaciones intensifican la reorganización sistémica\n")
    reporte.append(
        "Prueba principal ampliada y preregistrada: la reorganización se mide en los dominios "
        "estructurales de Daçel — energía (E), información procesada (I), externalización/"
        "conectividad (telefonía fija, móvil e internet, contadas como un solo dominio) y "
        "conocimiento (patentes y artículos científicos, contados como un solo dominio). "
        "Ansiedad y depresión no entran aquí porque pertenecen a H2.\n"
    )
    reporte.append(
        f"Se conserva la ventana original de ±{ventana} años. Cada dominio compara su régimen "
        "de crecimiento posterior con el anterior, estandarizado por su variabilidad histórica. "
        "La prueba principal usa la magnitud absoluta del cambio: una crisis puede reorganizar "
        "el sistema haciendo subir unos dominios y caer otros, por lo que promediar signos "
        "opuestos cancelaría precisamente la reconfiguración que H3 intenta medir.\n"
    )

    dominios = {
        "energía": ["E"],
        "información": ["I"],
        "externalización/conectividad": ["X_fija", "X", "conexion"],
        "conocimiento": ["K_patentes", "K"],
    }
    variables = sorted({v for vs in dominios.values() for v in vs})
    crecimientos = {}
    escalas = {}
    for v in variables:
        if v in df:
            g = _crecimiento_anual(df, v)
        else:
            g = pd.Series(dtype=float)
        crecimientos[v] = g
        escalas[v] = float(g.std(ddof=1)) if len(g) >= 3 else np.nan

    eventos = sorted(int(a) for a in df.loc[df["P"] > 0, "anio"].dropna().unique())
    detalles = []
    for ev in eventos:
        comp = _composicion_evento(ev, dominios, crecimientos, escalas, ventana)
        magnitud = _efecto_sistemico_en_anio(
            ev, comp, crecimientos, escalas, ventana, modo="magnitud"
        )
        direccion = _efecto_sistemico_en_anio(
            ev, comp, crecimientos, escalas, ventana, modo="direccion"
        )
        efectos_dom = _efectos_dominios_en_anio(
            ev, comp, crecimientos, escalas, ventana
        )
        if np.isfinite(magnitud) and len(comp) >= 2:
            detalles.append((ev, comp, magnitud, direccion, efectos_dom))

    if len(detalles) < 3:
        reporte.append(
            f"**Veredicto H3: {VEREDICTO_NC}.** Solo {len(detalles)} perturbaciones tienen "
            "al menos 2 dominios con ventanas completas; se necesitan al menos 3.\n"
        )
        return VEREDICTO_NC

    reporte.append("Perturbaciones evaluables y evidencia usada:")
    for ev, comp, magnitud, direccion, efectos_dom in detalles:
        desc = "; ".join(f"{d}: {','.join(vs)}" for d, vs in comp.items())
        dom_txt = ", ".join(f"{d}={v:+.2f}" for d, v in efectos_dom.items())
        reporte.append(
            f"- {ev}: {len(comp)} dominios [{desc}] → intensidad {magnitud:.3f} DE; "
            f"dirección neta {direccion:+.3f} DE ({dom_txt})"
        )

    obs = float(np.mean([x[2] for x in detalles]))
    direccion_obs = float(np.mean([x[3] for x in detalles]))

    # Placebo emparejado: para cada evento real se exige la MISMA composición de variables.
    eventos_set = set(eventos)
    candidatos_por_evento = {}
    anio_min = int(df["anio"].min())
    anio_max = int(df["anio"].max())
    for ev, comp, _, _, _ in detalles:
        candidatos = []
        for a in range(anio_min + ventana, anio_max - ventana + 1):
            if a in eventos_set:
                continue
            e = _efecto_sistemico_en_anio(
                a, comp, crecimientos, escalas, ventana, modo="magnitud"
            )
            if np.isfinite(e):
                candidatos.append((a, e))
        candidatos_por_evento[ev] = candidatos

    if any(len(candidatos_por_evento[ev]) < 5 for ev, _, _, _, _ in detalles):
        reporte.append(
            f"**Veredicto H3: {VEREDICTO_NC}.** No hay suficientes años placebo "
            "comparables para todas las perturbaciones.\n"
        )
        return VEREDICTO_NC

    nulos = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        vals = []
        for ev, comp, _, _, _ in detalles:
            cand = candidatos_por_evento[ev]
            idx = int(rng.integers(0, len(cand)))
            vals.append(cand[idx][1])
        nulos[i] = float(np.mean(vals))

    p = (np.sum(nulos >= obs) + 1) / (n_perm + 1)

    reporte.append(
        f"\nResultado sistémico: promedio de {len(detalles)} perturbaciones evaluables "
        f"frente a {n_perm} panoramas placebo emparejados por disponibilidad de evidencia."
    )
    reporte.append(f"- Intensidad de reorganización observada: {obs:.3f} desviaciones estándar")
    reporte.append(f"- Intensidad placebo media: {np.mean(nulos):.3f} DE")
    reporte.append(f"- Dirección neta observada (diagnóstica): {direccion_obs:+.3f} DE")
    reporte.append(f"- p = {p:.4f}")

    if p < ALFA:
        v = VEREDICTO_OK
        txt = ("La intensidad de cambio del sistema después de las perturbaciones es mayor "
               "que la esperada en años comparables al azar, considerando varios dominios Daçel.")
    else:
        v = VEREDICTO_NO
        txt = ("La evidencia sistémica disponible no distingue la magnitud de reorganización "
               "posterior a perturbaciones de la observada en años comparables al azar.")
    reporte.append(f"\n**Veredicto H3: {v}.** {txt}\n")

    # Diagnóstico histórico: conservar la prueba antigua basada solo en I para auditoría.
    sub_i = df[["anio", "I", "P"]].dropna().reset_index(drop=True)
    g_i = np.diff(np.log(sub_i["I"].values))
    anios_g_i = sub_i["anio"].values[1:]
    ev_i = sub_i.loc[sub_i["P"] > 0, "anio"].values

    def efecto_i(lista):
        difs = []
        for e in lista:
            pre = g_i[(anios_g_i < e) & (anios_g_i >= e - ventana)]
            post = g_i[(anios_g_i > e) & (anios_g_i <= e + ventana)]
            if len(pre) == ventana and len(post) == ventana:
                difs.append(post.mean() - pre.mean())
        return (np.mean(difs) if difs else np.nan), len(difs)

    obs_i, n_ev_i = efecto_i(ev_i)
    reporte.append(
        "### H3a diagnóstica — prueba histórica basada solo en I\n"
        "Se conserva para auditoría y comparación, pero ya no representa por sí sola "
        "el veredicto de H3 sistémica."
    )
    if n_ev_i >= 3:
        validos_i = anios_g_i[(anios_g_i >= anios_g_i[0] + ventana) &
                              (anios_g_i <= anios_g_i[-1] - ventana)]
        candidatos_i = np.setdiff1d(validos_i, ev_i)
        nulos_i = []
        # subflujo determinista derivado del mismo RNG sin afectar el veredicto principal
        for _ in range(2000):
            falsos = rng.choice(candidatos_i, size=n_ev_i, replace=False)
            nulos_i.append(efecto_i(falsos)[0])
        nulos_i = np.asarray(nulos_i)
        p_i = (np.sum(nulos_i >= obs_i) + 1) / (len(nulos_i) + 1)
        reporte.append(
            f"- Eventos evaluables: {n_ev_i}; efecto I: {obs_i*100:+.2f} pp; "
            f"p diagnóstica = {p_i:.4f}"
        )
    else:
        reporte.append(f"- Solo {n_ev_i} eventos evaluables en I.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(nulos, bins=45, color="lightgray", label="panoramas placebo")
    ax.axvline(obs, color="red", lw=2, label="perturbaciones reales")
    ax.set_title(f"H3 — Intensidad de reorganización sistémica ({etiqueta})")
    ax.set_xlabel("magnitud del cambio de régimen (desviaciones estándar)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "H3_perturbaciones.png"), dpi=130)
    plt.close(fig)
    return v



def _h3_dominios(df):
    return {
        "energía": ["E"],
        "información": ["I"],
        "externalización/conectividad": ["X_fija", "X", "conexion"],
        "conocimiento": ["K_patentes", "K"],
    }


def _h3_reorganizacion_continua(df, ventana=5):
    """R(t): magnitud del cambio de régimen pre/post en dominios Daçel."""
    dominios = _h3_dominios(df)
    variables = sorted({v for vs in dominios.values() for v in vs})
    crecimientos, escalas = {}, {}
    for v in variables:
        g = _crecimiento_anual(df, v) if v in df else pd.Series(dtype=float)
        crecimientos[v] = g
        escalas[v] = float(g.std(ddof=1)) if len(g) >= 3 else np.nan

    anios = np.arange(int(df["anio"].min()) + ventana,
                      int(df["anio"].max()) - ventana + 1)
    bruto = pd.DataFrame(index=anios, columns=list(dominios), dtype=float)
    for a in anios:
        for dom, vars_dom in dominios.items():
            vals = []
            for v in vars_dom:
                e = _efecto_variable(crecimientos[v], int(a), ventana, escalas[v])
                if np.isfinite(e):
                    vals.append(abs(e))
            if vals:
                bruto.loc[a, dom] = float(np.mean(vals))

    # Escala robusta por dominio. NO se centra en cero: R es intensidad >=0.
    norm = bruto.copy()
    for c in norm.columns:
        ss = norm[c].dropna()
        if len(ss) >= 8:
            med = float(ss.median())
            mad = float(np.median(np.abs(ss - med)))
            esc = 1.4826 * mad
            if not np.isfinite(esc) or esc <= 0:
                esc = float(ss.std(ddof=1))
            norm[c] = norm[c] / esc if esc > 0 else np.nan
        else:
            norm[c] = np.nan
    n_dom = norm.notna().sum(axis=1)
    R = norm.mean(axis=1, skipna=True)
    R[n_dom < 2] = np.nan
    return R.dropna(), n_dom.loc[R.dropna().index], bruto


def _leer_gpr_anual(ruta="datos_fuente/ai_gpr_data_monthly.csv"):
    """AI-GPR mensual -> promedio anual; excluye años incompletos."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe {ruta}")
    g = pd.read_csv(ruta)
    if "Date" not in g.columns or "GPR_AI" not in g.columns:
        raise ValueError("AI-GPR requiere columnas Date y GPR_AI")
    g["Date"] = pd.to_datetime(g["Date"], errors="coerce")
    g["GPR_AI"] = pd.to_numeric(g["GPR_AI"], errors="coerce")
    g = g.dropna(subset=["Date", "GPR_AI"])
    g["anio"] = g["Date"].dt.year.astype(int)
    agg = g.groupby("anio")["GPR_AI"].agg(["mean", "count"])
    return agg.loc[agg["count"] == 12, "mean"].astype(float)


def _percentil_expansivo(s, min_hist=10):
    """Percentil ex-ante: en t solo usa observaciones <=t; evita mirar el futuro."""
    s = pd.Series(s, dtype=float).sort_index()
    out = pd.Series(index=s.index, dtype=float)
    for i, (a, v) in enumerate(s.items()):
        hist = s.iloc[:i+1].dropna()
        if len(hist) >= min_hist and np.isfinite(v):
            out.loc[a] = float(stats.percentileofscore(hist.values, v, kind="mean") / 100.0)
    return out


def _presion_externa(df):
    """P_ext(t): perturbación externa continua, sin lista manual de eventos."""
    gpr = _leer_gpr_anual()
    if "G_mundo" not in df.columns:
        raise ValueError("Falta G_mundo (crecimiento anual del PIB mundial)")
    pib = df.set_index("anio")["G_mundo"].dropna().astype(float)
    idx = gpr.index.intersection(pib.index)
    geo = np.log1p(gpr.loc[idx].clip(lower=0))
    # Disrupción macro ex-ante: desviación respecto a la mediana de los 10 años PREVIOS.
    macro = pd.Series(index=idx, dtype=float)
    for a in idx:
        prev = pib.loc[(pib.index >= a-10) & (pib.index <= a-1)].dropna()
        if len(prev) >= 7:
            macro.loc[a] = abs(float(pib.loc[a]) - float(prev.median()))
    geo_p = _percentil_expansivo(geo)
    mac_p = _percentil_expansivo(macro)
    Pext = pd.concat([geo_p.rename("geo"), mac_p.rename("macro")], axis=1).mean(axis=1, skipna=False)
    return Pext.dropna()


def _memoria_perturbacion(Pext, vida_media=3.0):
    """M(t): presión acumulada con memoria exponencial; vida media en años."""
    lam = np.exp(-np.log(2.0) / float(vida_media))
    out = pd.Series(index=Pext.index, dtype=float)
    m = 0.0
    previo = None
    for a, p in Pext.sort_index().items():
        if previo is not None:
            m *= lam ** max(int(a-previo), 1)
        m = float(p) + m
        out.loc[a] = m
        previo = a
    # escala interpretable 0..1 dentro de la historia ya observada, ex-ante
    return _percentil_expansivo(out, min_hist=10).dropna()

def _memoria_perturbacion_historica(Pext, vida_media=3.0):
    """
    Memoria causal de cobertura histórica para H3-v5.
    Conserva toda observación válida de P_ext.
    No repercentiliza M ni introduce información futura.
    """
    lam = np.exp(-np.log(2.0) / float(vida_media))
    out = pd.Series(index=Pext.index, dtype=float)
    m = 0.0
    previo = None
    for a, p in Pext.sort_index().items():
        if previo is not None:
            m *= lam ** max(int(a-previo), 1)
        m = float(p) + m
        out.loc[a] = m
        previo = a
    return out.dropna()
def _capacidad_adaptativa(df):
    """
    A_cap(t): capacidad previa para absorber/reorganizarse.
    Usa niveles E, X/conexión y K/patentes; EXCLUYE I para no construir el predictor
    con la variable que luego queremos explicar en la ecuación general.
    Cada dominio se expresa como percentil ex-ante y se promedian >=2 dominios.
    """
    d = df.set_index("anio").sort_index()
    dominios = {
        "energía": ["E"],
        "externalización": ["X_fija", "X", "conexion"],
        "conocimiento": ["K_patentes", "K"],
    }
    dom_series = {}
    for dom, vars_ in dominios.items():
        partes=[]
        for v in vars_:
            if v in d.columns and d[v].notna().sum() >= 10:
                x = pd.to_numeric(d[v], errors="coerce")
                # log para magnitudes positivas; percentil ex-ante elimina unidades.
                x = np.log1p(x.clip(lower=0))
                partes.append(_percentil_expansivo(x).rename(v))
        if partes:
            dom_series[dom] = pd.concat(partes, axis=1).mean(axis=1, skipna=True)
    if not dom_series:
        return pd.Series(dtype=float)
    tab = pd.DataFrame(dom_series)
    n = tab.notna().sum(axis=1)
    A = tab.mean(axis=1, skipna=True)
    A[n < 2] = np.nan
    # lag 1: la capacidad debe existir ANTES de la respuesta observada.
    return A.shift(1).dropna()


def _p_circular(x, y, estadistico):
    """p unilateral con todos los desplazamientos circulares no nulos."""
    x=np.asarray(x,float); y=np.asarray(y,float)
    obs=float(estadistico(x,y))
    nul=np.array([estadistico(np.roll(x,k),y) for k in range(1,len(x))],float)
    nul=nul[np.isfinite(nul)]
    p=float((np.sum(nul >= obs)+1)/(len(nul)+1)) if len(nul) else np.nan
    return obs,p,nul


def prueba_h3(df, reporte, carpeta, etiqueta, rng=None, ventana=5):
    """
    H3-v4: mecanismo Daçel dinámico.
      P_ext -> M (memoria) ; Q=M*A_cap -> R (reorganización).
    Separa perturbación, capacidad y reorganización. No exige que un shock produzca
    crecimiento inmediato ni que todas las variables cambien con el mismo signo.
    """
    reporte.append("## H3 / F2 — Perturbación, memoria y reorganización sistémica\n")
    reporte.append(
        "### H3-v4 principal — mecanismo dinámico Daçel\n"
        "La perturbación no se trata como crecimiento. Se separan cuatro conceptos: "
        "P_ext(t), perturbación externa; M(t), memoria acumulada de perturbaciones; "
        "A_cap(t), capacidad adaptativa existente antes de la respuesta; y R(t), magnitud "
        "de reorganización del sistema. La presión adaptativa es Q(t)=M(t)×A_cap(t). "
        "Una perturbación puede destruir unas variables y acelerar otras; por eso R mide "
        "magnitud de cambio de régimen y no crecimiento neto.\n"
    )
    reporte.append(
        "P_ext combina AI-GPR y disrupción del crecimiento mundial. Sus transformaciones "
        "son ex-ante: cada año se compara solo con historia disponible hasta ese año. "
        "M usa memoria exponencial con vida media fija de 3 años. A_cap usa energía, "
        "externalización/conectividad y conocimiento, excluyendo I para evitar circularidad. "
        f"R compara {ventana} años previos y {ventana} posteriores en los cuatro dominios Daçel.\n"
    )
    try:
        R,n_dom,_=_h3_reorganizacion_continua(df,ventana)
        Pext=_presion_externa(df)
        M=_memoria_perturbacion(Pext,3.0)
        A=_capacidad_adaptativa(df)
    except Exception as e:
        reporte.append(f"**Veredicto H3: {VEREDICTO_NC}.** No se pudo construir H3-v4: {e}\n")
        return VEREDICTO_NC
    idx=R.index.intersection(M.index).intersection(A.index)
    if len(idx)<20:
        reporte.append(f"**Veredicto H3: {VEREDICTO_NC}.** Solo {len(idx)} años evaluables; se requieren al menos 20.\n")
        return VEREDICTO_NC
    q=(M.loc[idx]*A.loc[idx]).astype(float)
    r=R.loc[idx].astype(float)
    stat=lambda x,y: float(stats.spearmanr(x,y).statistic)
    rho,p,nulos=_p_circular(q.values,r.values,stat)
    reporte.append(f"- Años evaluables: {len(idx)} ({int(idx.min())}–{int(idx.max())})")
    reporte.append(f"- Dominios de R por año: mediana {float(n_dom.loc[idx].median()):.1f}; rango {int(n_dom.loc[idx].min())}–{int(n_dom.loc[idx].max())}")
    reporte.append(f"- Asociación Q(t)=M×A_cap → R(t): rho = {rho:+.3f}")
    reporte.append(f"- Nulo temporal: {len(nulos)} desplazamientos circulares; p = {p:.4f}")
    if rho>0 and p<ALFA:
        v=VEREDICTO_OK; txt="La presión adaptativa acumulada se asocia con una reorganización mayor que bajo alineaciones temporales placebo."
    elif rho>0:
        v=VEREDICTO_NO; txt="La dirección es la prevista, pero no se distingue del nulo temporal al 5%."
    else:
        v=VEREDICTO_NO; txt="La relación observada no tiene la dirección prevista por el mecanismo dinámico."
    reporte.append(f"\n**Veredicto H3-v4: {v}.** {txt}\n")
    reporte.append(
        "### Auditoría\nH3-v2 (eventos manuales) y H3-v3 (perturbación contemporánea continua) "
        "permanecen en el historial del repositorio. H3-v4 no reescribe esos resultados; "
        "prueba una formulación dinámica explícita de la teoría.\n"
    )
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(q.values,r.values,alpha=.75)
    ax.set_xlabel("Q(t)=memoria de perturbación × capacidad adaptativa previa")
    ax.set_ylabel("R(t): intensidad de reorganización")
    ax.set_title(f"H3-v4 — mecanismo dinámico Daçel ({etiqueta})")
    fig.tight_layout(); fig.savefig(os.path.join(carpeta,"H3_perturbaciones.png"),dpi=130); plt.close(fig)
    return v

def prueba_h3_v5_historica(df, reporte, carpeta, etiqueta, ventana=5):
    """
    H3-v5 — cobertura histórica causal.
    Prueba registrada para ampliar la cobertura temporal sin imputar datos,
    sin usar información futura adicional y sin modificar H3-v4.
    La formulación se fija antes de observar su resultado.
    """
    reporte.append("### H3-v5 — cobertura histórica causal\n")
    reporte.append(
        "Esta prueba conserva P_ext, vida media de 3 años, A_cap, R, "
        "ventana temporal, alfa y nulo circular de H3-v4. "
        "La única diferencia es que M conserva desde su primera observación "
        "válida la memoria causal ya construida, sin imponer un segundo "
        "calentamiento de 10 observaciones para repercentilizarla.\n"
    )

    try:
        R, n_dom, _ = _h3_reorganizacion_continua(df, ventana)
        Pext = _presion_externa(df)
        M = _memoria_perturbacion_historica(Pext, 3.0)
        A = _capacidad_adaptativa(df)
    except Exception as e:
        reporte.append(
            f"**H3-v5 SIN DATOS / ERROR DE CONSTRUCCIÓN:** "
            f"{type(e).__name__}: {e}\n"
        )
        return VEREDICTO_NC

    idx = R.index.intersection(M.index).intersection(A.index)

    def cobertura(nombre, s):
        if len(s) == 0:
            reporte.append(f"- {nombre}: sin años disponibles")
        else:
            reporte.append(
                f"- {nombre}: {len(s)} años "
                f"({int(s.index.min())}–{int(s.index.max())})"
            )

    reporte.append("#### Auditoría de cobertura")
    cobertura("P_ext", Pext)
    cobertura("M histórica", M)
    cobertura("A_cap", A)
    cobertura("R", R)

    if len(idx):
        reporte.append(
            f"- Intersección R ∩ M ∩ A_cap: {len(idx)} años "
            f"({int(idx.min())}–{int(idx.max())})"
        )

    if len(idx) < 20:
        reporte.append(
            f"**Veredicto H3-v5: {VEREDICTO_NC}.** "
            f"Solo {len(idx)} años evaluables; se requieren al menos 20.\n"
        )
        return VEREDICTO_NC

    q = (M.loc[idx] * A.loc[idx]).astype(float)
    r = R.loc[idx].astype(float)

    stat = lambda x, y: float(stats.spearmanr(x, y).statistic)
    rho, p, nulos = _p_circular(q.values, r.values, stat)

    reporte.append(
        f"- Asociación Q(t)=M_hist×A_cap → R(t): rho = {rho:+.3f}"
    )
    reporte.append(
        f"- Nulo temporal: {len(nulos)} desplazamientos circulares; "
        f"p = {p:.4f}"
    )
    reporte.append(
        f"- Dominios de R: mediana "
        f"{float(n_dom.loc[idx].median()):.1f}; "
        f"rango {int(n_dom.loc[idx].min())}–"
        f"{int(n_dom.loc[idx].max())}"
    )

    if rho > 0 and p < ALFA:
        v = VEREDICTO_OK
        txt = (
            "La presión adaptativa acumulada se asocia con una "
            "reorganización mayor que bajo alineaciones temporales placebo."
        )
    elif rho > 0:
        v = VEREDICTO_NO
        txt = (
            "La dirección es la prevista, pero no se distingue "
            "del nulo temporal al 5%."
        )
    else:
        v = VEREDICTO_NO
        txt = (
            "La relación observada no tiene la dirección prevista "
            "por el mecanismo dinámico."
        )

    reporte.append(f"**Veredicto H3-v5: {v}.** {txt}\n")

    return v

def prueba_h4(df, reporte, anio_limite=2040):
    """H4: la externalización cognitiva seguirá creciendo hasta 2040 (proyección, no prueba)."""
    reporte.append("## H4 — La externalización cognitiva seguirá aumentando hasta 2040\n")
    sub = df[["anio", "X"]].dropna()
    if len(sub) < 11:
        reporte.append(f"**Veredicto H4: {VEREDICTO_NC}.** Faltan datos de X.\n")
        return VEREDICTO_NC
    ultimos = sub.tail(10)
    g = np.diff(np.log(ultimos["X"].values))
    reporte.append(f"- Crecimiento medio de X en los últimos 10 años: {g.mean()*100:.2f}% anual.")
    reporte.append(f"- Años con caída: {int(np.sum(g < 0))} de {len(g)}.")
    reporte.append(f"**Veredicto H4: {VEREDICTO_NC}** hasta {anio_limite}. Es una predicción a futuro: "
                   "hay que registrarla hoy y comprobarla cuando lleguen los datos. "
                   f"Quedará refutada si X deja de crecer de forma sostenida antes de {anio_limite} (criterio F3).\n")
    return VEREDICTO_NC


def prueba_ecuacion_general(df, reporte, carpeta, etiqueta):
    """
    Ecuación Daçel dinámica (v2):
      gI(t+1) = c + α gE(t) + γ gX(t) + β Q(t) - δ L(t)
      Q(t)=M(t)*A_cap(t)
    Q sustituye al viejo P binario: una perturbación es presión, no desarrollo directo.
    Todo predictor está fechado antes del objetivo para evitar fuga temporal.
    """
    reporte.append("## Ecuación General dinámica — gI(t+1) = c + αgE(t) + γgX(t) + βQ(t) − δL(t)\n")
    try:
        Pext=_presion_externa(df); M=_memoria_perturbacion(Pext,3.0); A=_capacidad_adaptativa(df)
    except Exception as e:
        reporte.append(f"**Veredicto Ecuación General: {VEREDICTO_NC}.** No se pudo construir Q(t): {e}\n")
        return VEREDICTO_NC
    d=df.set_index("anio").sort_index()
    gI=_crecimiento_anual(df,"I"); gE=_crecimiento_anual(df,"E"); gX=_crecimiento_anual(df,"X")
    Q=(M*A).rename("Q")
    # objetivo del año siguiente: crecimiento I observado en t+1
    rows=[]
    for t in sorted(set(gE.index)&set(gX.index)&set(Q.index)):
        if t+1 not in gI.index: continue
        row=[int(t),float(gI.loc[t+1]),float(gE.loc[t]),float(gX.loc[t]),float(Q.loc[t])]
        if "L" in d.columns and pd.notna(d["L"].get(t,np.nan)):
            row.append(float(d.loc[t,"L"]))
        else: row.append(0.0)
        if np.all(np.isfinite(row[1:])): rows.append(row)
    tab=pd.DataFrame(rows,columns=["t","y","gE","gX","Q","L"])
    if len(tab)<20:
        reporte.append(f"**Veredicto Ecuación General: {VEREDICTO_NC}.** Solo {len(tab)} transiciones completas; se requieren al menos 20.\n")
        return VEREDICTO_NC
    Mx=np.column_stack([np.ones(len(tab)),tab.gE,tab.gX,tab.Q,-tab.L])
    y=tab.y.to_numpy(float); n_ent=max(12,int(len(y)*.8)); n_ent=min(n_ent,len(y)-4)
    coef,*_=np.linalg.lstsq(Mx[:n_ent],y[:n_ent],rcond=None)
    pred=Mx[n_ent:]@coef
    pred_const=np.full(len(y)-n_ent,y[:n_ent].mean())
    # persistencia: último crecimiento I conocido en t
    pers=[]
    for t in tab.t.iloc[n_ent:]: pers.append(float(gI.loc[t]) if t in gI.index else y[:n_ent].mean())
    pers=np.array(pers)
    e=rmse(y[n_ent:],pred); ec=rmse(y[n_ent:],pred_const); ep=rmse(y[n_ent:],pers)
    ss=np.sum((y[n_ent:]-y[:n_ent].mean())**2)
    r2=1-np.sum((y[n_ent:]-pred)**2)/ss if ss>0 else np.nan
    nombres=["c","α (energía)","γ (externalización)","β (presión adaptativa Q)","δ (pérdidas)"]
    reporte.append(f"Transiciones evaluables: {len(tab)}; entrenamiento {n_ent}, prueba {len(tab)-n_ent}.\n")
    reporte.append("Parámetros estimados solo con el tramo de entrenamiento:")
    for nm,cf in zip(nombres,coef): reporte.append(f"- {nm} = {cf:+.4f}")
    reporte.append("\nError fuera de muestra (menor es mejor):")
    reporte.append(f"- Ecuación Daçel dinámica: {e*100:.3f}")
    reporte.append(f"- Crecimiento constante: {ec*100:.3f}")
    reporte.append(f"- Persistencia: {ep*100:.3f}")
    reporte.append(f"- R² fuera de muestra: {r2:.3f}")
    mejora=1-e/min(ec,ep)
    if mejora>.10: v=VEREDICTO_OK; txt=f"Reduce el error {mejora*100:.0f}% frente al mejor rival simple."
    elif mejora>0: v=VEREDICTO_NC; txt=f"Mejora {mejora*100:.0f}% frente al mejor rival, insuficiente para el criterio del 10%."
    else: v=VEREDICTO_NO; txt="Un rival simple predice igual o mejor."
    reporte.append(f"\n**Veredicto Ecuación General dinámica: {v}.** {txt}\n")
    fig,ax=plt.subplots(figsize=(8,5)); years=(tab.t+1).to_numpy()
    ax.plot(years,y*100,"k-",lw=1,label="real"); ax.plot(years[n_ent:],pred*100,"r-",label="Daçel dinámica")
    ax.plot(years[n_ent:],pred_const*100,"b--",label="constante"); ax.plot(years[n_ent:],pers*100,"g:",label="persistencia")
    ax.set_title(f"Ecuación General dinámica — crecimiento futuro de I ({etiqueta})"); ax.set_xlabel("Año"); ax.set_ylabel("% anual"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(carpeta,"Ecuacion_General.png"),dpi=130); plt.close(fig)
    return v


# ============================================================
# 4. DATOS SINTÉTICOS (solo para verificar que el programa funciona)
# ============================================================

def generar_mundo(con_dacel, semilla, anios=range(1900, 2026)):
    """
    con_dacel=True : el mecanismo Daçel existe (I responde a E, X y perturbaciones;
                     el vacío humano responde a la conectividad).
    con_dacel=False: todo crece con tendencias parecidas, pero SIN relación causal.
    Un buen método debe aprobar el primero y reprobar el segundo.
    """
    rng = np.random.default_rng(semilla)
    anios = np.array(list(anios)); n = len(anios)
    P = np.zeros(n)
    for e in [1914, 1929, 1939, 1957, 1973, 1991, 2008, 2020]:
        P[anios == e] = 1

    gE = 0.025 + 0.012 * rng.standard_normal(n)
    gX = 0.035 + 0.015 * rng.standard_normal(n)
    gK = 0.030 + 0.010 * rng.standard_normal(n)
    if con_dacel:
        efecto_P = np.zeros(n)
        for i in np.where(P > 0)[0]:
            efecto_P[i + 1:i + 6] += 0.035
        gI = 0.005 + 0.6 * gE + 0.7 * gX + efecto_P + 0.004 * rng.standard_normal(n)
    else:
        gI = 0.045 + 0.018 * rng.standard_normal(n)

    E = 100 * np.exp(np.cumsum(gE)); X = 1 * np.exp(np.cumsum(gX))
    K = 10 * np.exp(np.cumsum(gK)); I = 5 * np.exp(np.cumsum(gI))

    conexion = 100 / (1 + np.exp(-0.18 * (anios - 2008)))
    conexion = conexion + 0.8 * rng.standard_normal(n)
    dcon = np.diff(conexion, prepend=conexion[0])
    tend = (anios - 1900) / 125
    if con_dacel:
        A = 20 + 5 * tend + np.cumsum(0.25 * dcon + 0.3 * rng.standard_normal(n))
        S = 15 + 4 * tend + np.cumsum(0.20 * dcon + 0.3 * rng.standard_normal(n))
    else:
        A = 20 + 5 * tend + 0.03 * conexion.cumsum() / n + np.cumsum(0.3 * rng.standard_normal(n))
        S = 15 + 4 * tend + 0.02 * conexion.cumsum() / n + np.cumsum(0.3 * rng.standard_normal(n))
    D = 0.5 * conexion + rng.standard_normal(n)
    T = 1 + 0.08 * conexion + 0.2 * rng.standard_normal(n)
    L = 0.01 * rng.random(n)

    df = pd.DataFrame(dict(anio=anios, E=E, I=I, X=X, K=K, A=A, S=S, D=D, T=T,
                           conexion=conexion, P=P, L=L))
    # los datos psicológicos solo existen desde 1990 (como en la realidad)
    df.loc[df["anio"] < 1990, ["A", "S", "D", "T", "conexion"]] = np.nan
    return df





# ============================================================
# 4A. H3-v6 — PERTURBACIÓN MULTICAUSAL / H3-v7 REPLICACIÓN
# ============================================================
#
# REGLA DE AUDITORÍA:
# - H3-v4 y H3-v5 NO se modifican.
# - H3-v6 añade familias independientes de perturbación.
# - No se imputan años faltantes.
# - No se seleccionan "años famosos".
# - Horizonte primario h=3; h=1 y h=5 son sensibilidad.
# - La respuesta R es MAGNITUD de reorganización, no progreso.
# - Una familia sin datos suficientes se publica como SIN DATOS.
#
# Los archivos externos pueden depositarse en datos_fuente/. El cargador
# busca nombres razonables, sin exigir que el usuario los renombre.


def _buscar_archivos_dacel(patrones, carpeta="datos_fuente", extensiones=None):
    """Devuelve TODOS los archivos coincidentes. Importante para WHO partido en chunks."""
    if not os.path.isdir(carpeta):
        return []
    pats=[str(p).lower() for p in patrones]
    exts={e.lower() for e in extensiones} if extensiones else None
    candidatos=[]
    for raiz, _, archivos in os.walk(carpeta):
        for nombre in archivos:
            bajo=nombre.lower()
            if any(p in bajo for p in pats):
                ruta=os.path.join(raiz,nombre)
                if exts is None or os.path.splitext(bajo)[1] in exts:
                    candidatos.append(ruta)
    return sorted(set(candidatos))


def _buscar_archivo_dacel(patrones, carpeta="datos_fuente"):
    rutas=_buscar_archivos_dacel(patrones,carpeta)
    return rutas[0] if rutas else None


def _leer_csv_flexible(ruta):
    """CSV tolerante a BOM/latin1 y separadores comunes."""
    ultimo=None
    for enc in ("utf-8-sig","utf-8","latin1"):
        for sep in (",",";","\t"):
            try:
                d=pd.read_csv(ruta,encoding=enc,sep=sep,low_memory=False)
                if d.shape[1] > 1:
                    return d
            except Exception as e:
                ultimo=e
    raise ValueError(f"No se pudo leer {ruta}: {ultimo}")


def _leer_tabla_dacel(ruta):
    bajo=ruta.lower()
    if bajo.endswith((".xlsx",".xls")):
        return pd.read_excel(ruta)
    return _leer_csv_flexible(ruta)


def _anio_numerico(s):
    return pd.to_numeric(s,errors="coerce").astype("Int64")


def _percentil_exante_min(s, min_hist=8):
    """Normalización causal 0..1; cada t usa únicamente historia <=t."""
    s=pd.Series(s,dtype=float).sort_index()
    out=pd.Series(index=s.index,dtype=float)
    for i,(a,v) in enumerate(s.items()):
        hist=s.iloc[:i+1].dropna()
        if len(hist)>=min_hist and np.isfinite(v):
            out.loc[a]=stats.percentileofscore(hist.to_numpy(),v,kind="mean")/100.0
    return out


def _combinar_canales_exante(canales, min_canales=1):
    """Promedia canales ya normalizados; nunca rellena un canal ausente."""
    validos=[pd.Series(s,dtype=float).rename(n) for n,s in canales.items()
             if len(pd.Series(s).dropna())]
    if not validos:
        return pd.Series(dtype=float)
    t=pd.concat(validos,axis=1)
    n=t.notna().sum(axis=1)
    p=t.mean(axis=1,skipna=True)
    p[n<min_canales]=np.nan
    return p.dropna()


def _serie_conflicto_ucdp():
    ruta=_buscar_archivo_dacel(["ucdpprioconflict","ucdp_prio","armedconflict"])
    if not ruta:
        return pd.Series(dtype=float), {"estado":"SIN DATOS","fuente":"UCDP/PRIO"}
    d=_leer_csv_flexible(ruta)
    cmap={str(c).lower():c for c in d.columns}; cy=cmap.get("year")
    if cy is None: raise ValueError("UCDP no contiene columna year")
    d["__anio"]=_anio_numerico(d[cy]); d=d.dropna(subset=["__anio"]); d["__anio"]=d["__anio"].astype(int)
    cid=cmap.get("conflict_id"); cint=cmap.get("intensity_level"); cloc=cmap.get("location")
    cstart=cmap.get("start_date") or cmap.get("start_date2"); g=d.groupby("__anio")
    activos=g[cid].nunique() if cid else g.size()
    altos=(d.assign(__hi=(pd.to_numeric(d[cint],errors="coerce")>=2).astype(float))
           .groupby("__anio")["__hi"].sum()) if cint else pd.Series(index=activos.index,dtype=float)
    dispersion=g[cloc].nunique() if cloc else pd.Series(index=activos.index,dtype=float)
    if cstart:
        sy=pd.to_datetime(d[cstart],errors="coerce").dt.year
        nuevos=d.assign(__nuevo=(sy==d["__anio"]).astype(float)).groupby("__anio")["__nuevo"].sum()
    elif cid:
        primero=d.groupby(cid)["__anio"].transform("min")
        nuevos=d.assign(__nuevo=(d["__anio"]==primero).astype(float)).groupby("__anio")["__nuevo"].sum()
    else: nuevos=pd.Series(index=activos.index,dtype=float)
    canales={"activos":_percentil_exante_min(np.log1p(activos)),
             "alta_intensidad":_percentil_exante_min(np.log1p(altos)),
             "nuevos":_percentil_exante_min(np.log1p(nuevos)),
             "dispersion":_percentil_exante_min(np.log1p(dispersion))}
    p=_combinar_canales_exante(canales,min_canales=2)
    return p,{"estado":"OK","fuente":os.path.basename(ruta),
              "rango":[int(p.index.min()),int(p.index.max())] if len(p) else None}


def _serie_muertes_batalla():
    ruta=_buscar_archivo_dacel(["battledeaths","battle_deaths"])
    if not ruta: return pd.Series(dtype=float), {"estado":"SIN DATOS","fuente":"UCDP Battle Deaths"}
    d=_leer_csv_flexible(ruta); cmap={str(c).lower():c for c in d.columns}
    if "year" not in cmap or "bd_best" not in cmap: raise ValueError("BattleDeaths requiere year y bd_best")
    z=pd.DataFrame({"anio":_anio_numerico(d[cmap["year"]]),
                    "v":pd.to_numeric(d[cmap["bd_best"]],errors="coerce")}).dropna()
    s=z.groupby("anio")["v"].sum().astype(float)
    p=_percentil_exante_min(np.log1p(s.clip(lower=0))).dropna()
    return p,{"estado":"OK","fuente":os.path.basename(ruta),
              "rango":[int(p.index.min()),int(p.index.max())] if len(p) else None}


def _serie_sismos_usgs():
    # El archivo real del usuario se llama consulta.csv.
    ruta=_buscar_archivo_dacel(["consulta.csv","query.csv","earthquake","usgs","sismo"])
    if not ruta: return pd.Series(dtype=float), {"estado":"SIN DATOS","fuente":"USGS"}
    d=_leer_csv_flexible(ruta); cmap={str(c).lower():c for c in d.columns}
    if "time" not in cmap or "mag" not in cmap: raise ValueError("USGS requiere time y mag")
    fecha=pd.to_datetime(d[cmap["time"]],errors="coerce",utc=True)
    mag=pd.to_numeric(d[cmap["mag"]],errors="coerce")
    z=pd.DataFrame({"anio":fecha.dt.year,"mag":mag}).dropna(); z=z[z.mag>=6.0]; g=z.groupby("anio")
    conteo=g.size().astype(float); maxmag=g.mag.max().astype(float)
    def logenergia(x):
        a=1.5*np.asarray(x,float); m=float(np.max(a))
        return m+np.log10(np.sum(10**(a-m)))
    energia=g.mag.apply(logenergia).astype(float)
    p=_combinar_canales_exante({
        "conteo":_percentil_exante_min(np.log1p(conteo)),
        "maxmag":_percentil_exante_min(maxmag),
        "energia":_percentil_exante_min(energia)},min_canales=2)
    return p,{"estado":"OK","fuente":os.path.basename(ruta),
              "rango":[int(p.index.min()),int(p.index.max())] if len(p) else None}


def _leer_wdi_largo():
    """Lee WDI tanto si se subió CSV como el ZIP original de World Bank."""
    rutas=_buscar_archivos_dacel(["p_data_extract","world_development","wdi","data.csv"])
    if not rutas: return pd.DataFrame(), {"estado":"SIN DATOS","fuente":"World Bank WDI"}
    ultimo=None
    for ruta in rutas:
        try:
            if ruta.lower().endswith(".zip"):
                with zipfile.ZipFile(ruta) as z:
                    miembros=[n for n in z.namelist() if n.lower().endswith(".csv") and
                              "metadata" not in n.lower()]
                    for n in miembros:
                        raw=z.read(n)
                        for enc in ("utf-8-sig","utf-8","latin1"):
                            try:
                                d=pd.read_csv(io.BytesIO(raw),encoding=enc,low_memory=False)
                                if d.shape[1]>2: break
                            except Exception: d=None
                        if d is not None and d.shape[1]>2: break
            else:
                d=_leer_csv_flexible(ruta)
            cmap={str(c).strip().lower():c for c in d.columns}
            ccode=cmap.get("series code") or cmap.get("indicator code") or cmap.get("código de serie")
            cname=cmap.get("country name") or cmap.get("nombre del país")
            ciso=cmap.get("country code") or cmap.get("código del país")
            if ccode is None: continue
            yearcols=[]
            for c in d.columns:
                m=re.search(r"(19|20)\d{2}",str(c))
                if m: yearcols.append((c,int(m.group(0))))
            if not yearcols: continue
            ids=[ccode]+([cname] if cname else [])+([ciso] if ciso else [])
            q=d[ids+[c for c,_ in yearcols]].copy().rename(columns={c:str(y) for c,y in yearcols})
            q=q.melt(id_vars=ids,var_name="anio",value_name="valor")
            q["anio"]=pd.to_numeric(q["anio"],errors="coerce"); q["valor"]=pd.to_numeric(q["valor"],errors="coerce")
            q=q.dropna(subset=["anio","valor"]); q["anio"]=q["anio"].astype(int); q=q.rename(columns={ccode:"codigo"})
            if cname: q=q.rename(columns={cname:"pais"})
            if ciso: q=q.rename(columns={ciso:"iso"})
            return q,{"estado":"OK","fuente":os.path.basename(ruta)}
        except Exception as e: ultimo=e
    raise ValueError(f"WDI detectado pero no interpretable: {ultimo}")


def _wdi_global_por_codigo(wdi,codigo):
    if wdi.empty: return pd.Series(dtype=float)
    z=wdi[wdi["codigo"].astype(str)==str(codigo)].copy()
    return z.groupby("anio")["valor"].median().astype(float).sort_index() if len(z) else pd.Series(dtype=float)


def _series_wdi_multicausal():
    wdi,meta=_leer_wdi_largo()
    if wdi.empty: return {},meta
    eco={}
    for cod in ["NY.GDP.MKTP.KD.ZG","NY.GDP.PCAP.KD.ZG","FP.CPI.TOTL.ZG"]:
        s=_wdi_global_por_codigo(wdi,cod)
        if len(s):
            dis=pd.Series(index=s.index,dtype=float)
            for a in s.index:
                prev=s.loc[(s.index>=a-10)&(s.index<=a-1)].dropna()
                if len(prev)>=7: dis.loc[a]=abs(float(s.loc[a])-float(prev.median()))
            eco[cod]=_percentil_exante_min(dis)
    Peco=_combinar_canales_exante(eco)
    ecol={}
    for cod in ["AG.LND.FRST.ZS","EN.ATM.CO2E.PC","EN.ATM.PM25.MC.M3","ER.H2O.FWST.ZS"]:
        s=_wdi_global_por_codigo(wdi,cod)
        if len(s)>=3: ecol[cod]=_percentil_exante_min(s.diff().abs())
    Pecol=_combinar_canales_exante(ecol)
    tec={}
    for cod in ["IT.NET.USER.ZS","IT.CEL.SETS.P2","IP.PAT.RESD","GB.XPD.RSDV.GD.ZS",
                "SP.POP.SCIE.RD.P6","IP.JRN.ARTC.SC","IT.NET.SECR.P6"]:
        s=_wdi_global_por_codigo(wdi,cod)
        if len(s)>=3: tec[cod]=_percentil_exante_min(np.log1p(s.clip(lower=0)).diff().abs())
    return {"económica":Peco,"ecológica":Pecol,
            "tecnológica":_combinar_canales_exante(tec)},meta


def _who_archivos_mortalidad():
    """Reconoce los nombres reales que subimos, incluidos todos los chunks ICD10."""
    return _buscar_archivos_dacel(
        ["who_mortality_icd7","who_mortality_icd8","who_mortality_icd9",
         "who_mortality_icd10_sourcepart","who_mortality_icd10_add",
         "morticd07","morticd08","morticd09","morticd10"],
        extensiones={".csv",".xlsx",".xls"})


def _who_detectar_columnas(d):
    cmap={str(c).strip().lower():c for c in d.columns}
    year=next((cmap[k] for k in ("year","anio","año") if k in cmap),None)
    cause=next((cmap[k] for k in ("cause","causa","list","listcode") if k in cmap),None)
    deaths=next((cmap[k] for k in ("deaths1","deaths","death","muertes") if k in cmap),None)
    country=next((cmap[k] for k in ("country","countrycode","country_code") if k in cmap),None)
    sex=next((cmap[k] for k in ("sex",) if k in cmap),None)
    return year,cause,deaths,country,sex


def _who_poblacion_pais_anio():
    """Población WHO nacional por país-año. Prefiere ambos sexos (9); si no, suma 1+2."""
    ruta=_buscar_archivo_dacel(["who_population_live_births","pop"])
    if not ruta:
        return pd.DataFrame(columns=["pais","anio","poblacion"])
    d=_leer_tabla_dacel(ruta)
    cmap={str(c).strip().lower():c for c in d.columns}
    cy=cmap.get("year"); cp=cmap.get("country"); cs=cmap.get("sex"); cpop=cmap.get("pop1")
    if cy is None or cp is None or cpop is None:
        return pd.DataFrame(columns=["pais","anio","poblacion"])
    z=pd.DataFrame({
        "pais":d[cp].astype(str),
        "anio":_anio_numerico(d[cy]),
        "sexo":pd.to_numeric(d[cs],errors="coerce") if cs is not None else np.nan,
        "poblacion":pd.to_numeric(d[cpop],errors="coerce")
    }).dropna(subset=["anio","poblacion"])
    z=z[z["poblacion"]>0]; z["anio"]=z["anio"].astype(int)
    filas=[]
    for (pais,anio),g in z.groupby(["pais","anio"],sort=False):
        ambos=g.loc[g["sexo"]==9,"poblacion"]
        if len(ambos):
            p=float(ambos.median())
        else:
            porsexo=g[g["sexo"].isin([1,2])].groupby("sexo")["poblacion"].median()
            p=float(porsexo.sum()) if len(porsexo) else float(g["poblacion"].median())
        if np.isfinite(p) and p>0:
            filas.append((pais,anio,p))
    return pd.DataFrame(filas,columns=["pais","anio","poblacion"])


def _who_anual_desde_archivos():
    """
    Reconstruye mortalidad WHO sin sumar causas jerárquicas.
    Usa exclusivamente la fila agregada de TODAS LAS CAUSAS de cada lista WHO:
      07A/08A -> A000; 09A/09B/09N -> B00; 101 -> 1000; 103 -> B00; 104 -> A000.
    Trabaja a nivel país-año. Si existe Sex=9 (ambos sexos), lo prefiere;
    si no, suma hombres+ mujeres. Si un país-año aparece bajo más de una
    lista ICD, toma la mediana del total all-cause para no duplicarlo.
    """
    rutas=_who_archivos_mortalidad()
    if not rutas:
        return pd.DataFrame(), {"estado":"SIN DATOS","fuente":"WHO Mortality Database"}
    piezas=[]; usados=[]; errores=[]
    # Mapeo de listas agregadas WHO a su fila de todas las causas, verificado en los archivos fuente.
    mapa_total={"07A":"A000","08A":"A000","09A":"B00","09B":"B00","09N":"B00",
                "101":"1000","103":"B00","104":"A000"}
    for ruta in rutas:
        try:
            d=_leer_tabla_dacel(ruta)
            cmap={str(c).strip().lower():c for c in d.columns}
            year=cmap.get("year"); cause=cmap.get("cause"); deaths=cmap.get("deaths1")
            country=cmap.get("country"); sex=cmap.get("sex"); lista=cmap.get("list")
            admin1=cmap.get("admin1"); subdiv=cmap.get("subdiv")
            if year is None or cause is None or deaths is None or country is None:
                errores.append(f"{os.path.basename(ruta)}: faltan Year/Cause/Deaths1/Country"); continue

            listas=d[lista].astype(str).str.strip() if lista is not None else pd.Series("",index=d.index)
            causas=d[cause].astype(str).str.strip()
            esperado=listas.map(mapa_total)
            mask=causas.eq(esperado)
            # Solo registros nacionales cuando las columnas administrativas existen.
            if admin1 is not None:
                mask &= d[admin1].isna() | (d[admin1].astype(str).str.strip().isin(["","nan","0"]))
            if subdiv is not None:
                mask &= d[subdiv].isna() | (d[subdiv].astype(str).str.strip().isin(["","nan","0"]))

            q=d.loc[mask].copy()
            if q.empty:
                errores.append(f"{os.path.basename(ruta)}: sin registro all-cause"); continue
            z=pd.DataFrame({
                "pais":q[country].astype(str),
                "anio":_anio_numerico(q[year]),
                "lista":q[lista].astype(str) if lista is not None else "",
                "sexo":pd.to_numeric(q[sex],errors="coerce") if sex is not None else np.nan,
                "muertes":pd.to_numeric(q[deaths],errors="coerce")
            }).dropna(subset=["anio","muertes"])
            z=z[z["muertes"]>=0]; z["anio"]=z["anio"].astype(int)
            piezas.append(z); usados.append(os.path.basename(ruta))
        except Exception as e:
            errores.append(f"{os.path.basename(ruta)}: {e}")

    if not piezas:
        return pd.DataFrame(), {"estado":"ERROR","errores":errores}

    z=pd.concat(piezas,ignore_index=True).drop_duplicates()
    por_lista=[]
    for (pais,anio,lista),g in z.groupby(["pais","anio","lista"],sort=False):
        ambos=g.loc[g["sexo"]==9,"muertes"]
        if len(ambos):
            total=float(ambos.median())
        else:
            porsexo=g[g["sexo"].isin([1,2])].groupby("sexo")["muertes"].median()
            total=float(porsexo.sum()) if len(porsexo) else float(g["muertes"].median())
        if np.isfinite(total):
            por_lista.append((pais,anio,lista,total))
    cy=pd.DataFrame(por_lista,columns=["pais","anio","lista","muertes"])
    # Una sola observación por país-año aunque existan varias listas/revisiones.
    cy=cy.groupby(["pais","anio"],as_index=False)["muertes"].median()

    pob=_who_poblacion_pais_anio()
    if not pob.empty:
        cy=cy.merge(pob,on=["pais","anio"],how="left")
        cy["tasa"]=1e5*cy["muertes"]/cy["poblacion"]
    else:
        cy["tasa"]=np.nan

    # Magnitud de transformación biológica: cambio interanual dentro del mismo país.
    cy=cy.sort_values(["pais","anio"])
    prev_anio=cy.groupby("pais")["anio"].shift(1)
    prev_tasa=cy.groupby("pais")["tasa"].shift(1)
    consecutivo=(cy["anio"]-prev_anio)==1
    cy["cambio_tasa"]=np.where(
        consecutivo & (cy["tasa"]>0) & (prev_tasa>0),
        np.abs(np.log(cy["tasa"]/prev_tasa)), np.nan)

    anual=cy.groupby("anio").agg(
        cambio_tasa=("cambio_tasa","median"),
        cobertura_paises=("pais","nunique"),
        paises_con_cambio=("cambio_tasa","count")
    ).sort_index()

    return anual,{"estado":"OK","archivos":usados,"errores":errores,
                  "metodo":"all-cause por país-año; tasa por población; mediana del cambio log interanual",
                  "rango":[int(anual.index.min()),int(anual.index.max())]}


def _serie_biologica_who():
    """
    Señal biológica WHO auditada: magnitud del cambio interanual de la tasa
    de mortalidad por todas las causas dentro de cada país. No suma causas
    jerárquicas y no interpreta mortalidad como progreso.
    """
    anual,meta=_who_anual_desde_archivos()
    if anual.empty or "cambio_tasa" not in anual:
        return pd.Series(dtype=float),meta
    s=anual["cambio_tasa"].where(anual["paises_con_cambio"]>=3)
    p=_percentil_exante_min(s,min_hist=8).dropna()
    meta["nota"]="All-cause; cambios dentro de país; cobertura anual no se interpreta como mortalidad."
    return p,meta


def _h3v6_respuesta_global(df,h=3):
    """
    Respuesta común para la primera batería multicausal: distancia futura del estado
    civilizacional/tecnológico disponible. No usa P para construir R.
    """
    rr=_R_estado(df,["E","I","X","K","K_patentes","conexion"],h=h)
    if rr.empty: return pd.Series(dtype=float)
    return rr.set_index("anio")["R"].astype(float)


def _h3v6_evaluar_familia(P,R,min_n=10):
    """Asociación temporal preregistrada + nulo circular. Devuelve efecto e incertidumbre simple."""
    idx=P.dropna().index.intersection(R.dropna().index)
    if len(idx)<min_n:
        return {"estado":"SIN DATOS","n":int(len(idx))}
    x=P.loc[idx].astype(float); y=R.loc[idx].astype(float)
    stat=lambda a,b: float(stats.spearmanr(a,b).statistic)
    rho,p,nul=_p_circular(x.to_numpy(),y.to_numpy(),stat)
    # IC95 bootstrap descriptivo determinista por pares; NO decide el veredicto.
    rng=np.random.default_rng(20260916)
    boots=[]
    n=len(idx)
    for _ in range(2000):
        ii=rng.integers(0,n,n)
        r=stat(x.to_numpy()[ii],y.to_numpy()[ii])
        if np.isfinite(r): boots.append(r)
    lo,hi=(np.percentile(boots,[2.5,97.5]) if boots else (np.nan,np.nan))
    if rho>0 and p<ALFA:
        estado=VEREDICTO_OK
    else:
        estado=VEREDICTO_NO
    return {"estado":estado,"n":int(n),"desde":int(idx.min()),"hasta":int(idx.max()),
            "rho":float(rho),"p_circular":float(p),"ic95_rho":[float(lo),float(hi)],
            "n_placebos":int(len(nul))}


def prueba_h3_v6_multicausal(df,reporte,carpeta,etiqueta,h=3):
    """
    H3-v6: batería amplia. Cada familia se conserva separada.
    Esta versión NO reescribe H3-v4/v5 y NO combina familias para fabricar significancia.
    """
    reporte.append("## H3-v6 — Perturbación multicausal y reorganización\n")
    reporte.append(
        f"Especificación congelada: horizonte primario h={h} años. "
        "Cada familia de perturbación se evalúa por separado contra la magnitud futura "
        "de reorganización. H3-v4/v5 permanecen intactas. Los faltantes se reportan SIN DATOS.\n"
    )
    R=_h3v6_respuesta_global(df,h=h)
    familias={}; auditoria={}
    try:
        pg,mg=_serie_conflicto_ucdp()
        pb,mb=_serie_muertes_batalla()
        # Battle deaths es canal adicional solo en su cobertura; no recorta UCDP.
        if len(pg) and len(pb):
            pwar=_combinar_canales_exante({"conflicto":pg,"muertes":pb},min_canales=1)
        else:
            pwar=pg if len(pg) else pb
        familias["conflicto"]=pwar; auditoria["conflicto"]={"UCDP":mg,"battle_deaths":mb}
    except Exception as e:
        familias["conflicto"]=pd.Series(dtype=float); auditoria["conflicto"]={"estado":"ERROR","error":str(e)}

    try:
        ps,ms=_serie_sismos_usgs()
        familias["geofísica"]=ps; auditoria["geofísica"]=ms
    except Exception as e:
        familias["geofísica"]=pd.Series(dtype=float); auditoria["geofísica"]={"estado":"ERROR","error":str(e)}

    try:
        wb,mw=_series_wdi_multicausal()
        for k in ("económica","ecológica","tecnológica"):
            familias[k]=wb.get(k,pd.Series(dtype=float))
            auditoria[k]=mw
    except Exception as e:
        for k in ("económica","ecológica","tecnológica"):
            familias[k]=pd.Series(dtype=float); auditoria[k]={"estado":"ERROR","error":str(e)}

    pbi,mbi=_serie_biologica_who()
    familias["biológica"]=pbi; auditoria["biológica"]=mbi

    resultados={}
    reporte.append("### Auditoría de cobertura")
    if len(R):
        reporte.append(f"- R común h={h}: {len(R)} años ({int(R.index.min())}–{int(R.index.max())})")
    else:
        reporte.append("- R común: SIN DATOS")
    for nombre,P in familias.items():
        if len(P.dropna()):
            reporte.append(f"- {nombre}: {len(P.dropna())} años ({int(P.dropna().index.min())}–{int(P.dropna().index.max())})")
        else:
            reporte.append(f"- {nombre}: {auditoria.get(nombre,{}).get('estado','SIN DATOS')}")
        res=_h3v6_evaluar_familia(P,R)
        resultados[nombre]=res
        if res["estado"]=="SIN DATOS":
            reporte.append(f"  - resultado: **SIN DATOS SUFICIENTES** (intersección n={res['n']})")
        else:
            reporte.append(
                f"  - n={res['n']}; rho={res['rho']:+.3f}; IC95 descriptivo "
                f"[{res['ic95_rho'][0]:+.3f}, {res['ic95_rho'][1]:+.3f}]; "
                f"p circular={res['p_circular']:.4f}; **{res['estado']}**"
            )

    paquete={"version":"H3-v6","h_primario":h,"resultados":resultados,"auditoria":auditoria}
    with open(os.path.join(carpeta,"H3_v6_multicausal.json"),"w",encoding="utf-8") as f:
        json.dump(paquete,f,ensure_ascii=False,indent=2)
    return paquete


def prueba_h3_v7_replicacion(h3v6,reporte,carpeta):
    """
    H3-v7: síntesis transparente. NO declara universalidad por mayoría.
    Cuenta familias evaluables y conserva negativas/no evaluables.
    """
    reporte.append("## H3-v7 — Replicación entre familias independientes\n")
    r=h3v6.get("resultados",{})
    evaluables={k:v for k,v in r.items() if v.get("estado")!="SIN DATOS"}
    positivas={k:v for k,v in evaluables.items() if v.get("estado")==VEREDICTO_OK}
    negativas={k:v for k,v in evaluables.items() if v.get("estado")==VEREDICTO_NO}
    noeval=[k for k,v in r.items() if v.get("estado")=="SIN DATOS"]
    resumen={
        "familias_totales":len(r),
        "evaluables":len(evaluables),
        "con_senal":len(positivas),
        "sin_senal":len(negativas),
        "no_evaluables":len(noeval),
        "familias_con_senal":list(positivas),
        "familias_sin_senal":list(negativas),
        "familias_no_evaluables":noeval,
        "veredicto":"SÍNTESIS, NO VEREDICTO UNIVERSAL"
    }
    reporte.append(f"- Familias evaluables: {len(evaluables)}/{len(r)}")
    reporte.append(f"- Con señal según criterio preregistrado: {len(positivas)}/{len(r)}")
    reporte.append(f"- Sin señal: {len(negativas)}/{len(r)}")
    reporte.append(f"- No evaluables: {len(noeval)}/{len(r)}")
    reporte.append(
        "H3-v7 no convierte una mayoría en 'teoría demostrada'. Su función es mostrar "
        "replicación, heterogeneidad y límites sin ocultar resultados negativos.\n"
    )
    with open(os.path.join(carpeta,"H3_v7_replicacion.json"),"w",encoding="utf-8") as f:
        json.dump(resumen,f,ensure_ascii=False,indent=2)
    return resumen


def auditoria_datos_maestros(reporte,carpeta):
    """Inventario reproducible de archivos externos detectados por la corrida."""
    claves={
        "UCDP conflicto":["ucdpprioconflict","ucdp_prio","armedconflict"],
        "UCDP muertes":["battledeaths","battle_deaths"],
        "USGS sismos":["consulta.csv","query.csv","earthquake","usgs","sismo"],
        "WDI":["p_data_extract","data.csv","world_development","wdi"],
        "WHO población":["who_population_live_births"],
        "WHO códigos":["who_country_codes"],
        "WHO disponibilidad":["who_mortality_availability"],
        "WHO ICD":["who_mortality_icd","morticd10","morticd09","morticd08","morticd07"],
    }
    inv={}
    reporte.append("## Auditoría de datos maestros\n")
    for nombre,pats in claves.items():
        rutas=_buscar_archivos_dacel(pats)
        inv[nombre]=rutas
        if rutas:
            reporte.append(f"- {nombre}: OK — {len(rutas)} archivo(s)")
            for ruta in rutas:
                reporte.append(f"  - {ruta}")
        else:
            reporte.append(f"- {nombre}: NO DETECTADO")
    with open(os.path.join(carpeta,"auditoria_datos_maestros.json"),"w",encoding="utf-8") as f:
        json.dump(inv,f,ensure_ascii=False,indent=2)
    return inv


# ============================================================
# 4B. MOTOR DAÇEL MULTIESCALA v6 — formulación v1.2
# ============================================================

def _z_dacel(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce")
    sd = s.std(ddof=0)
    return (s-s.mean())/sd if np.isfinite(sd) and sd > 1e-12 else s*np.nan


def _comp_dacel(df, cols):
    cols=[c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan,index=df.index,dtype=float)
    return pd.concat([_z_dacel(df[c]).rename(c) for c in cols],axis=1).mean(axis=1,skipna=True)


def _R_estado(df, cols, h=3):
    """R_s(t→t+h)=d_s[x(t),x(t+h)]/h. Magnitud, no progreso."""
    cols=[c for c in cols if c in df.columns]
    if not cols: return pd.DataFrame(columns=['anio','R','dims'])
    z=pd.concat([_z_dacel(df[c]).rename(c) for c in cols],axis=1)
    years=df['anio'].astype(int).to_numpy(); rows=[]
    for i,a in enumerate(years):
        jj=np.where(years==a+h)[0]
        if not len(jj): continue
        j=jj[0]; x=z.iloc[i].to_numpy(float); y=z.iloc[j].to_numpy(float)
        ok=np.isfinite(x)&np.isfinite(y)
        minimo=max(1,min(2,len(cols)))
        if ok.sum()>=minimo:
            rows.append((a,float(np.sqrt(np.mean((y[ok]-x[ok])**2))/h),int(ok.sum())))
    return pd.DataFrame(rows,columns=['anio','R','dims'])


def _ajuste_oos_dacel(m, features, train_frac=.80):
    m=m[['anio','R']+features].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if len(m)<max(18,len(features)+6): return None,m
    cut=max(len(features)+3,int(np.floor(len(m)*train_frac))); cut=min(cut,len(m)-3)
    tr,te=m.iloc[:cut],m.iloc[cut:]
    X=np.column_stack([np.ones(len(tr)),tr[features].to_numpy(float)])
    b=np.linalg.lstsq(X,tr.R.to_numpy(float),rcond=None)[0]
    pred=np.column_stack([np.ones(len(te)),te[features].to_numpy(float)])@b
    const=np.repeat(tr.R.mean(),len(te))
    pers=m.R.shift(1).iloc[cut:].fillna(tr.R.iloc[-1]).to_numpy()
    rm=lambda a,p: float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(p,float))**2)))
    ed,ec,ep=rm(te.R,pred),rm(te.R,const),rm(te.R,pers); best=min(ec,ep)
    skill=1-ed/best if best>1e-12 else np.nan
    sst=float(np.sum((te.R-te.R.mean())**2)); r2=1-float(np.sum((te.R.to_numpy()-pred)**2))/sst if sst>1e-12 else np.nan
    return dict(b=b,ed=ed,ec=ec,ep=ep,skill=skill,r2=r2,n=len(m),train=len(tr),test=len(te)),m


def prueba_dacel_multiescala(df,reporte,carpeta,etiqueta):
    """Prueba directa de la arquitectura v1.2 sin reducir Daçel a una H3 binaria."""
    reporte.append("## Motor Daçel multiescala v6 — formulación v1.2\n")
    reporte.append("Núcleo: `R_s(t→t+h)=d_s[x(t),x(t+h)]/h`. R es magnitud de reorganización: expansión, contracción, deterioro, sustitución, adaptación o colapso cuentan como transformación.\n")
    reporte.append("Modelo: `R_s(t+h)=α+β1P+β2P²+β3E+β4(P×E)+β5IOE(t−1)+β6L+ε`. No se exige que β1 sea positivo: la respuesta puede ser no lineal y depender de recursos/restricciones.\n")
    resultados={}

    # Módulo civilizacional/tecnológico con los datos reales hoy cargados.
    try:
        w=df.copy()
        Pext = _presion_externa(df)
        w['P']=w['anio'].map(Pext)
        w['Ecap']=_z_dacel(w['E']) if 'E' in w else np.nan
        w['IOE']=_comp_dacel(w,['I','X','K','K_patentes','conexion'])
        w['IOEprev']=w['IOE'].shift(1)
        w['Lcap']=_z_dacel(w['L']) if 'L' in w and pd.to_numeric(w['L'],errors='coerce').notna().sum()>=10 else 0.0
        rr=_R_estado(w,['E','I','X','K','K_patentes','conexion'],h=3)
        w=w.merge(rr,on='anio',how='inner')
        w['P2']=w.P**2; w['PxE']=w.P*w.Ecap
        features=['P','P2','Ecap','PxE','IOEprev','Lcap']
        fit,m=_ajuste_oos_dacel(w,features)
        if fit is None:
            resultados['civilizacional_tecnologico']={'estado':'SIN DATOS','n':len(m)}
            reporte.append(f"### Civilizacional / tecnológico\n**SIN DATOS SUFICIENTES** para el modelo completo (n={len(m)}).\n")
        else:
            rho=float(m.P.rank().corr(m.R.rank()))
            coef=dict(zip(['intercept']+features,[float(x) for x in fit['b']]))
            estado='SUPERA BASELINE' if fit['skill']>0 else 'NO SUPERA BASELINE'
            resultados['civilizacional_tecnologico']={'estado':estado,'rho_P_R':rho,'skill':fit['skill'],'r2_oos':fit['r2'],'rmse_dacel':fit['ed'],'rmse_constante':fit['ec'],'rmse_persistencia':fit['ep'],'coeficientes':coef,'n':fit['n']}
            reporte.append("### Civilizacional / tecnológico\n")
            reporte.append(f"Años completos del modelo: {fit['n']} (entrenamiento {fit['train']}, prueba {fit['test']}). Spearman descriptivo P→R: rho={rho:+.3f}.\n")
            reporte.append(f"RMSE Daçel={fit['ed']:.4f}; constante={fit['ec']:.4f}; persistencia={fit['ep']:.4f}; skill frente al mejor baseline={100*fit['skill']:+.1f}%; R² fuera de muestra={fit['r2']:+.3f}.\n")
            reporte.append(f"**Resultado del módulo: {estado}.**\n")
    except Exception as e:
        resultados['civilizacional_tecnologico']={'estado':'SIN DATOS','error':str(e)}
        reporte.append(f"### Civilizacional / tecnológico\n**SIN DATOS / ERROR DE CONSTRUCCIÓN:** {e}\n")

    # Humano/cognitivo: transformación psicológica; X/conexión son perturbación/exposición.
    try:
        wh=df.copy(); rr=_R_estado(wh,['A','Dp'],h=3); wh=wh.merge(rr,on='anio',how='inner')
        wh['P']=_comp_dacel(wh,['X','conexion']); wh['P2']=wh.P**2; wh['Ecap']=0.0; wh['PxE']=0.0; wh['IOEprev']=0.0; wh['Lcap']=0.0
        fit,m=_ajuste_oos_dacel(wh,['P','P2','Ecap','PxE','IOEprev','Lcap'])
        if fit is None:
            resultados['humano_cognitivo']={'estado':'SIN DATOS','n':len(m)}
            reporte.append(f"### Humano / cognitivo\n**SIN DATOS SUFICIENTES** para el modelo completo (n={len(m)}). H2 se conserva como prueba específica independiente.\n")
        else:
            resultados['humano_cognitivo']={'estado':'EXPLORATORIO','n':fit['n'],'skill':fit['skill'],'r2_oos':fit['r2']}
            reporte.append(f"### Humano / cognitivo\nResultado exploratorio con n={fit['n']}; skill={100*fit['skill']:+.1f}%; R² OOS={fit['r2']:+.3f}. H2 permanece como prueba específica independiente.\n")
    except Exception as e:
        resultados['humano_cognitivo']={'estado':'SIN DATOS','error':str(e)}
        reporte.append(f"### Humano / cognitivo\n**SIN DATOS:** {e}\n")

    # Escalas adicionales: se informa la cobertura real ya cargada sin fabricar un "sí".
    try:
        pbi, mbi = _serie_biologica_who()
        resultados['biológico']={'estado':'DATOS DISPONIBLES' if len(pbi) else 'SIN DATOS',
                                 'n':int(len(pbi)), 'auditoria':mbi}
        reporte.append(f"### Biológico\nWHO aporta una serie de transformación biológica de {len(pbi)} años. Se integra en H3-v6 como familia independiente; no se interpreta mortalidad como progreso.\n")
    except Exception as e:
        resultados['biológico']={'estado':'SIN DATOS','error':str(e)}
    try:
        sw, mw = _series_wdi_multicausal()
        pe=sw.get('ecológica',pd.Series(dtype=float))
        resultados['ecológico']={'estado':'DATOS DISPONIBLES' if len(pe) else 'SIN DATOS',
                                 'n':int(len(pe)), 'auditoria':mw}
        reporte.append(f"### Ecológico\nWDI aporta {len(pe)} años de señal ecológica evaluable en H3-v6.\n")
    except Exception as e:
        resultados['ecológico']={'estado':'SIN DATOS','error':str(e)}
    resultados['físico / cosmológico']={'estado':'SIN DATOS'}
    resultados['artificial / IA']={'estado':'PARCIAL','nota':'La dimensión tecnológica/IA se evalúa con WDI y AI-GPR; una batería IA independiente más amplia queda abierta.'}
    reporte.append("### Físico / cosmológico\n**SIN DATOS específicos en esta corrida.** No se extrapola desde sismos terrestres al origen del universo.\n")
    reporte.append("### Artificial / IA\nCobertura parcial mediante WDI tecnológico y AI-GPR; se conserva como dominio abierto a nuevas series independientes.\n")

    reporte.append("### Lectura global\nEl motor no produce un único 'H3 sí/no' universal con una sola serie mundial. Evalúa la misma arquitectura matemática por dominio y conserva los resultados negativos. La universalidad de Daçel solo puede sostenerse si el patrón reaparece en baterías independientes.\n")
    with open(os.path.join(carpeta,'resultados_multiescala.json'),'w',encoding='utf-8') as f:
        json.dump(resultados,f,ensure_ascii=False,indent=2)
    return resultados


# ============================================================
# 5. ORQUESTADOR
# ============================================================

def _rango(df, cols):
    cols = [c for c in cols if c in df]
    sub = df[["anio"] + cols].dropna()
    return [int(sub["anio"].min()), int(sub["anio"].max())] if len(sub) else None


def ejecutar(df, etiqueta, carpeta, semilla=42):
    os.makedirs(carpeta, exist_ok=True)

    # Corridas Monte Carlo reproducibles e independientes por hipótesis.
    # Antes H2 y H3 compartían el mismo generador: si H2 no tenía datos,
    # H3 recibía una secuencia aleatoria distinta. SeedSequence evita ese
    # acoplamiento sin cambiar las hipótesis ni sus criterios.
    ss = np.random.SeedSequence(semilla)
    ss_h2, ss_h3 = ss.spawn(2)
    rng_h2 = np.random.default_rng(ss_h2)
    rng_h3 = np.random.default_rng(ss_h3)
    reporte = [f"# TEORÍA DAÇEL — Reporte de validación ({etiqueta})\n",
               f"Años en los datos: {int(df['anio'].min())}–{int(df['anio'].max())}\n",
               "Nivel de significancia: 5%. Cada hipótesis se enfrenta a un rival sin Daçel.\n"]

    prueba_colinealidad(df, reporte)
    cidi = calcular_cidi(df)
    hdf = calcular_hdf(df)
    pd.DataFrame({"CIDI": cidi}).join(pd.DataFrame({"HDF": hdf}), how="outer") \
        .rename_axis("anio").to_csv(os.path.join(carpeta, "indices_CIDI_HDF.csv"))

    veredictos = {
        "H1 (CIDI exponencial)": prueba_h1(cidi, reporte, carpeta, etiqueta),
        "H2 (vacío humano)": prueba_h2(df, reporte, carpeta, etiqueta, rng_h2),
        "H3-v4 histórica (auditoría)": prueba_h3(df, reporte, carpeta, etiqueta, rng_h3),
      "H3-v5 cobertura histórica": prueba_h3_v5_historica(df, reporte, carpeta, etiqueta),
        "H4 (externalización 2040)": prueba_h4(df, reporte),
        "Ecuación General": prueba_ecuacion_general(df, reporte, carpeta, etiqueta),
    }
    # Panorama nuevo: inventario -> H3-v6 multicausal -> H3-v7 replicación.
    # No altera ni borra los veredictos históricos v4/v5.
    auditoria_datos_maestros(reporte, carpeta)
    multiescala = prueba_dacel_multiescala(df, reporte, carpeta, etiqueta)
    h3v6 = prueba_h3_v6_multicausal(df, reporte, carpeta, etiqueta, h=3)
    h3v7 = prueba_h3_v7_replicacion(h3v6, reporte, carpeta)
    reporte.append("## Resumen\n")
    for k, v in veredictos.items():
        reporte.append(f"- {k}: **{v}**")
    reporte.append("\nUna teoría no se demuestra con una corrida. Se fortalece cada vez que "
                   "sobrevive a una prueba que pudo haberla refutado.\n")

    rango = lambda cols: _rango(df, cols)
    resumen = {
        "etiqueta": etiqueta,
        "fecha_ejecucion": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "veredictos": veredictos,
        "H3_v6_multicausal": h3v6,
        "H3_v7_replicacion": h3v7,
        "anios": {"CIDI": rango(["E", "I", "X", "K"]),
                  "vacio_humano": rango(["A", "conexion"]),
                  "perturbaciones": rango(["I", "P"])},
        "cidi": {str(int(k)): round(float(v), 4) for k, v in cidi.items()},
        "hdf": {str(int(k)): round(float(v), 4) for k, v in hdf.items()},
    }
    import json
    with open(os.path.join(carpeta, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    texto = "\n".join(reporte)
    with open(os.path.join(carpeta, "reporte.md"), "w", encoding="utf-8") as f:
        f.write(texto)
    return veredictos, texto


def main():
    ap = argparse.ArgumentParser(description="Validación empírica de la Teoría Daçel")
    ap.add_argument("--datos", help="CSV con datos reales (ver plantilla_datos.csv)")
    ap.add_argument("--demo", action="store_true", help="Correr con dos mundos sintéticos")
    ap.add_argument("--salida", default="resultados", help="Carpeta de resultados")
    a = ap.parse_args()

    if a.datos:
        df = pd.read_csv(a.datos).sort_values("anio").reset_index(drop=True)
        if "anio" not in df:
            sys.exit("El CSV necesita la columna 'anio'.")
        for c in ["E", "I", "X", "K", "A", "S", "Dp", "D", "T", "conexion", "P", "L"]:
            if c not in df:
                df[c] = np.nan
        df["P"] = df["P"].fillna(0)
        v, texto = ejecutar(df, "datos reales", a.salida)
        print(texto)
    elif a.demo:
        print("=" * 60)
        print("MODO DEMO — datos SINTÉTICOS, no reales. Sirven para comprobar")
        print("que el programa distingue un mundo con Daçel de uno sin Daçel.")
        print("=" * 60)
        v1, _ = ejecutar(generar_mundo(True, 7), "mundo sintético CON Daçel",
                         os.path.join(a.salida, "demo_con_dacel"))
        v2, _ = ejecutar(generar_mundo(False, 7), "mundo sintético SIN Daçel",
                         os.path.join(a.salida, "demo_sin_dacel"))
        print(f"\n{'Prueba':30s} {'CON Daçel':22s} {'SIN Daçel':22s}")
        for k in v1:
            print(f"{k:30s} {v1[k]:22s} {v2[k]:22s}")
        print(f"\nReportes y gráficas en: {os.path.abspath(a.salida)}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
