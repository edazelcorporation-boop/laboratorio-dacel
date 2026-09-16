#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEORÍA DAÇEL — Programa de validación empírica (v2.0)
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
import os
import sys
import warnings

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


def prueba_h3(df, reporte, carpeta, etiqueta, rng, ventana=5, n_perm=5000):
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
    Ecuación General Daçel, en forma de tasas para evitar la colinealidad:
        g_I(t) = c + α·g_E(t) + β·P̃(t) + γ·g_X(t) − δ·L(t)
    donde P̃(t) = 1 si hubo una perturbación en los 5 años previos (efecto que dura).
    Rivales: (a) crecimiento constante, (b) persistencia g_I(t) = g_I(t−1).
    Daçel debe predecir MEJOR que ambos en años que no vio.
    """
    reporte.append("## Ecuación General — dI/dt = αE + βP + γX − δL\n")
    cols = ["anio", "I", "E", "X", "P"] + (["L"] if "L" in df and df["L"].notna().any() else [])
    sub = df[cols].dropna().reset_index(drop=True)
    if len(sub) < 27:
        reporte.append(f"**Veredicto Ecuación General: {VEREDICTO_NC}.** Solo {len(sub)} años completos; "
                       "se necesitan al menos 27 para validar fuera de muestra con un mínimo de 5 años.\n")
        return VEREDICTO_NC
    gI = np.diff(np.log(sub["I"].values))
    gE = np.diff(np.log(sub["E"].values))
    gX = np.diff(np.log(sub["X"].values))
    anios_s = sub["anio"].values
    ev = anios_s[sub["P"].values > 0]
    P_vent = np.array([float(np.any((ev < a) & (ev >= a - 5))) for a in anios_s[1:]])
    columnas = [np.ones_like(gI), gE, P_vent, gX]
    nombres = ["c", "α (energía)", "β (perturbación)", "γ (externalización)"]
    if "L" in sub:
        columnas.append(-sub["L"].values[1:]); nombres.append("δ (pérdidas)")
    M = np.column_stack(columnas)

    # quitamos el primer año para poder usar la persistencia
    y, M = gI[1:], M[1:]
    persistencia = gI[:-1]
    n = len(y); n_ent = int(n * 0.8)

    coef, *_ = np.linalg.lstsq(M[:n_ent], y[:n_ent], rcond=None)
    pred_dacel = M[n_ent:] @ coef
    pred_const = np.full(n - n_ent, y[:n_ent].mean())
    pred_pers = persistencia[n_ent:]
    e_dacel = rmse(y[n_ent:], pred_dacel)
    e_const = rmse(y[n_ent:], pred_const)
    e_pers = rmse(y[n_ent:], pred_pers)
    ss_tot = np.sum((y[n_ent:] - y[:n_ent].mean()) ** 2)
    r2_fuera = 1 - np.sum((y[n_ent:] - pred_dacel) ** 2) / ss_tot

    reporte.append("Parámetros estimados con el 80% inicial de los años:\n")
    for nm, cf in zip(nombres, coef):
        reporte.append(f"- {nm} = {cf:+.4f}")
    reporte.append("\nError de predicción en el 20% final (menor es mejor):\n")
    reporte.append(f"- Ecuación Daçel:        {e_dacel*100:.3f}")
    reporte.append(f"- Crecimiento constante: {e_const*100:.3f}")
    reporte.append(f"- Persistencia:          {e_pers*100:.3f}")
    reporte.append(f"- R² fuera de muestra de Daçel: {r2_fuera:.3f}")
    mejora = 1 - e_dacel / min(e_const, e_pers)
    if mejora > 0.10:
        v, txt = VEREDICTO_OK, f"La ecuación reduce el error {mejora*100:.0f}% respecto al mejor rival."
    elif mejora > 0:
        v, txt = VEREDICTO_NC, f"Mejora solo {mejora*100:.0f}% respecto al mejor rival; no basta para afirmar nada."
    else:
        v, txt = VEREDICTO_NO, "Un modelo sin Daçel predice igual o mejor."
    reporte.append(f"\n**Veredicto Ecuación General: {v}.** {txt}\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    anios = sub["anio"].values[2:]
    ax.plot(anios, y * 100, "k-", lw=1, label="real")
    ax.plot(anios[n_ent:], pred_dacel * 100, "r-", label="Daçel")
    ax.plot(anios[n_ent:], pred_const * 100, "b--", label="constante")
    ax.plot(anios[n_ent:], pred_pers * 100, "g:", label="persistencia")
    ax.set_title(f"Ecuación General — crecimiento de I ({etiqueta})")
    ax.set_xlabel("Año"); ax.set_ylabel("% anual"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(carpeta, "Ecuacion_General.png"), dpi=130); plt.close(fig)
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
        "H3 (perturbaciones)": prueba_h3(df, reporte, carpeta, etiqueta, rng_h3),
        "H4 (externalización 2040)": prueba_h4(df, reporte),
        "Ecuación General": prueba_ecuacion_general(df, reporte, carpeta, etiqueta),
    }
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
