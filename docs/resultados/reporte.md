# TEORÍA DAÇEL — Reporte de validación (datos reales)

Años en los datos: 1960–2025

Nivel de significancia: 5%. Cada hipótesis se enfrenta a un rival sin Daçel.

## Control previo — Colinealidad temporal

- PC1 en niveles explica 97.1% de la varianza; su correlación con el año es |r| = 0.982.
- PC1 en tasas de crecimiento explica 86.7%.
- ADVERTENCIA: en niveles, el PCA está midiendo básicamente el paso del tiempo. Por eso todas las pruebas causales de este programa trabajan con tasas de cambio (series sin tendencia), no con niveles.

## H1 — El CIDI crece exponencialmente

Escala logarítmica. Entrenamiento: primer 80% de los años; validación fuera de muestra: último 20%.

     modelo      AIC  RMSE_fuera
   gompertz -104.713       0.085
  logistico  -95.682       0.313
exponencial  -54.772       0.651
     lineal  -17.202       0.969

Tabla descriptiva. El veredicto no depende de ella, porque con series tan largas un logístico con techo lejano es casi idéntico a un exponencial.

- Crecimiento medio del CIDI: 23.27% anual (p = 0.0024)
- Cambio de esa tasa por década: -14.408 pp (p de descenso = 0.0739)

**Veredicto H1: SUPERA LA PRUEBA.** El CIDI crece a una tasa positiva que no muestra desaceleración: comportamiento exponencial.
Nota: H1 es una hipótesis débil. Casi cualquier medida tecnológica crece así, con o sin Daçel. Superarla no distingue a Daçel de otras teorías.

## H2 / F4 — El vacío humano aumenta con la hiperconectividad

Se usa solo la parte psicológica del HDF (ansiedad, soledad y/o depresión). Incluir D y T sería circular, porque ya son medidas de conectividad.

- Correlación en niveles: r = 0.893 (engañosa: dos series que suben juntas siempre correlacionan).
- Correlación en cambios año a año: r = 0.650
- Significancia contra 1000 surrogates IAAFT: p = 0.0430

**Veredicto H2: SUPERA LA PRUEBA.** Cuando la conectividad sube más rápido, el desajuste psicológico también.

## H3 / F2 — Las perturbaciones aceleran la reorganización

Estudio de eventos: crecimiento de I en los 5 años posteriores menos los 5 años anteriores, comparado con 2000 conjuntos de años al azar (placebo).

- Eventos analizados: 3
- Aceleración observada: +7.02 puntos porcentuales de crecimiento anual
- Aceleración típica en años al azar: -6.51 pp
- p = 0.0960

**Veredicto H3: NO SUPERA LA PRUEBA.** Las perturbaciones no muestran una aceleración distinta a la de años al azar.

## H4 — La externalización cognitiva seguirá aumentando hasta 2040

- Crecimiento medio de X en los últimos 10 años: 1.29% anual.
- Años con caída: 1 de 9.
**Veredicto H4: NO CONCLUYENTE** hasta 2040. Es una predicción a futuro: hay que registrarla hoy y comprobarla cuando lleguen los datos. Quedará refutada si X deja de crecer de forma sostenida antes de 2040 (criterio F3).

## Ecuación General — dI/dt = αE + βP + γX − δL

Parámetros estimados con el 80% inicial de los años:

- c = -0.1201
- α (energía) = +15.4624
- β (perturbación) = +0.4191
- γ (externalización) = +1.3345

Error de predicción en el 20% final (menor es mejor):

- Ecuación Daçel:        98.191
- Crecimiento constante: 47.832
- Persistencia:          88.535
- R² fuera de muestra de Daçel: -3.214

**Veredicto Ecuación General: NO SUPERA LA PRUEBA.** Un modelo sin Daçel predice igual o mejor.

## Resumen

- H1 (CIDI exponencial): **SUPERA LA PRUEBA**
- H2 (vacío humano): **SUPERA LA PRUEBA**
- H3 (perturbaciones): **NO SUPERA LA PRUEBA**
- H4 (externalización 2040): **NO CONCLUYENTE**
- Ecuación General: **NO SUPERA LA PRUEBA**

Una teoría no se demuestra con una corrida. Se fortalece cada vez que sobrevive a una prueba que pudo haberla refutado.
