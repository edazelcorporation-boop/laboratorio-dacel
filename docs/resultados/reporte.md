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
- Significancia contra 1000 surrogates IAAFT: p = 0.0310

**Veredicto H2: SUPERA LA PRUEBA.** Cuando la conectividad sube más rápido, el desajuste psicológico también.

## H3 / F2 — Las perturbaciones intensifican la reorganización sistémica

### H3-v3 principal — prueba continua, sin selección manual de acontecimientos
La perturbación externa S(t) combina con igual peso dos dimensiones independientes del Laboratorio Daçel: (1) AI-GPR (Iacoviello y Tong), riesgo geopolítico mensual convertido a promedio anual con años completos, y (2) disrupción macroeconómica, medida como la desviación absoluta del crecimiento anual del PIB mundial respecto de su mediana histórica (Banco Mundial). Cada dimensión se transforma a rango percentil antes de promediarse.

La reorganización R(t) se calcula para cada año posible como la magnitud del cambio de régimen entre los 5 años anteriores y los 5 posteriores en los dominios energía, información, externalización/conectividad y conocimiento. Cada dominio se normaliza por su propia historia y se requieren al menos 2 dominios. Ansiedad y depresión quedan fuera porque pertenecen a H2.

Criterio fijado antes de ejecutar esta versión: correlación de Spearman positiva entre S(t) y R(t), contrastada contra todos los desplazamientos circulares no nulos de S(t). Este placebo conserva la estructura temporal de la perturbación y elimina la alineación concreta con la reorganización. Nivel de significancia: 5%.

- Años evaluables: 45 (1976–2020)
- Dominios disponibles por año: mediana 3.0; rango 2–4
- Asociación S(t) → reorganización: rho = +0.003
- Placebo temporal: 44 desplazamientos circulares; p = 0.4889

**Veredicto H3: NO SUPERA LA PRUEBA.** La asociación observada es positiva, pero no alcanza el umbral preregistrado del 5%; la evidencia continua disponible no basta para distinguirla del placebo.

### H3-v2 diagnóstica — prueba por acontecimientos previamente usada
La versión por eventos se conserva en el historial del repositorio para auditoría. No interviene en el veredicto de H3-v3, que usa todos los años evaluables y un índice externo continuo.

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
