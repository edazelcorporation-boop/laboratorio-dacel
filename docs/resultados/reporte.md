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

## H3 / F2 — Principio de transformación por perturbación

### H3-v5 principal — desviación de trayectoria, sin signo obligatorio
Daçel no exige que una perturbación produzca progreso. La respuesta puede ser expansión, contracción, deterioro, sustitución, diversificación, adaptación o colapso. Por ello H3-v5 pregunta si una perturbación externa suficientemente intensa hace que el sistema se aparte de la trayectoria que habría sido esperable a partir de su propia historia previa.

La transformación T(t) se mide durante 3 años como el error absoluto estandarizado entre la trayectoria observada y una proyección contrafactual simple construida SOLO con los 10 años anteriores. Se promedian los dominios energía, información, externalización/conectividad y conocimiento, requiriendo al menos dos. El signo se conserva solo para diagnóstico: una caída fuerte y un aumento fuerte son ambos transformación. P_ext usa AI-GPR + disrupción macroeconómica ex-ante; M(t) conserva la memoria exponencial preregistrada de 3 años.

**Alcance:** con las fuentes actuales esta corrida prueba el mecanismo en el sistema histórico macro-tecnológico mundial. No se presenta como demostración universal en biología, ecología, individuos o astrofísica; esos dominios requieren baterías de datos independientes con el mismo criterio matemático.

- Años evaluables: 38 (1986–2023)
- Dominios de transformación por año: mediana 4.0; rango 2–4
- Asociación M(t) → T(t): rho = +0.305
- Nulo temporal: 37 desplazamientos circulares; p = 0.1579

**Veredicto H3-v5: NO SUPERA LA PRUEBA.** La dirección es la prevista, pero la evidencia disponible no permite distinguirla del nulo temporal al 5%.

### Diagnóstico de capacidad adaptativa
Q(t)=M×A_cap frente a T(t): rho = +0.293. Este valor NO decide H3-v5: A_cap puede modificar el tipo de respuesta, pero una transformación destructiva también cuenta como transformación Daçel.

### Falsación
H3-v5 queda contradicha en esta batería si perturbaciones mayores no producen desviaciones de trayectoria mayores que el nulo temporal. Un resultado negativo se conserva; no se cambia el signo, horizonte ni definición después de observarlo.
### Auditoría
H3-v2, H3-v3 y H3-v4 permanecen en el historial del repositorio. H3-v5 cambia la operacionalización porque la definición teórica se aclaró ANTES de esta corrida: perturbación implica transformación posible en cualquier dirección, no progreso.

## H4 — La externalización cognitiva seguirá aumentando hasta 2040

- Crecimiento medio de X en los últimos 10 años: 1.29% anual.
- Años con caída: 1 de 9.
**Veredicto H4: NO CONCLUYENTE** hasta 2040. Es una predicción a futuro: hay que registrarla hoy y comprobarla cuando lleguen los datos. Quedará refutada si X deja de crecer de forma sostenida antes de 2040 (criterio F3).

## Ecuación General dinámica — gI(t+1) = c + αgE(t) + γgX(t) + βQ(t) − δL(t)

Transiciones evaluables: 27; entrenamiento 21, prueba 6.

Parámetros estimados solo con el tramo de entrenamiento:
- c = +0.2962
- α (energía) = +12.6168
- γ (externalización) = +0.2360
- β (presión adaptativa Q) = -0.1408
- δ (pérdidas) = +0.0000

Error fuera de muestra (menor es mejor):
- Ecuación Daçel dinámica: 39.076
- Crecimiento constante: 43.821
- Persistencia: 82.004
- R² fuera de muestra: 0.205

**Veredicto Ecuación General dinámica: SUPERA LA PRUEBA.** Reduce el error 11% frente al mejor rival simple.

## Resumen

- H1 (CIDI exponencial): **SUPERA LA PRUEBA**
- H2 (vacío humano): **SUPERA LA PRUEBA**
- H3 (transformación por perturbación): **NO SUPERA LA PRUEBA**
- H4 (externalización 2040): **NO CONCLUYENTE**
- Ecuación General: **SUPERA LA PRUEBA**

Una teoría no se demuestra con una corrida. Se fortalece cada vez que sobrevive a una prueba que pudo haberla refutado.
