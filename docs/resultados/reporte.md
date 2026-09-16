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

## H3 / F2 — Perturbación, memoria y reorganización sistémica

### H3-v4 principal — mecanismo dinámico Daçel
La perturbación no se trata como crecimiento. Se separan cuatro conceptos: P_ext(t), perturbación externa; M(t), memoria acumulada de perturbaciones; A_cap(t), capacidad adaptativa existente antes de la respuesta; y R(t), magnitud de reorganización del sistema. La presión adaptativa es Q(t)=M(t)×A_cap(t). Una perturbación puede destruir unas variables y acelerar otras; por eso R mide magnitud de cambio de régimen y no crecimiento neto.

P_ext combina AI-GPR y disrupción del crecimiento mundial. Sus transformaciones son ex-ante: cada año se compara solo con historia disponible hasta ese año. M usa memoria exponencial con vida media fija de 3 años. A_cap usa energía, externalización/conectividad y conocimiento, excluyendo I para evitar circularidad. R compara 5 años previos y 5 posteriores en los cuatro dominios Daçel.

- Años evaluables: 35 (1986–2020)
- Dominios de R por año: mediana 4.0; rango 2–4
- Asociación Q(t)=M×A_cap → R(t): rho = +0.331
- Nulo temporal: 34 desplazamientos circulares; p = 0.1714

**Veredicto H3-v4: NO SUPERA LA PRUEBA.** La dirección es la prevista, pero no se distingue del nulo temporal al 5%.

### Auditoría
H3-v2 (eventos manuales) y H3-v3 (perturbación contemporánea continua) permanecen en el historial del repositorio. H3-v4 no reescribe esos resultados; prueba una formulación dinámica explícita de la teoría.

### H3-v5 — cobertura histórica causal

Esta prueba conserva P_ext, vida media de 3 años, A_cap, R, ventana temporal, alfa y nulo circular de H3-v4. La única diferencia es que M conserva desde su primera observación válida la memoria causal ya construida, sin imponer un segundo calentamiento de 10 observaciones para repercentilizarla.

#### Auditoría de cobertura
- P_ext: 49 años (1977–2025)
- M histórica: 49 años (1977–2025)
- A_cap: 49 años (1977–2025)
- R: 45 años (1976–2020)
- Intersección R ∩ M ∩ A_cap: 44 años (1977–2020)
- Asociación Q(t)=M_hist×A_cap → R(t): rho = +0.189
- Nulo temporal: 43 desplazamientos circulares; p = 0.2500
- Dominios de R: mediana 3.0; rango 2–4
**Veredicto H3-v5: NO SUPERA LA PRUEBA.** La dirección es la prevista, pero no se distingue del nulo temporal al 5%.

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

## Motor Daçel multiescala v6 — formulación v1.2

Núcleo: `R_s(t→t+h)=d_s[x(t),x(t+h)]/h`. R es magnitud de reorganización: expansión, contracción, deterioro, sustitución, adaptación o colapso cuentan como transformación.

Modelo: `R_s(t+h)=α+β1P+β2P²+β3E+β4(P×E)+β5IOE(t−1)+β6L+ε`. No se exige que β1 sea positivo: la respuesta puede ser no lineal y depender de recursos/restricciones.

### Civilizacional / tecnológico

Años completos del modelo: 46 (entrenamiento 36, prueba 10). Spearman descriptivo P→R: rho=+0.263.

RMSE Daçel=0.0911; constante=0.1912; persistencia=0.0977; skill frente al mejor baseline=+6.7%; R² fuera de muestra=+0.247.

**Resultado del módulo: SUPERA BASELINE.**

### Humano / cognitivo
Resultado exploratorio con n=26; skill=-70.6%; R² OOS=-1.483. H2 permanece como prueba específica independiente.

### Biológico / ecológico
**SIN DATOS.** Esta corrida no fabrica proxies. Requiere una batería independiente de estados, perturbaciones, recursos y restricciones del dominio.

### Físico / cosmológico
**SIN DATOS.** Esta corrida no fabrica proxies. Requiere una batería independiente de estados, perturbaciones, recursos y restricciones del dominio.

### Artificial / ia
**SIN DATOS.** Esta corrida no fabrica proxies. Requiere una batería independiente de estados, perturbaciones, recursos y restricciones del dominio.

### Lectura global
El motor no produce un único 'H3 sí/no' universal con una sola serie mundial. Evalúa la misma arquitectura matemática por dominio y conserva los resultados negativos. La universalidad de Daçel solo puede sostenerse si el patrón reaparece en baterías independientes.

## Resumen

- H1 (CIDI exponencial): **SUPERA LA PRUEBA**
- H2 (vacío humano): **SUPERA LA PRUEBA**
- H3-v4 histórica (auditoría): **NO SUPERA LA PRUEBA**
- H3-v5 cobertura histórica: **NO SUPERA LA PRUEBA**
- H4 (externalización 2040): **NO CONCLUYENTE**
- Ecuación General: **SUPERA LA PRUEBA**

Una teoría no se demuestra con una corrida. Se fortalece cada vez que sobrevive a una prueba que pudo haberla refutado.
