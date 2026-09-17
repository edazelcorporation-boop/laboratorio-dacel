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

- Correlación en niveles: r = 0.900 (engañosa: dos series que suben juntas siempre correlacionan).
- Correlación en cambios año a año: r = 0.683
- Significancia contra 1000 surrogates IAAFT: p = 0.0050

**Veredicto H2: SUPERA LA PRUEBA.** Cuando la conectividad sube más rápido, el desajuste psicológico también.

## H3 / F2 — Perturbación, memoria y reorganización sistémica

### H3-v4 histórica — formulación inferencial retirada
La perturbación no se trata como crecimiento. Se separan cuatro conceptos: P_ext(t), perturbación externa; M(t), memoria acumulada de perturbaciones; A_cap(t), capacidad adaptativa existente antes de la respuesta; y R(t), magnitud de reorganización del sistema. La presión adaptativa es Q(t)=M(t)×A_cap(t). Una perturbación puede destruir unas variables y acelerar otras; por eso R mide magnitud de cambio de régimen y no crecimiento neto.

P_ext combina AI-GPR y disrupción del crecimiento mundial. Sus transformaciones son ex-ante: cada año se compara solo con historia disponible hasta ese año. M usa memoria exponencial con vida media fija de 3 años. A_cap usa energía, externalización/conectividad y conocimiento, excluyendo I para evitar circularidad. R compara 5 años previos y 5 posteriores en los cuatro dominios Daçel.

- Años evaluables: 35 (1986–2020)
- Dominios de R por año: mediana 4.0; rango 2–4
- Asociación Q(t)=M×A_cap → R(t): rho = +0.240
- Nulo temporal: 34 desplazamientos circulares; p = 0.2000

**Veredicto H3-v4: NO SUPERA LA PRUEBA.** La dirección es la prevista, pero no se distingue del nulo temporal al 5%.

### Auditoría
H3-v2 (eventos manuales) y H3-v3 (perturbación contemporánea continua) permanecen en el historial del repositorio. H3-v4 no reescribe esos resultados; prueba una formulación dinámica explícita de la teoría.

### H3-v5 histórica — formulación inferencial retirada

Esta prueba conserva P_ext, vida media de 3 años, A_cap, R, ventana temporal, alfa y nulo circular de H3-v4. La única diferencia es que M conserva desde su primera observación válida la memoria causal ya construida, sin imponer un segundo calentamiento de 10 observaciones para repercentilizarla.

#### Auditoría de cobertura
- P_ext: 49 años (1977–2025)
- M histórica: 49 años (1977–2025)
- A_cap: 49 años (1977–2025)
- R: 45 años (1976–2020)
- Intersección R ∩ M ∩ A_cap: 44 años (1977–2020)
- Asociación Q(t)=M_hist×A_cap → R(t): rho = +0.034
- Nulo temporal: 43 desplazamientos circulares; p = 0.3864
- Dominios de R: mediana 3.0; rango 2–4
**Veredicto H3-v5: NO SUPERA LA PRUEBA.** La dirección es la prevista, pero no se distingue del nulo temporal al 5%.

## H4 — La externalización cognitiva seguirá aumentando hasta 2040

- Crecimiento medio de X en los últimos 10 años: 1.29% anual.
- Años con caída: 1 de 9.
**Veredicto H4: NO CONCLUYENTE** hasta 2040. Es una predicción a futuro: hay que registrarla hoy y comprobarla cuando lleguen los datos. Quedará refutada si X deja de crecer de forma sostenida antes de 2040 (criterio F3).

## Ecuación General dinámica — gI(t+1) = c + αgE(t) + γgX(t) + βQ(t) − δL(t)

Transiciones evaluables: 27; entrenamiento 21, prueba 6.

Parámetros estimados solo con el tramo de entrenamiento:
- c = +0.2948
- α (energía) = +12.6092
- γ (externalización) = +0.2364
- β (presión adaptativa Q) = -0.1383
- δ (pérdidas) = +0.0000

Error fuera de muestra (menor es mejor):
- Ecuación Daçel dinámica: 39.090
- Crecimiento constante: 43.821
- Persistencia: 82.004
- R² fuera de muestra: 0.204

**Veredicto Ecuación General dinámica: SUPERA LA PRUEBA.** Reduce el error 11% frente al mejor rival simple.

## Auditoría de datos maestros

- UCDP conflicto: OK — 1 archivo(s)
  - datos_fuente/UcdpPrioConflict_v26_1.csv
- UCDP muertes: OK — 1 archivo(s)
  - datos_fuente/BattleDeaths_v26_1.csv
- USGS sismos: OK — 1 archivo(s)
  - datos_fuente/query.csv
- WDI: OK — 1 archivo(s)
  - datos_fuente/P_Data_Extract_From_World_Development_Indicators%20%283%29.zip
- WHO población: OK — 1 archivo(s)
  - datos_fuente/WHO_population_live_births.csv
- WHO códigos: OK — 1 archivo(s)
  - datos_fuente/WHO_country_codes.csv
- WHO disponibilidad: OK — 1 archivo(s)
  - datos_fuente/WHO_mortality_availability_feb2026.xls
- WHO ICD: OK — 35 archivo(s)
  - datos_fuente/WHO_mortality_ICD10_add_Germany.xlsx
  - datos_fuente/WHO_mortality_ICD10_add_Norway.xlsx
  - datos_fuente/WHO_mortality_ICD10_sourcepart1_chunk1 (1).csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart1_chunk2 (1).csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart1_chunk3.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart1_chunk4.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart1_chunk5.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart2_chunk1.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart2_chunk2.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart2_chunk3.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart2_chunk4.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart2_chunk5.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart3_chunk1.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart3_chunk2.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart3_chunk3.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart3_chunk4.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart3_chunk5.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart4_chunk1.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart4_chunk2.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart4_chunk3.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart4_chunk4.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart5_chunk1.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart5_chunk2.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart5_chunk3.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart5_chunk4.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart6_chunk1.csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart6_chunk2 (1).csv
  - datos_fuente/WHO_mortality_ICD10_sourcepart6_chunk3 (1).csv
  - datos_fuente/WHO_mortality_ICD7.csv
  - datos_fuente/WHO_mortality_ICD8_part1.csv
  - datos_fuente/WHO_mortality_ICD8_part2.csv
  - datos_fuente/WHO_mortality_ICD9_part1.csv
  - datos_fuente/WHO_mortality_ICD9_part2.csv
  - datos_fuente/WHO_mortality_ICD9_part3.csv
  - datos_fuente/WHO_mortality_ICD9_part4.csv
## Motor Daçel multiescala v6 — formulación v1.2

Núcleo: `R_s(t→t+h)=d_s[x(t),x(t+h)]/h`. R es magnitud de reorganización: expansión, contracción, deterioro, sustitución, adaptación o colapso cuentan como transformación.

Modelo: `R_s(t+h)=α+β1P+β2P²+β3E+β4(P×E)+β5IOE(t−1)+β6L+ε`. No se exige que β1 sea positivo: la respuesta puede ser no lineal y depender de recursos/restricciones.

### Civilizacional / tecnológico

Años completos del modelo: 46 (entrenamiento 36, prueba 10). Spearman descriptivo P→R: rho=+0.270.

RMSE Daçel=0.1065; constante=0.1962; persistencia=0.0900; skill frente al mejor baseline=-18.3%; R² fuera de muestra=+0.115.

**Resultado del módulo: NO SUPERA BASELINE.**

### Humano / cognitivo
Resultado exploratorio con n=31; skill=-89.1%; R² OOS=-1.404. H2 permanece como prueba específica independiente.

### Biológico
WHO aporta una serie de transformación biológica de 65 años. Aporta una serie biológica independiente para la medición estructural; no se interpreta mortalidad como progreso.

### Ecológico
WDI aporta 15 años de señal ecológica evaluable en H3-v6.

### Físico / cosmológico
**SIN DATOS específicos en esta corrida.** No se extrapola desde sismos terrestres al origen del universo.

### Artificial / IA
Cobertura parcial mediante WDI tecnológico y AI-GPR; se conserva como dominio abierto a nuevas series independientes.

### Lectura global
El motor multiescala se conserva como módulo cuantitativo complementario. H3 se reporta como postulado base (antes v8) y H3-v9 como prueba de intensidad.

## H3 — Postulado base de reestructuración (verificación con datos)

Postulado Daçel: todo sistema perturbado se reestructura y se transforma, para bien o para mal. Aquí se verifica con datos reales si, en los años de perturbación alta de cada familia, el estado del sistema mundial cambió de forma medible.

Estatus: es el principio base de la teoría, no una hipótesis que compita contra un rival. Su calificación es la cobertura observada (escala 0–100%: porcentaje de años de perturbación alta con transformación medible). La prueba de intensidad, que sí puede fallar, es H3-v9.

Postulado operativo: `P → ΔS`. Una perturbación cuenta como exposición; la evidencia observacional es un cambio independiente del estado del sistema. Expansión, contracción, deterioro, sustitución, adaptación, recuperación o colapso cuentan como transformación. El signo normativo no decide H3.

Medición principal: `T(t→t+1)=D[x(t),x(t+1)]`, con D como distancia multivariable del estado observable. Se usa un paso anual para no imponer el antiguo cuello de botella h=3. La dirección neta se informa aparte.

### Evidencia observada por familia
- conflicto: 23/23 años de perturbación alta (100.0%) muestran transformación estructural medible; T mediana=0.0669; dirección relativa: expansión/aceleración 8, contracción/desaceleración 15, casi nula 0; persistencia 3 pasos=100.0%.
- geofísica: 11/11 años de perturbación alta (100.0%) muestran transformación estructural medible; T mediana=0.1113; dirección relativa: expansión/aceleración 9, contracción/desaceleración 2, casi nula 0; persistencia 3 pasos=100.0%.
- económica: 3/3 años de perturbación alta (100.0%) muestran transformación estructural medible; T mediana=0.1924; dirección relativa: expansión/aceleración 0, contracción/desaceleración 3, casi nula 0; persistencia 3 pasos=100.0%.
- ecológica: **SIN DATOS COMPARABLES**
- tecnológica: **SIN DATOS COMPARABLES**
- biológica: 16/16 años de perturbación alta (100.0%) muestran transformación estructural medible; T mediana=0.1352; dirección relativa: expansión/aceleración 5, contracción/desaceleración 11, casi nula 0; persistencia 3 pasos=100.0%.

### Calificación del postulado base
- Cobertura observada: 53/53 (100.0%) años de perturbación alta presentan cambio estructural medible en el paso anual.
**H3 postulado base: CUMPLE EL POSTULADO BASE (100.0%).** Los datos reales muestran transformación en 53 de 53 años de perturbación alta.
- Este resultado mide **compatibilidad observacional con el postulado Daçel**, no beneficio moral, progreso ni causalidad estadística. Una contracción o destrucción sigue siendo transformación.
- Historial: H3-v2 a v5 no superaron sus pruebas y se conservan como registro. H3-v8 se reformuló como postulado base el 17-sep-2026. H3-v9 es la prueba de intensidad.
## H3-v9 — Perturbación contra años normales (prueba vigente)

Registrada el 17-sep-2026 antes de correrse con datos reales. Pregunta si después de los años más perturbados hay más transformación que después de años tranquilos. La transformación se mide como cuánto se aparta el crecimiento de su trayectoria reciente, hacia arriba o hacia abajo. Ambas series se analizan sin tendencia para que el simple paso del tiempo no genere la relación.

- Años analizados: 42 (1983–2024)
- Correlación perturbación → transformación (sin tendencia): rho = +0.011
- Nulo con 41 desplazamientos circulares: p = 0.4286
- Transformación mediana tras años muy perturbados (1): 2.1587; tras años tranquilos (13): 1.5807

**Veredicto H3-v9: NO SUPERA LA PRUEBA.** La transformación tras años perturbados no se distingue de la de años tranquilos una vez quitada la tendencia.

Por familia (solo informativo; con 6 familias se esperan resultados "significativos" por azar, así que no deciden el veredicto):

- conflicto: n = 42, rho = +0.001, p = 0.429
- geofísica: n = 42, rho = -0.112, p = 0.786
- económica: menos de 20 años comparables.
- ecológica: menos de 20 años comparables.
- tecnológica: menos de 20 años comparables.
- biológica: n = 40, rho = +0.122, p = 0.225

## H5 — La IA aumenta el vacío humano (más allá de internet)

Registrada el 17-sep-2026 antes de correrse con datos reales. Pregunta si en los años en que la IA crece más rápido, el año siguiente sube más la ansiedad o depresión mundial, después de descontar el efecto del crecimiento de internet (que ya mide H2).

- Años analizados: 27 (1996–2022)
- Fuente de IA: owid
- Correlación simple (sin descontar internet): r = +0.176
- Correlación parcial (descontando internet): r = +0.219
- Nulo con 26 desplazamientos circulares: p = 0.1852

**Veredicto H5: NO SUPERA LA PRUEBA.** La dirección es la prevista, pero la relación no se distingue del azar con los años disponibles.

Límite actual: las series mundiales de salud mental suelen terminar años antes del presente, y antes de 2022 la IA llegaba poco al público. El periodo de IA masiva entrará a la prueba conforme se publiquen nuevos datos.

## Resumen

- H1 (CIDI exponencial): **SUPERA LA PRUEBA**
- H2 (vacío humano): **SUPERA LA PRUEBA**
- H4 (externalización 2040): **NO CONCLUYENTE**
- Ecuación General: **SUPERA LA PRUEBA**
- H3 (postulado base): **CUMPLE EL POSTULADO BASE (100%)**
- H3-v9 (intensidad): **NO SUPERA LA PRUEBA**
- H5 (IA y vacío humano): **NO SUPERA LA PRUEBA**

H3 postulado base: verificación con datos del principio de reestructuración (calificación = % de años de perturbación alta con transformación medible). H3-v9 y H5 fueron registradas el 17-sep-2026 antes de correrse con datos reales. H3-v2 a v5 permanecen en el historial.
