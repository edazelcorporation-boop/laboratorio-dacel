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

Prueba principal ampliada y preregistrada: la reorganización se mide en los dominios estructurales de Daçel — energía (E), información procesada (I), externalización/conectividad (telefonía fija, móvil e internet, contadas como un solo dominio) y conocimiento (patentes y artículos científicos, contados como un solo dominio). Ansiedad y depresión no entran aquí porque pertenecen a H2.

Se conserva la ventana original de ±5 años. Cada dominio compara su régimen de crecimiento posterior con el anterior, estandarizado por su variabilidad histórica. La prueba principal usa la magnitud absoluta del cambio: una crisis puede reorganizar el sistema haciendo subir unos dominios y caer otros, por lo que promediar signos opuestos cancelaría precisamente la reconfiguración que H3 intenta medir.

Perturbaciones evaluables y evidencia usada:
- 1987: 2 dominios [energía: E; externalización/conectividad: X_fija,X] → intensidad 0.186 DE; dirección neta -0.186 DE (energía=-0.21, externalización/conectividad=-0.16)
- 1991: 3 dominios [energía: E; externalización/conectividad: X_fija,X; conocimiento: K_patentes] → intensidad 0.313 DE; dirección neta -0.027 DE (energía=-0.51, externalización/conectividad=+0.01, conocimiento=+0.42)
- 2001: 4 dominios [energía: E; información: I; externalización/conectividad: X_fija; conocimiento: K_patentes] → intensidad 0.446 DE; dirección neta +0.009 DE (energía=+0.79, información=+0.12, externalización/conectividad=-0.73, conocimiento=-0.14)
- 2008: 4 dominios [energía: E; información: I; externalización/conectividad: X_fija; conocimiento: K_patentes,K] → intensidad 0.688 DE; dirección neta -0.515 DE (energía=-0.88, información=+0.35, externalización/conectividad=-1.23, conocimiento=-0.29)
- 2020: 2 dominios [información: I; externalización/conectividad: X_fija,X,conexion] → intensidad 0.101 DE; dirección neta -0.101 DE (información=-0.03, externalización/conectividad=-0.17)

Resultado sistémico: promedio de 5 perturbaciones evaluables frente a 5000 panoramas placebo emparejados por disponibilidad de evidencia.
- Intensidad de reorganización observada: 0.347 desviaciones estándar
- Intensidad placebo media: 0.530 DE
- Dirección neta observada (diagnóstica): -0.164 DE
- p = 0.9620

**Veredicto H3: NO SUPERA LA PRUEBA.** La evidencia sistémica disponible no distingue la magnitud de reorganización posterior a perturbaciones de la observada en años comparables al azar.

### H3a diagnóstica — prueba histórica basada solo en I
Se conserva para auditoría y comparación, pero ya no representa por sí sola el veredicto de H3 sistémica.
- Eventos evaluables: 3; efecto I: +7.02 pp; p diagnóstica = 0.0990
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
